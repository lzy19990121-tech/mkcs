#!/usr/bin/env python3
"""
SPL-6b Optimizer Regression Gate

检查优化器回归：
1. Risk budget non-regression（组合 worst-case 不突破风险预算）
2. Correlation spike guard（压力期相关性不超阈值）
3. Co-crash guard（同时尾部亏损策略数不超阈值）
4. Optimizer stability guard（权重抖动不超过阈值）

FAIL 行为：
- 任何 FAIL → 退出码 1（阻断 CI）
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.replay_schema import load_replay_outputs, ReplayOutput
from analysis.optimization_problem import OptimizationProblem
from analysis.optimization_risk_proxies import RiskProxyCalculator
from analysis.portfolio_optimizer_v2 import PortfolioOptimizerV2
from analysis.pipeline_optimizer_v2 import PipelineOptimizerV2, PipelineConfig


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    message: str
    details: Dict[str, Any] = None


@dataclass
class RunManifest:
    """运行清单"""
    timestamp: str
    commit_hash: str
    block_release: bool
    tests: List[TestResult]


def calculate_portfolio_metrics(
    weights: Dict[str, float],
    replays: List[ReplayOutput]
) -> Dict[str, float]:
    """计算组合指标

    Args:
        weights: 策略权重字典
        replays: 回测数据列表

    Returns:
        组合指标字典
    """
    # 构建收益矩阵
    returns_matrix = []
    strategy_ids = []

    for replay in replays:
        df = replay.to_dataframe()
        if 'step_pnl' in df.columns:
            returns_matrix.append(df['step_pnl'].values)
            strategy_ids.append(replay.strategy_id)

    if not returns_matrix:
        return {
            "total_return": 0.0,
            "cvar_95": 0.0,
            "cvar_99": 0.0,
            "max_drawdown": 0.0,
            "tail_correlation": 0.0,
            "co_crash_count": 0
        }

    # 找到最小长度并填充
    min_length = min(len(r) for r in returns_matrix)
    returns_matrix = np.array([r[:min_length] for r in returns_matrix]).T

    # 构建权重向量
    weights_array = np.array([weights.get(sid, 0.0) for sid in strategy_ids])

    # 计算组合收益
    portfolio_returns = returns_matrix @ weights_array

    # 计算 CVaR
    def calculate_cvar(returns, confidence=0.95):
        var = np.percentile(returns, (1 - confidence) * 100)
        tail_losses = returns[returns <= var]
        return tail_losses.mean() if len(tail_losses) > 0 else var

    # 计算最大回撤
    cumulative = np.cumsum(portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-10)
    max_dd = abs(drawdown.min())

    # 计算尾部相关性（压力期）
    stress_mask = portfolio_returns < -0.02
    tail_correlation = 0.0
    if stress_mask.sum() > 1:
        stress_returns = returns_matrix[stress_mask]
        corr_matrix = np.corrcoef(stress_returns.T)
        upper_tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
        tail_correlation = np.abs(upper_tri).mean()

    # 计算 co-crash 次数
    co_crash_count = 0
    for i in range(len(portfolio_returns)):
        losses = returns_matrix[i] < -0.02
        if losses.sum() >= 2:
            co_crash_count += 1

    return {
        "total_return": float(portfolio_returns.sum()),
        "cvar_95": float(calculate_cvar(portfolio_returns, 0.95)),
        "cvar_99": float(calculate_cvar(portfolio_returns, 0.99)),
        "max_drawdown": float(max_dd),
        "tail_correlation": float(tail_correlation),
        "co_crash_count": int(co_crash_count)
    }


def test_risk_budget_non_regression(
    optimizer_metrics: Dict[str, float],
    risk_budgets: Dict[str, float]
) -> TestResult:
    """测试 1: Risk budget non-regression

    组合 worst-case 不突破风险预算
    """
    details = {
        "optimizer_metrics": optimizer_metrics,
        "risk_budgets": risk_budgets
    }

    # 检查 P95 return (CVaR-95)
    cvar_95_budget = risk_budgets.get("cvar_95_budget", -0.10)  # -10%
    if optimizer_metrics["cvar_95"] < cvar_95_budget:
        return TestResult(
            test_name="risk_budget_non_regression",
            status="FAIL",
            message=f"CVaR-95 ({optimizer_metrics['cvar_95']:.4f}) 突破预算 ({cvar_95_budget:.4f})",
            details=details
        )

    # 检查 P99 return (CVaR-99)
    cvar_99_budget = risk_budgets.get("cvar_99_budget", -0.15)  # -15%
    if optimizer_metrics["cvar_99"] < cvar_99_budget:
        return TestResult(
            test_name="risk_budget_non_regression",
            status="FAIL",
            message=f"CVaR-99 ({optimizer_metrics['cvar_99']:.4f}) 突破预算 ({cvar_99_budget:.4f})",
            details=details
        )

    # 检查最大回撤
    max_dd_budget = risk_budgets.get("max_drawdown_budget", 0.12)  # 12%
    if optimizer_metrics["max_drawdown"] > max_dd_budget:
        return TestResult(
            test_name="risk_budget_non_regression",
            status="FAIL",
            message=f"Max DD ({optimizer_metrics['max_drawdown']:.2%}) 突破预算 ({max_dd_budget:.2%})",
            details=details
        )

    return TestResult(
        test_name="risk_budget_non_regression",
        status="PASS",
        message="Risk budgets satisfied",
        details=details
    )


def test_correlation_spike_guard(
    optimizer_metrics: Dict[str, float],
    threshold: float = 0.5
) -> TestResult:
    """测试 2: Correlation spike guard

    压力期相关性不超过阈值
    """
    details = {
        "tail_correlation": optimizer_metrics["tail_correlation"],
        "threshold": threshold
    }

    if optimizer_metrics["tail_correlation"] > threshold:
        return TestResult(
            test_name="correlation_spike_guard",
            status="FAIL",
            message=f"Tail correlation ({optimizer_metrics['tail_correlation']:.3f}) 超过阈值 ({threshold:.3f})",
            details=details
        )

    return TestResult(
        test_name="correlation_spike_guard",
        status="PASS",
        message=f"Tail correlation ({optimizer_metrics['tail_correlation']:.3f}) within threshold",
        details=details
    )


def test_co_crash_guard(
    optimizer_metrics: Dict[str, float],
    max_co_crash: int = 2
) -> TestResult:
    """测试 3: Co-crash guard

    同时尾部亏损策略数不超过阈值
    """
    details = {
        "co_crash_count": optimizer_metrics["co_crash_count"],
        "max_co_crash": max_co_crash
    }

    if optimizer_metrics["co_crash_count"] > max_co_crash:
        return TestResult(
            test_name="co_crash_guard",
            status="FAIL",
            message=f"Co-crash count ({optimizer_metrics['co_crash_count']}) 超过阈值 ({max_co_crash})",
            details=details
        )

    return TestResult(
        test_name="co_crash_guard",
        status="PASS",
        message=f"Co-crash count ({optimizer_metrics['co_crash_count']}) within threshold",
        details=details
    )


def test_optimizer_stability_guard(
    current_weights: Dict[str, float],
    previous_weights: Dict[str, float],
    max_change: float = 0.2
) -> TestResult:
    """测试 4: Optimizer stability guard

    权重抖动不超过阈值
    """
    max_actual_change = 0.0
    changes = []

    for strategy_id in current_weights:
        if strategy_id in previous_weights:
            change = abs(current_weights[strategy_id] - previous_weights[strategy_id])
            changes.append(change)
            max_actual_change = max(max_actual_change, change)

    details = {
        "max_weight_change": max_actual_change,
        "threshold": max_change,
        "all_changes": changes
    }

    if max_actual_change > max_change:
        return TestResult(
            test_name="optimizer_stability_guard",
            status="FAIL",
            message=f"Max weight change ({max_actual_change:.2%}) 超过阈值 ({max_change:.2%})",
            details=details
        )

    return TestResult(
        test_name="optimizer_stability_guard",
        status="PASS",
        message=f"Max weight change ({max_actual_change:.2%}) within threshold",
        details=details
    )


def get_commit_hash() -> str:
    """获取 git commit hash"""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()[:8]
    except:
        return "unknown"


def main():
    """主测试流程"""
    print("="*70)
    print("SPL-6b: Optimizer Regression Gate")
    print("="*70)
    print(f"时间: {datetime.now().isoformat()}")

    # 加载数据
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("⚠️  runs/ 目录不存在，跳过测试")
        return 0

    replays = load_replay_outputs(str(runs_dir))
    if len(replays) < 2:
        print("⚠️  数据不足（<2 个策略），跳过测试")
        return 0

    strategy_ids = [r.strategy_id for r in replays]

    # 风险预算定义
    risk_budgets = {
        "cvar_95_budget": -0.10,  # -10%
        "cvar_99_budget": -0.15,  # -15%
        "max_drawdown_budget": 0.12  # 12%
    }

    # 运行优化器
    print(f"\n策略数量: {len(strategy_ids)}")
    print("运行优化器...")

    try:
        # 创建 pipeline
        config = PipelineConfig(enable_gating=False, enable_optimizer=True)
        pipeline = PipelineOptimizerV2(strategy_ids, config)

        # 运行 pipeline
        result = pipeline.run_pipeline(replays)

        if not result.success:
            print(f"⚠️  Pipeline 失败: {result.fallback_reason}")

        # 计算组合指标
        optimizer_metrics = calculate_portfolio_metrics(result.weights, replays)

        print("\n优化器指标:")
        for key, value in optimizer_metrics.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ 优化器运行失败: {e}")
        # 创建失败结果
        optimizer_metrics = {
            "total_return": 0.0,
            "cvar_95": -1.0,  # 故意失败
            "cvar_99": -1.0,
            "max_drawdown": 1.0,
            "tail_correlation": 1.0,
            "co_crash_count": 100
        }
        result = None

    # 运行测试
    print("\n" + "="*70)
    print("运行回归测试")
    print("="*70)

    tests = []

    # Test 1: Risk budget non-regression
    test1 = test_risk_budget_non_regression(optimizer_metrics, risk_budgets)
    tests.append(test1)
    print(f"\n[{test1.status}] {test1.test_name}")
    print(f"  {test1.message}")

    # Test 2: Correlation spike guard
    test2 = test_correlation_spike_guard(optimizer_metrics, threshold=0.5)
    tests.append(test2)
    print(f"\n[{test2.status}] {test2.test_name}")
    print(f"  {test2.message}")

    # Test 3: Co-crash guard
    test3 = test_co_crash_guard(optimizer_metrics, max_co_crash=2)
    tests.append(test3)
    print(f"\n[{test3.status}] {test3.test_name}")
    print(f"  {test3.message}")

    # Test 4: Optimizer stability guard（如果有历史权重）
    if result and result.previous_weights:
        test4 = test_optimizer_stability_guard(
            result.weights,
            result.previous_weights,
            max_change=0.2
        )
        tests.append(test4)
        print(f"\n[{test4.status}] {test4.test_name}")
        print(f"  {test4.message}")
    else:
        print(f"\n[SKIP] optimizer_stability_guard")
        print("  无历史权重数据")

    # 汇总结果
    fail_count = sum(1 for t in tests if t.status == "FAIL")
    pass_count = sum(1 for t in tests if t.status == "PASS")
    skip_count = sum(1 for t in tests if t.status == "SKIP")

    # 创建 RunManifest
    manifest = RunManifest(
        timestamp=datetime.now().isoformat(),
        commit_hash=get_commit_hash(),
        block_release=(fail_count > 0),
        tests=tests
    )

    # 保存结果
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = output_dir / "spl6b_optimizer_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump({
            "timestamp": manifest.timestamp,
            "commit_hash": manifest.commit_hash,
            "block_release": manifest.block_release,
            "tests": [
                {
                    "name": t.test_name,
                    "status": t.status,
                    "message": t.message,
                    "details": t.details
                }
                for t in manifest.tests
            ]
        }, f, indent=2, default=str)

    print("\n" + "="*70)
    print("测试汇总")
    print("="*70)
    print(f"Total: {len(tests)}")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")
    print(f"SKIP: {skip_count}")
    print(f"Block Release: {'🚫 YES' if manifest.block_release else '✅ NO'}")
    print(f"\n结果已保存: {manifest_file}")

    # 返回退出码
    if fail_count > 0:
        print("\n❌ CI Gate FAILED - 有 FAIL 测试")
        return 1
    else:
        print("\n✅ CI Gate PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
