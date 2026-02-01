#!/usr/bin/env python3
"""
CI 集成测试：运行 SPL-5 风险回归测试

FAIL 行为：
- 任何 FAIL → 退出码 1（阻断 CI）
- 生成 RunManifest 标记 block_release=true
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.replay_schema import load_replay_outputs, ReplayOutput
from analysis.window_scanner import WindowScanner


@dataclass
class TestResult:
    """Result of a regression test"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    message: str
    details: Dict[str, Any] = None


def calculate_basic_metrics(replay: ReplayOutput) -> Dict[str, float]:
    """计算基本风险指标"""
    df = replay.to_dataframe()

    if df.empty:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "worst_20d_return": 0.0,
            "total_steps": 0
        }

    step_pnl = df['step_pnl'].values
    total_return = step_pnl.sum()

    # 计算最大回撤
    cummax = np.maximum.accumulate(step_pnl.cumsum())
    equity = 100000 + step_pnl.cumsum()
    drawdown = (cummax - equity) / cummax
    max_dd = drawdown.max()

    # 波动率
    volatility = np.std(step_pnl) if len(step_pnl) > 0 else 0

    # Worst 20d return
    scanner = WindowScanner()
    windows_20d = scanner.scan_replay(replay, "20d")
    worst_20d = min([w.total_return for w in windows_20d]) if windows_20d else 0.0

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "volatility": volatility,
        "worst_20d_return": worst_20d,
        "total_steps": len(replay.steps)
    }


def test_basic_regression(baseline: ReplayOutput, current: ReplayOutput) -> TestResult:
    """基本回归测试：比较核心指标"""
    baseline_metrics = calculate_basic_metrics(baseline)
    current_metrics = calculate_basic_metrics(current)

    details = {
        "baseline": baseline_metrics,
        "current": current_metrics
    }

    # 检查基本指标的显著劣化
    tolerance_pct = 0.10  # 10% 容差

    # 1. Total return 不能显著降低
    return_delta = (current_metrics['total_return'] - baseline_metrics['total_return'])
    if return_delta < -abs(baseline_metrics['total_return']) * tolerance_pct:
        return TestResult(
            test_name="basic_regression",
            status="FAIL",
            message=f"Total return 劣化: {return_delta:.2f} (容差: {abs(baseline_metrics['total_return']) * tolerance_pct:.2f})",
            details=details
        )

    # 2. Max drawdown 不能显著增加
    mdd_delta = current_metrics['max_drawdown'] - baseline_metrics['max_drawdown']
    if not np.isnan(mdd_delta) and not np.isnan(baseline_metrics['max_drawdown']):
        if mdd_delta > baseline_metrics['max_drawdown'] * tolerance_pct:
            return TestResult(
                test_name="basic_regression",
                status="FAIL",
                message=f"Max drawdown 增加: {mdd_delta*100:.2f}% (baseline: {baseline_metrics['max_drawdown']*100:.2f}%)",
                details=details
            )

    # 3. Worst 20d return 不能显著降低
    worst_20d_delta = current_metrics['worst_20d_return'] - baseline_metrics['worst_20d_return']
    if worst_20d_delta < -0.05:  # 5% 绝对阈值
        return TestResult(
            test_name="basic_regression",
            status="FAIL",
            message=f"Worst 20d return 劣化: {worst_20d_delta*100:.2f}%",
            details=details
        )

    return TestResult(
        test_name="basic_regression",
        status="PASS",
        message=f"基本指标正常: Return={current_metrics['total_return']:.2f}, MDD={current_metrics['max_drawdown']*100:.2f}%",
        details=details
    )


def run_ci_gate():
    """运行 CI gate 测试"""
    print("=" * 60)
    print("SPL-5 CI Gate: 风险回归测试")
    print("=" * 60)

    # 加载 baseline
    runs_dir = str(project_root / "runs")
    all_replays = load_replay_outputs(runs_dir)

    if len(all_replays) < 2:
        print(f"[SKIP] 需要至少2个replay，当前: {len(all_replays)}")
        return 0

    baseline = all_replays[0]
    current = all_replays[1]

    print(f"\nBaseline: {baseline.run_id}")
    print(f"Current: {current.run_id}")

    # 运行测试套件
    manifest = {
        "run_date": datetime.now().isoformat(),
        "baseline_run_id": baseline.run_id,
        "current_run_id": current.run_id,
        "tests": []
    }

    all_passed = True

    # 1. Basic regression test
    print("\n" + "-" * 60)
    print("测试 1: Basic Regression Test")
    print("-" * 60)

    test_result = test_basic_regression(baseline, current)
    manifest["tests"].append({
        "name": test_result.test_name,
        "status": test_result.status,
        "message": test_result.message
    })

    if test_result.status == "FAIL":
        all_passed = False
        print(f"[FAIL] {test_result.message}")
    elif test_result.status == "SKIP":
        print(f"[SKIP] {test_result.message}")
    else:
        print(f"[PASS] {test_result.message}")

    # 生成 manifest
    manifest["block_release"] = not all_passed

    manifest_path = Path("reports/run_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"总测试数: {len(manifest['tests'])}")
    print(f"通过: {sum(1 for t in manifest['tests'] if t['status'] == 'PASS')}")
    print(f"失败: {sum(1 for t in manifest['tests'] if t['status'] == 'FAIL')}")
    print(f"跳过: {sum(1 for t in manifest['tests'] if t['status'] == 'SKIP')}")
    print(f"\nBlock Release: {manifest['block_release']}")
    print(f"Manifest: {manifest_path}")
    print("=" * 60)

    if all_passed:
        print("\n✅ CI Gate PASSED")
        return 0
    else:
        print("\n❌ CI Gate FAILED - 阻断发布")
        return 1


if __name__ == "__main__":
    exit_code = run_ci_gate()
    sys.exit(exit_code)
