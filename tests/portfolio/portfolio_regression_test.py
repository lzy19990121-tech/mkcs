"""
SPL-5b-E: 组合回归测试

实现组合级回归测试，确保风险预算不被突破。
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np

from analysis.portfolio.risk_budget import PortfolioRiskBudget, BudgetAllocation
from analysis.portfolio.backtest_validator import (
    PortfolioWindowMetrics,
    BudgetHitReport,
    SynergyReductionMetrics,
    scan_portfolio_worst_cases,
    generate_budget_hit_report,
    detect_synergy_reduction
)
from analysis.replay_schema import ReplayOutput


@dataclass
class PortfolioTestResult:
    """组合测试结果"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    message: str
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "status": self.status,
            "message": self.message,
            "details": self.details or {}
        }


class PortfolioRegressionTests:
    """SPL-5b 组合回归测试

    Test Suite:
    1. C4.1: 组合包络非退化测试
    2. C4.2: 相关性激增守卫测试
    3. C4.3: 协同爆炸计数守卫测试
    4. C4.4: 预算违规检测测试
    """

    def __init__(
        self,
        tolerance_pct: float = 0.05
    ):
        """初始化测试套件

        Args:
            tolerance_pct: 容忍度（默认5%）
        """
        self.tolerance_pct = tolerance_pct

    def test_portfolio_envelope_non_regression(
        self,
        baseline_budget: PortfolioRiskBudget,
        current_windows: Dict[str, List[PortfolioWindowMetrics]],
        baseline_envelope: Optional[Dict[str, Dict[str, float]]] = None
    ) -> PortfolioTestResult:
        """C4.1: 组合包络非退化测试

        验证组合worst-case不应退化。

        Args:
            baseline_budget: 基线预算
            current_windows: 当前最坏窗口
            baseline_envelope: 基线包络指标（可选）

        Returns:
            PortfolioTestResult
        """
        try:
            details = {}
            violations = []

            for window_length, windows in current_windows.items():
                if not windows:
                    continue

                # 计算当前包络指标
                current_returns = [w.portfolio_return for w in windows]
                current_mdds = [w.portfolio_mdd for w in windows]
                current_durations = [w.portfolio_duration for w in windows]

                # P95指标
                return_p95 = np.percentile(current_returns, 5)
                mdd_p95 = np.percentile(current_mdds, 95)
                duration_p95 = int(np.percentile(current_durations, 95))

                details[window_length] = {
                    "return_p95": return_p95,
                    "mdd_p95": mdd_p95,
                    "duration_p95": duration_p95,
                    "budget_return_p95": baseline_budget.budget_return_p95,
                    "budget_mdd_p95": baseline_budget.budget_mdd_p95,
                    "budget_duration_p95": baseline_budget.budget_duration_p95
                }

                # 检查是否突破预算
                if return_p95 < baseline_budget.budget_return_p95 * (1 - self.tolerance_pct):
                    violations.append({
                        "window": window_length,
                        "metric": "return_p95",
                        "current": return_p95,
                        "budget": baseline_budget.budget_return_p95,
                        "diff": return_p95 - baseline_budget.budget_return_p95
                    })

                if mdd_p95 > baseline_budget.budget_mdd_p95 * (1 + self.tolerance_pct):
                    violations.append({
                        "window": window_length,
                        "metric": "mdd_p95",
                        "current": mdd_p95,
                        "budget": baseline_budget.budget_mdd_p95,
                        "diff": mdd_p95 - baseline_budget.budget_mdd_p95
                    })

                if duration_p95 > baseline_budget.budget_duration_p95 * 1.1:  # 10% tolerance
                    violations.append({
                        "window": window_length,
                        "metric": "duration_p95",
                        "current": duration_p95,
                        "budget": baseline_budget.budget_duration_p95,
                        "diff": duration_p95 - baseline_budget.budget_duration_p95
                    })

                # 与基线对比（如果有）
                if baseline_envelope and window_length in baseline_envelope:
                    base_env = baseline_envelope[window_length]
                    base_return_p95 = base_env.get("return_p95", 0)

                    if return_p95 < base_return_p95 * (1 - self.tolerance_pct):
                        violations.append({
                            "window": window_length,
                            "metric": "return_p95_vs_baseline",
                            "current": return_p95,
                            "baseline": base_return_p95,
                            "diff": return_p95 - base_return_p95
                        })

            if violations:
                return PortfolioTestResult(
                    test_name="portfolio_envelope_non_regression",
                    status="FAIL",
                    message=f"组合包络退化: {len(violations)}项指标突破预算",
                    details={"violations": violations, **details}
                )

            return PortfolioTestResult(
                test_name="portfolio_envelope_non_regression",
                status="PASS",
                message="组合包络指标在预算范围内",
                details=details
            )

        except Exception as e:
            return PortfolioTestResult(
                test_name="portfolio_envelope_non_regression",
                status="FAIL",
                message=f"测试错误: {str(e)}",
                details={"error": str(e)}
            )

    def test_correlation_spike_guard(
        self,
        current_windows: Dict[str, List[PortfolioWindowMetrics]],
        max_allowed_correlation: float = 0.9
    ) -> PortfolioTestResult:
        """C4.2: 相关性激增守卫测试

        验证压力期相关性上升不应突破阈值。

        Args:
            current_windows: 当前最坏窗口
            max_allowed_correlation: 允许的最大相关性

        Returns:
            PortfolioTestResult
        """
        try:
            details = {}
            spikes = []

            for window_length, windows in current_windows.items():
                high_correlation_count = 0
                max_correlation = 0

                for w in windows:
                    if w.correlation_matrix:
                        # 计算最大相关性
                        for strat1, corr_dict in w.correlation_matrix.items():
                            for strat2, corr in corr_dict.items():
                                if strat1 == strat2:
                                    continue
                                if corr > max_correlation:
                                    max_correlation = corr
                                if corr > max_allowed_correlation:
                                    high_correlation_count += 1

                details[window_length] = {
                    "max_correlation": max_correlation,
                    "high_correlation_count": high_correlation_count,
                    "max_allowed": max_allowed_correlation
                }

                if max_correlation > max_allowed_correlation:
                    spikes.append({
                        "window": window_length,
                        "max_correlation": max_correlation,
                        "threshold": max_allowed_correlation,
                        "exceed_by": max_correlation - max_allowed_correlation
                    })

            if spikes:
                return PortfolioTestResult(
                    test_name="correlation_spike_guard",
                    status="FAIL",
                    message=f"相关性突破阈值: 最大相关性 {spikes[0]['max_correlation']:.3f} > {max_allowed_correlation}",
                    details={"spikes": spikes, **details}
                )

            return PortfolioTestResult(
                test_name="correlation_spike_guard",
                status="PASS",
                message=f"压力期相关性在阈值内: 最大相关性 < {max_allowed_correlation}",
                details=details
            )

        except Exception as e:
            return PortfolioTestResult(
                test_name="correlation_spike_guard",
                status="FAIL",
                message=f"测试错误: {str(e)}",
                details={"error": str(e)}
            )

    def test_co_crash_count_guard(
        self,
        current_windows: Dict[str, List[PortfolioWindowMetrics]],
        max_simultaneous_losses: int = 2
    ) -> PortfolioTestResult:
        """C4.3: 协同爆炸计数守卫测试

        验证同时爆的策略数应受限。

        Args:
            current_windows: 当前最坏窗口
            max_simultaneous_losses: 允许同时尾部亏损的最大策略数

        Returns:
            PortfolioTestResult
        """
        try:
            details = {}
            breaches = []

            for window_length, windows in current_windows.items():
                exceed_count = 0
                max_simultaneous = 0

                for w in windows:
                    simultaneous = w.simultaneous_tail_losses
                    if simultaneous > max_simultaneous:
                        exceed_count += 1
                    if simultaneous > max_simultaneous:
                        max_simultaneous = simultaneous

                details[window_length] = {
                    "max_simultaneous_losses": max_simultaneous,
                    "exceed_count": exceed_count,
                    "max_allowed": max_simultaneous_losses
                }

                if max_simultaneous > max_simultaneous_losses:
                    breaches.append({
                        "window": window_length,
                        "max_simultaneous": max_simultaneous,
                        "threshold": max_simultaneous_losses,
                        "exceed_by": max_simultaneous - max_simultaneous_losses,
                        "exceed_count": exceed_count
                    })

            if breaches:
                return PortfolioTestResult(
                    test_name="co_crash_count_guard",
                    status="FAIL",
                    message=f"协同爆炸突破阈值: {breaches[0]['max_simultaneous']}个策略同时亏损 > {max_simultaneous_losses}",
                    details={"breaches": breaches, **details}
                )

            return PortfolioTestResult(
                test_name="co_crash_count_guard",
                status="PASS",
                message=f"协同爆炸在阈值内: 同时尾部亏损策略数 <= {max_simultaneous_losses}",
                details=details
            )

        except Exception as e:
            return PortfolioTestResult(
                test_name="co_crash_count_guard",
                status="FAIL",
                message=f"测试错误: {str(e)}",
                details={"error": str(e)}
            )

    def test_budget_breach_detection(
        self,
        budget_hit_report: BudgetHitReport,
        allow_breaches: bool = False
    ) -> PortfolioTestResult:
        """C4.4: 预算违规检测测试

        验证预算违规应被检测并记录。

        Args:
            budget_hit_report: 预算命中报告
            allow_breaches: 是否允许违规（False则违规导致FAIL）

        Returns:
            PortfolioTestResult
        """
        try:
            breached_budgets = [
                name for name, hit in budget_hit_report.budgets_hit.items()
                if hit
            ]

            details = {
                "breached_budgets": breached_budgets,
                "budget_hit_rate": budget_hit_report.budget_hit_rate,
                "total_checks": budget_hit_report.total_checks,
                "capped_strategies": budget_hit_report.capped_strategies,
                "date_range": {
                    "start": budget_hit_report.start_date.isoformat(),
                    "end": budget_hit_report.end_date.isoformat()
                }
            }

            if breached_budgets and not allow_breaches:
                return PortfolioTestResult(
                    test_name="budget_breach_detection",
                    status="FAIL",
                    message=f"预算违规: {', '.join(breached_budgets)}",
                    details=details
                )

            return PortfolioTestResult(
                test_name="budget_breach_detection",
                status="PASS",
                message=f"预算检查完成: {len(breached_budgets)}个违规, 命中率={budget_hit_report.budget_hit_rate*100:.1f}%",
                details=details
            )

        except Exception as e:
            return PortfolioTestResult(
                test_name="budget_breach_detection",
                status="FAIL",
                message=f"测试错误: {str(e)}",
                details={"error": str(e)}
            )


def run_portfolio_regression_tests(
    budget: PortfolioRiskBudget,
    current_windows: Dict[str, List[PortfolioWindowMetrics]],
    budget_hit_report: BudgetHitReport,
    baseline_envelope: Optional[Dict[str, Dict[str, float]]] = None,
    max_correlation: float = 0.9,
    max_simultaneous_losses: int = 2
) -> List[PortfolioTestResult]:
    """运行所有组合回归测试

    Args:
        budget: 风险预算
        current_windows: 当前最坏窗口
        budget_hit_report: 预算命中报告
        baseline_envelope: 基线包络（可选）
        max_correlation: 最大允许相关性
        max_simultaneous_losses: 最大同时亏损策略数

    Returns:
        [PortfolioTestResult]
    """
    test_suite = PortfolioRegressionTests()

    results = []

    # C4.1: 组合包络非退化
    results.append(test_suite.test_portfolio_envelope_non_regression(
        budget, current_windows, baseline_envelope
    ))

    # C4.2: 相关性激增守卫
    results.append(test_suite.test_correlation_spike_guard(
        current_windows, max_correlation
    ))

    # C4.3: 协同爆炸计数守卫
    results.append(test_suite.test_co_crash_count_guard(
        current_windows, max_simultaneous_losses
    ))

    # C4.4: 预算违规检测
    results.append(test_suite.test_budget_breach_detection(
        budget_hit_report
    ))

    return results


def save_portfolio_baseline(
    budget: PortfolioRiskBudget,
    output_path: str = "baselines/portfolio/portfolio_budget_spec.json"
) -> None:
    """保存组合预算基线

    Args:
        budget: 预算配置
        output_path: 输出路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    baseline = {
        "version": "portfolio_v1",
        "created_at": datetime.now().isoformat(),
        "budget": budget.to_dict()
    }

    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2, default=str)

    print(f"✓ 预算基线已保存到: {output_path}")


def load_portfolio_baseline(
    baseline_path: str = "baselines/portfolio/portfolio_budget_spec.json"
) -> Optional[PortfolioRiskBudget]:
    """加载组合预算基线

    Args:
        baseline_path: 基线路径

    Returns:
        PortfolioRiskBudget
    """
    path = Path(baseline_path)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    return PortfolioRiskBudget.from_dict(data["budget"])


if __name__ == "__main__":
    """测试代码"""
    print("=== PortfolioRegressionTests 测试 ===\n")

    import pandas as pd
    from analysis.portfolio.budget_allocator import AllocationResult
    from analysis.portfolio.risk_budget import create_moderate_budget

    # 创建模拟数据
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="D")

    strategy_returns = {
        "strategy_1": pd.Series(np.random.randn(n_samples) * 0.01, index=dates),
        "strategy_2": pd.Series(np.random.randn(n_samples) * 0.015, index=dates),
        "strategy_3": pd.Series(np.random.randn(n_samples) * 0.008, index=dates)
    }

    strategy_weights = {
        "strategy_1": 0.4,
        "strategy_2": 0.4,
        "strategy_3": 0.2
    }

    allocation = AllocationResult(
        target_weights=strategy_weights.copy(),
        weight_caps={},
        disabled_strategies=[],
        allocation_reasons={},
        applied_rules=[]
    )

    # 扫描最坏窗口
    from analysis.portfolio.backtest_validator import scan_portfolio_worst_cases, generate_budget_hit_report

    worst_windows = scan_portfolio_worst_cases(
        strategy_returns, strategy_weights, allocation,
        window_lengths=["20d"], top_k=5
    )

    # 生成预算报告
    budget = create_moderate_budget()
    hit_report = generate_budget_hit_report(budget, worst_windows, [allocation])

    # 运行测试
    print("运行组合回归测试:")
    results = run_portfolio_regression_tests(
        budget, worst_windows, hit_report
    )

    for result in results:
        print(f"\n{result.test_name}:")
        print(f"  状态: {result.status}")
        print(f"  消息: {result.message}")

    # 统计
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")

    print(f"\n总计: {passed} PASS, {failed} FAIL, {skipped} SKIP")

    # 保存预算基线
    print("\n保存预算基线:")
    save_portfolio_baseline(budget, "/tmp/test_portfolio_budget.json")

    print("\n✓ 测试完成")
