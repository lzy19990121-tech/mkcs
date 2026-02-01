"""
SPL-5b-D: 组合回测与扫描

实现组合最坏窗口扫描和预算使用验证。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from collections import defaultdict

from analysis.portfolio.risk_budget import PortfolioRiskBudget, BudgetAllocation, StrategyBudget
from analysis.portfolio.budget_allocator import AllocationResult, RuleBasedAllocator
from analysis.window_scanner import WindowScanner, WindowMetrics
from analysis.replay_schema import ReplayOutput
from analysis.regime_features import RegimeFeatures


@dataclass
class PortfolioWindowMetrics:
    """组合窗口指标"""

    window_id: str
    window_length: str
    start_date: datetime
    end_date: datetime

    # 组合级指标
    portfolio_return: float      # 组合收益
    portfolio_mdd: float         # 组合最大回撤
    portfolio_duration: int      # 回撤持续天数

    # 策略级指标
    strategy_returns: Dict[str, float]     # {strategy_id: return}
    strategy_weights: Dict[str, float]     # {strategy_id: weight}
    strategy_contributions: Dict[str, float]  # {strategy_id: contribution}

    # 协同风险指标
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    simultaneous_tail_losses: int = 0     # 同时尾部亏损的策略数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "window_length": self.window_length,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "portfolio_return": self.portfolio_return,
            "portfolio_mdd": self.portfolio_mdd,
            "portfolio_duration": self.portfolio_duration,
            "strategy_returns": self.strategy_returns,
            "strategy_weights": self.strategy_weights,
            "strategy_contributions": self.strategy_contributions,
            "simultaneous_tail_losses": self.simultaneous_tail_losses
        }


@dataclass
class BudgetHitReport:
    """预算使用报告"""

    # 预算是否被触达
    budgets_hit: Dict[str, bool]

    # 经常顶到cap的策略
    capped_strategies: Dict[str, int]  # {strategy_id: hit_count}

    # 触发预算收缩的regime
    contraction_triggers: List[str]

    # 统计
    total_checks: int
    budget_hit_rate: float

    # 时间范围
    start_date: datetime
    end_date: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budgets_hit": self.budgets_hit,
            "capped_strategies": self.capped_strategies,
            "contraction_triggers": self.contraction_triggers,
            "total_checks": self.total_checks,
            "budget_hit_rate": self.budget_hit_rate,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat()
        }


@dataclass
class SynergyReductionMetrics:
    """协同风险削弱指标"""

    co_crash_count_reduction: int       # 协同爆炸事件减少数量
    simultaneous_tail_loss_reduction: float  # 同时尾部亏损减少比例
    correlation_spike_reduction: int    # 相关性激增事件减少数量

    baseline_co_crash_count: int
    current_co_crash_count: int

    baseline_simultaneous_losses: float
    current_simultaneous_losses: float

    baseline_correlation_spikes: int
    current_correlation_spikes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "co_crash_count_reduction": self.co_crash_count_reduction,
            "simultaneous_tail_loss_reduction": self.simultaneous_tail_loss_reduction,
            "correlation_spike_reduction": self.correlation_spike_reduction,
            "baseline_co_crash_count": self.baseline_co_crash_count,
            "current_co_crash_count": self.current_co_crash_count,
            "baseline_simultaneous_losses": self.baseline_simultaneous_losses,
            "current_simultaneous_losses": self.current_simultaneous_losses,
            "baseline_correlation_spikes": self.baseline_correlation_spikes,
            "current_correlation_spikes": self.current_correlation_spikes
        }


def scan_portfolio_worst_cases(
    strategy_returns: Dict[str, pd.Series],
    strategy_weights: Dict[str, float],
    allocation_result: AllocationResult,
    window_lengths: List[str] = ["20d", "60d"],
    top_k: int = 5
) -> Dict[str, List[PortfolioWindowMetrics]]:
    """扫描组合最坏窗口

    Args:
        strategy_returns: {strategy_id: 收益序列}
        strategy_weights: {strategy_id: 基础权重}
        allocation_result: 分配结果
        window_lengths: 窗口长度列表
        top_k: 保留top-k最坏窗口

    Returns:
        {window_length: [PortfolioWindowMetrics]}
    """
    # 应用分配结果
    adjusted_weights = {}
    for strat_id in strategy_returns:
        if strat_id in allocation_result.target_weights:
            adjusted_weights[strat_id] = allocation_result.target_weights[strat_id]
        else:
            adjusted_weights[strat_id] = strategy_weights.get(strat_id, 0)

    # 计算组合收益序列
    common_index = strategy_returns[list(strategy_returns.keys())[0]].index
    for returns in strategy_returns.values():
        common_index = common_index.intersection(returns.index)

    portfolio_returns = pd.Series(0.0, index=common_index)
    for strat_id, returns in strategy_returns.items():
        portfolio_returns += returns.loc[common_index] * adjusted_weights[strat_id]

    # 扫描每个窗口长度
    worst_windows = {}
    scanner = WindowScanner()

    for window_length in window_lengths:
        window_days = scanner.parse_window_length(window_length)

        windows = []
        for i in range(len(portfolio_returns) - window_days + 1):
            window_portfolio = portfolio_returns.iloc[i:i + window_days]

            # 计算组合指标
            portfolio_return = window_portfolio.sum()
            portfolio_mdd = calculate_max_drawdown(window_portfolio)
            portfolio_duration = calculate_drawdown_duration(window_portfolio)

            # 计算策略级指标
            strategy_window_returns = {}
            strategy_contributions = {}
            for strat_id in strategy_returns:
                if strat_id in strategy_returns:
                    strat_window = strategy_returns[strat_id].iloc[i:i + window_days]
                    strategy_window_returns[strat_id] = strat_window.sum()
                    strategy_contributions[strat_id] = (
                        strat_window.sum() * adjusted_weights[strat_id]
                    )

            # 计算协同风险指标
            correlation_matrix = calculate_window_correlation_matrix(
                strategy_returns, window_days, i
            )
            simultaneous_losses = count_simultaneous_tail_losses(
                strategy_returns, window_days, i
            )

            window_metrics = PortfolioWindowMetrics(
                window_id=f"portfolio_{window_length}_{i}",
                window_length=window_length,
                start_date=window_portfolio.index[0],
                end_date=window_portfolio.index[-1],
                portfolio_return=portfolio_return,
                portfolio_mdd=portfolio_mdd,
                portfolio_duration=portfolio_duration,
                strategy_returns=strategy_window_returns,
                strategy_weights=adjusted_weights.copy(),
                strategy_contributions=strategy_contributions,
                correlation_matrix=correlation_matrix,
                simultaneous_tail_losses=simultaneous_losses
            )

            windows.append(window_metrics)

        # 按组合收益排序，取最坏窗口
        windows.sort(key=lambda w: w.portfolio_return)
        worst_windows[window_length] = windows[:top_k]

    return worst_windows


def calculate_max_drawdown(series: pd.Series) -> float:
    """计算最大回撤"""
    if len(series) == 0:
        return 0.0

    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return abs(drawdown.min())


def calculate_drawdown_duration(series: pd.Series) -> int:
    """计算回撤持续天数"""
    if len(series) == 0:
        return 0

    cummax = series.cummax()
    drawdown = (series - cummax) / cummax

    # 找到最大回撤点
    max_dd_idx = drawdown.idxmin()
    max_dd_value = drawdown.min()

    # 从最大回撤点开始，计算恢复时间
    recovery_mask = drawdown.loc[max_dd_idx:] >= -0.01  # 接近0视为恢复

    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        duration = (recovery_idx - max_dd_idx).days
    else:
        duration = len(series) - series.index.get_loc(max_dd_idx)

    return duration


def calculate_window_correlation_matrix(
    strategy_returns: Dict[str, pd.Series],
    window_days: int,
    start_idx: int
) -> Dict[str, Dict[str, float]]:
    """计算窗口相关性矩阵"""
    end_idx = start_idx + window_days

    window_data = {}
    for strat_id, returns in strategy_returns.items():
        if start_idx < len(returns) and end_idx <= len(returns):
            window_data[strat_id] = returns.iloc[start_idx:end_idx]

    if len(window_data) < 2:
        return {}

    df = pd.DataFrame(window_data)
    corr_matrix = df.corr()

    # 转换为字典
    result = {}
    for strat1 in corr_matrix.columns:
        result[strat1] = {}
        for strat2 in corr_matrix.columns:
            result[strat1][strat2] = float(corr_matrix.loc[strat1, strat2])

    return result


def count_simultaneous_tail_losses(
    strategy_returns: Dict[str, pd.Series],
    window_days: int,
    start_idx: int,
    tail_threshold: float = -0.05  # -5%
) -> int:
    """计算同时尾部亏损的策略数"""
    end_idx = start_idx + window_days

    tail_loss_strategies = 0
    for strat_id, returns in strategy_returns.items():
        if start_idx < len(returns) and end_idx <= len(returns):
            window_return = returns.iloc[start_idx:end_idx].sum()
            if window_return < tail_threshold:
                tail_loss_strategies += 1

    return tail_loss_strategies


def detect_synergy_reduction(
    baseline_worst: List[PortfolioWindowMetrics],
    budget_worst: List[PortfolioWindowMetrics]
) -> SynergyReductionMetrics:
    """检测协同风险是否被削弱"""
    # 统计协同爆炸事件
    baseline_co_crashes = sum(
        1 for w in baseline_worst
        if w.simultaneous_tail_losses >= 2
    )
    budget_co_crashes = sum(
        1 for w in budget_worst
        if w.simultaneous_tail_losses >= 2
    )

    # 统计同时尾部亏损比例
    baseline_simultaneous = np.mean([
        w.simultaneous_tail_losses for w in baseline_worst
    ]) if baseline_worst else 0
    budget_simultaneous = np.mean([
        w.simultaneous_tail_losses for w in budget_worst
    ]) if budget_worst else 0

    # 统计相关性激增事件（相关性>0.8）
    baseline_correlation_spikes = sum(
        1 for w in baseline_worst
        if w.correlation_matrix and count_high_correlations(w.correlation_matrix) > 0
    )
    budget_correlation_spikes = sum(
        1 for w in budget_worst
        if w.correlation_matrix and count_high_correlations(w.correlation_matrix) > 0
    )

    return SynergyReductionMetrics(
        co_crash_count_reduction=baseline_co_crashes - budget_co_crashes,
        simultaneous_tail_loss_reduction=baseline_simultaneous - budget_simultaneous,
        correlation_spike_reduction=baseline_correlation_spikes - budget_correlation_spikes,
        baseline_co_crash_count=baseline_co_crashes,
        current_co_crash_count=budget_co_crashes,
        baseline_simultaneous_losses=baseline_simultaneous,
        current_simultaneous_losses=budget_simultaneous,
        baseline_correlation_spikes=baseline_correlation_spikes,
        current_correlation_spikes=budget_correlation_spikes
    )


def count_high_correlations(
    correlation_matrix: Dict[str, Dict[str, float]],
    threshold: float = 0.8
) -> int:
    """计算高相关性对数量"""
    count = 0
    checked = set()

    for strat1, corr_dict in correlation_matrix.items():
        for strat2, corr in corr_dict.items():
            if strat1 == strat2:
                continue
            pair = tuple(sorted([strat1, strat2]))
            if pair in checked:
                continue
            checked.add(pair)

            if corr > threshold:
                count += 1

    return count


def generate_budget_hit_report(
    budget: PortfolioRiskBudget,
    worst_windows: Dict[str, List[PortfolioWindowMetrics]],
    allocations: List[AllocationResult]
) -> BudgetHitReport:
    """生成预算使用报告"""
    budgets_hit = {
        "return_p95": False,
        "mdd_p95": False,
        "duration_p95": False
    }

    capped_strategies = defaultdict(int)
    contraction_triggers = []

    # 检查每个窗口
    total_checks = 0
    budget_hits = 0

    for window_length, windows in worst_windows.items():
        for window in windows:
            total_checks += 1

            # 检查P95收益预算
            if window.portfolio_return < budget.budget_return_p95:
                budgets_hit["return_p95"] = True
                budget_hits += 1

            # 检查P95 MDD预算
            if window.portfolio_mdd > budget.budget_mdd_p95:
                budgets_hit["mdd_p95"] = True
                budget_hits += 1

            # 检查P95持续期预算
            if window.portfolio_duration > budget.budget_duration_p95:
                budgets_hit["duration_p95"] = True
                budget_hits += 1

            # 统计顶到cap的策略
            for allocation in allocations:
                for strat_id, cap in allocation.weight_caps.items():
                    weight = allocation.target_weights.get(strat_id, 0)
                    if weight >= cap * 0.95:  # 95% usage
                        capped_strategies[strat_id] += 1

    # 获取时间范围
    all_dates = []
    for windows in worst_windows.values():
        for w in windows:
            all_dates.append(w.start_date)
            all_dates.append(w.end_date)

    start_date = min(all_dates) if all_dates else datetime.now()
    end_date = max(all_dates) if all_dates else datetime.now()

    return BudgetHitReport(
        budgets_hit=budgets_hit,
        capped_strategies=dict(capped_strategies),
        contraction_triggers=contraction_triggers,
        total_checks=total_checks,
        budget_hit_rate=budget_hits / total_checks if total_checks > 0 else 0,
        start_date=start_date,
        end_date=end_date
    )


if __name__ == "__main__":
    """测试代码"""
    print("=== BacktestValidator 测试 ===\n")

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

    # 创建模拟分配结果
    from analysis.portfolio.budget_allocator import AllocationResult

    allocation = AllocationResult(
        target_weights=strategy_weights.copy(),
        weight_caps={"strategy_2": 0.5},
        disabled_strategies=[],
        allocation_reasons={},
        applied_rules=["rule_1", "rule_2"]
    )

    # 测试1: 扫描最坏窗口
    print("1. 扫描组合最坏窗口:")
    worst_windows = scan_portfolio_worst_cases(
        strategy_returns,
        strategy_weights,
        allocation,
        window_lengths=["20d"],
        top_k=3
    )

    for window_length, windows in worst_windows.items():
        print(f"   {window_length} 窗口:")
        for w in windows[:2]:
            print(f"     {w.window_id}:")
            print(f"       组合收益: {w.portfolio_return*100:.2f}%")
            print(f"       组合回撤: {w.portfolio_mdd*100:.2f}%")
            print(f"       同时尾部亏损策略数: {w.simultaneous_tail_losses}")

    # 测试2: 协同风险削弱检测
    print("\n2. 协同风险削弱检测:")
    # 创建一个"改进的"窗口（降低协同爆炸）
    improved_windows = [
        replace(w, simultaneous_tail_losses=max(0, w.simultaneous_tail_losses - 1))
        for w in worst_windows["20d"]
    ]

    synergy_metrics = detect_synergy_reduction(
        worst_windows["20d"],
        improved_windows
    )

    print(f"   协同爆炸事件减少: {synergy_metrics.co_crash_count_reduction}")
    print(f"   同时尾部亏损减少: {synergy_metrics.simultaneous_tail_loss_reduction:.2f}")

    # 测试3: 预算使用报告
    print("\n3. 预算使用报告:")
    from analysis.portfolio.risk_budget import create_moderate_budget

    budget = create_moderate_budget()
    hit_report = generate_budget_hit_report(
        budget,
        worst_windows,
        [allocation]
    )

    print(f"   预算命中情况: {hit_report.budgets_hit}")
    print(f"   总检查次数: {hit_report.total_checks}")
    print(f"   预算命中率: {hit_report.budget_hit_rate*100:.1f}%")
    print(f"   顶到cap的策略: {hit_report.capped_strategies}")

    print("\n✓ 测试完成")
