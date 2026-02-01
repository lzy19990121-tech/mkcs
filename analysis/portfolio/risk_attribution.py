"""
SPL-5b-B: 风险归因与分摊

实现策略对组合风险的贡献分解，以及基于风险评分的预算分配。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.portfolio.risk_budget import PortfolioRiskBudget, StrategyBudget, BudgetUnit, BudgetAllocation

# Optional imports for type hints
try:
    from analysis.risk_envelope import RiskEnvelope
    from analysis.structural_analysis import StructuralAnalysisResult, RiskPatternType
    from analysis.stability_analysis import StabilityReport
    from analysis.regime_features import RegimeFeatures
except ImportError:
    RiskEnvelope = None
    StructuralAnalysisResult = None
    RiskPatternType = None
    StabilityReport = None
    RegimeFeatures = None


@dataclass
class StrategyContribution:
    """策略贡献"""

    strategy_id: str
    contribution_ratio: float     # 贡献比例（0-1）
    absolute_contribution: float  # 绝对贡献值
    window_count: int             # 贡献窗口数

    # 统计
    mean_contribution: float
    max_contribution: float
    min_contribution: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "contribution_ratio": self.contribution_ratio,
            "absolute_contribution": self.absolute_contribution,
            "window_count": self.window_count,
            "mean_contribution": self.mean_contribution,
            "max_contribution": self.max_contribution,
            "min_contribution": self.min_contribution
        }


@dataclass
class CoCrashPair:
    """协同爆炸对"""

    strategy_1: str
    strategy_2: str
    correlation: float           # 压力期相关性
    co_crash_count: int          # 同时亏损次数
    total_windows: int           # 总窗口数

    # 统计
    co_crash_rate: float = field(init=False)

    def __post_init__(self):
        self.co_crash_rate = self.co_crash_count / self.total_windows if self.total_windows > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_1": self.strategy_1,
            "strategy_2": self.strategy_2,
            "correlation": self.correlation,
            "co_crash_count": self.co_crash_count,
            "total_windows": self.total_windows,
            "co_crash_rate": self.co_crash_rate
        }


@dataclass
class StrategyRiskProfile:
    """策略风险画像"""

    strategy_id: str

    # 风险评分（0-100，越高越风险）
    risk_score: float

    # 分项得分
    envelope_score: float        # Envelope worst-case (0-40分)
    structural_score: float      # Structural label (0-30分)
    stability_score: float       # Stability (0-20分)
    regime_score: float          # Regime sensitivity (0-10分)

    # 详细指标
    worst_return: float          # 最坏收益
    risk_pattern: str            # 风险模式
    stability_rating: float      # 稳定性评分
    regime_variance: float       # Regime方差

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "risk_score": self.risk_score,
            "envelope_score": self.envelope_score,
            "structural_score": self.structural_score,
            "stability_score": self.stability_score,
            "regime_score": self.regime_score,
            "worst_return": self.worst_return,
            "risk_pattern": self.risk_pattern,
            "stability_rating": self.stability_rating,
            "regime_variance": self.regime_variance
        }


def decompose_strategy_contributions(
    strategy_returns: Dict[str, pd.Series],
    strategy_weights: Dict[str, float],
    worst_window_indices: List[int] = None
) -> Dict[str, StrategyContribution]:
    """分解策略对组合worst-case的贡献

    Args:
        strategy_returns: {strategy_id: 收益序列}
        strategy_weights: {strategy_id: 权重}
        worst_window_indices: 最坏窗口索引列表（None则使用全部）

    Returns:
        {strategy_id: StrategyContribution}
    """
    if not strategy_returns:
        return {}

    # 获取共同的索引
    common_index = strategy_returns[list(strategy_returns.keys())[0]].index
    for returns in strategy_returns.values():
        common_index = common_index.intersection(returns.index)

    # 确定要分析的窗口
    if worst_window_indices is None:
        window_indices = range(len(common_index))
    else:
        window_indices = worst_window_indices

    # 计算每个策略的贡献
    contributions = defaultdict(list)
    absolute_contributions = defaultdict(float)

    for idx in window_indices:
        if idx >= len(common_index):
            continue

        # 获取该时间点的收益
        window_contribution = {}
        for strat_id, returns in strategy_returns.items():
            if idx < len(returns):
                weight = strategy_weights.get(strat_id, 0)
                ret = returns.iloc[idx]
                contribution = weight * ret
                window_contribution[strat_id] = contribution

        # 累加负贡献
        for strat_id, contrib in window_contribution.items():
            if contrib < 0:
                contributions[strat_id].append(contrib)
                absolute_contributions[strat_id] += abs(contrib)

    # 计算总贡献
    total = sum(absolute_contributions.values())

    # 归一化并创建结果
    result = {}
    for strat_id, contribs in contributions.items():
        contrib_array = np.array(contribs)

        contribution_ratio = (
            absolute_contributions[strat_id] / total
            if total > 0 else 0
        )

        result[strat_id] = StrategyContribution(
            strategy_id=strat_id,
            contribution_ratio=contribution_ratio,
            absolute_contribution=absolute_contributions[strat_id],
            window_count=len(contribs),
            mean_contribution=float(np.mean(contrib_array)) if len(contribs) > 0 else 0,
            max_contribution=float(np.max(contrib_array)) if len(contribs) > 0 else 0,
            min_contribution=float(np.min(contrib_array)) if len(contribs) > 0 else 0
        )

    return result


def identify_co_crash_pairs(
    strategy_returns: Dict[str, pd.Series],
    correlation_threshold: float = 0.7,
    window_size: int = 20
) -> List[CoCrashPair]:
    """识别协同爆炸对

    Args:
        strategy_returns: {strategy_id: 收益序列}
        correlation_threshold: 相关性阈值
        window_size: 滚动窗口大小

    Returns:
        [CoCrashPair]
    """
    if len(strategy_returns) < 2:
        return []

    # 获取共同的索引
    common_index = strategy_returns[list(strategy_returns.keys())[0]].index
    for returns in strategy_returns.values():
        common_index = common_index.intersection(returns.index)

    # 构建DataFrame
    df = pd.DataFrame({strat_id: returns.loc[common_index] for strat_id, returns in strategy_returns.items()})

    # 滚动窗口分析
    co_crash_data = defaultdict(lambda: {"count": 0, "total": 0, "correlations": []})

    for i in range(window_size, len(df)):
        window_df = df.iloc[i-window_size:i]

        # 计算相关性矩阵
        corr_matrix = window_df.corr()

        # 检查每对策略
        strat_ids = list(strategy_returns.keys())
        for i, strat1 in enumerate(strat_ids):
            for j, strat2 in enumerate(strat_ids):
                if i >= j:  # 避免重复和自相关
                    continue

                corr = corr_matrix.loc[strat1, strat2]
                pair_key = tuple(sorted([strat1, strat2]))

                # 检查是否同时亏损
                mean1 = window_df[strat1].mean()
                mean2 = window_df[strat2].mean()

                if corr > correlation_threshold and mean1 < 0 and mean2 < 0:
                    co_crash_data[pair_key]["count"] += 1
                    co_crash_data[pair_key]["correlations"].append(corr)

                co_crash_data[pair_key]["total"] += 1

    # 生成结果
    result = []
    for (strat1, strat2), data in co_crash_data.items():
        if data["count"] > 0:
            mean_corr = np.mean(data["correlations"])
            result.append(CoCrashPair(
                strategy_1=strat1,
                strategy_2=strat2,
                correlation=mean_corr,
                co_crash_count=data["count"],
                total_windows=data["total"]
            ))

    # 按协同爆炸率排序
    result.sort(key=lambda x: x.co_crash_rate, reverse=True)

    return result


def calculate_strategy_risk_score(
    strategy_id: str,
    envelope: RiskEnvelope,
    structural_result: StructuralAnalysisResult,
    stability_report: StabilityReport,
    regime_sensitivity: Optional[Dict[str, float]] = None
) -> StrategyRiskProfile:
    """计算策略风险评分（0-100，越高越风险）

    维度：
    1. Envelope worst-case (40分)
    2. Structural label (30分)
    3. Stability (20分)
    4. Regime sensitivity (10分)

    Args:
        strategy_id: 策略ID
        envelope: 风险包络
        structural_result: 结构分析结果
        stability_report: 稳定性报告
        regime_sensitivity: {regime: return} 各regime下的收益

    Returns:
        StrategyRiskProfile
    """
    score = 0

    # 1. Envelope (0-40分)
    worst_return = envelope.return_p95
    envelope_score = min(40, abs(worst_return) * 200)  # -20% → 40分
    score += envelope_score

    # 2. Structural (0-30分)
    risk_pattern = structural_result.risk_pattern_type.value
    if risk_pattern == "structural":
        structural_score = 30  # 结构性风险
    elif risk_pattern == "single_outlier":
        structural_score = 10  # 单一异常
    else:
        structural_score = 15  # 其他
    score += structural_score

    # 3. Stability (0-20分)
    stability_rating = stability_report.stability_score
    stability_score = min(20, (100 - stability_rating) * 0.2)
    score += stability_score

    # 4. Regime sensitivity (0-10分)
    if regime_sensitivity:
        regime_variance = np.var(list(regime_sensitivity.values()))
        regime_score = min(10, regime_variance * 100)
    else:
        regime_variance = 0
        regime_score = 0
    score += regime_score

    # 总分限制在0-100
    total_score = min(100, max(0, score))

    return StrategyRiskProfile(
        strategy_id=strategy_id,
        risk_score=total_score,
        envelope_score=envelope_score,
        structural_score=structural_score,
        stability_score=stability_score,
        regime_score=regime_score,
        worst_return=worst_return,
        risk_pattern=risk_pattern,
        stability_rating=stability_rating,
        regime_variance=regime_variance
    )


def allocate_initial_budget(
    strategy_profiles: Dict[str, StrategyRiskProfile],
    total_budget: PortfolioRiskBudget,
    co_crash_pairs: List[CoCrashPair] = None
) -> BudgetAllocation:
    """基于风险评分分配初始预算

    规则：
    - 高风险(>60) → 预算↓
    - 低风险(<40) → 预算↑
    - 协同对 → 互斥约束

    Args:
        strategy_profiles: {strategy_id: StrategyRiskProfile}
        total_budget: 总预算
        co_crash_pairs: 协同爆炸对列表

    Returns:
        BudgetAllocation
    """
    if not strategy_profiles:
        return BudgetAllocation(total_budget=total_budget, strategy_budgets={})

    # 1. 计算基础权重（风险越高权重越低）
    base_weights = {}
    for strat_id, profile in strategy_profiles.items():
        score = profile.risk_score
        if score > 60:
            weight = 0.5  # 高风险减半
        elif score < 40:
            weight = 1.5  # 低风险增加
        else:
            weight = 1.0  # 中等风险保持
        base_weights[strat_id] = weight

    # 2. 协同对约束
    if co_crash_pairs:
        co_crash_map = defaultdict(list)
        for pair in co_crash_pairs:
            co_crash_map[pair.strategy_1].append((pair.strategy_2, pair.co_crash_rate))
            co_crash_map[pair.strategy_2].append((pair.strategy_1, pair.co_crash_rate))

        # 对高协同风险的策略降低权重
        for strat_id, co_crashes in co_crash_map.items():
            if strat_id in base_weights:
                max_crash_rate = max(rate for _, rate in co_crashes)
                if max_crash_rate > 0.3:  # 协同爆炸率>30%
                    base_weights[strat_id] *= 0.7

    # 3. 归一化权重
    total_weight = sum(base_weights.values())
    normalized_weights = {
        strat_id: w / total_weight
        for strat_id, w in base_weights.items()
    }

    # 4. 创建策略预算
    strategy_budgets = {}
    for strat_id, profile in strategy_profiles.items():
        weight = normalized_weights[strat_id]

        # 应用预算约束
        weight = max(
            total_budget.min_weight_per_strategy,
            min(total_budget.max_weight_per_strategy, weight)
        )

        # 检查是否应该禁用（风险极高）
        disabled = profile.risk_score > 80

        # 计算尾部损失贡献（简化）
        tail_loss_contribution = profile.worst_return * weight

        strategy_budgets[strat_id] = StrategyBudget(
            strategy_id=strat_id,
            allocated_weight=weight if not disabled else 0,
            risk_score=profile.risk_score,
            tail_loss_contribution=tail_loss_contribution,
            weight_cap=total_budget.max_weight_per_strategy,
            disabled=disabled
        )

    # 5. 重新归一化（如果有禁用的策略）
    active_weights = sum(
        b.allocated_weight for b in strategy_budgets.values()
        if not b.disabled
    )
    if active_weights < 1.0 and active_weights > 0:
        for budget in strategy_budgets.values():
            if not budget.disabled:
                budget.allocated_weight /= active_weights

    # 6. 创建分配结果
    allocation = BudgetAllocation(
        total_budget=total_budget,
        strategy_budgets=strategy_budgets
    )
    allocation.calculate_totals()

    return allocation


if __name__ == "__main__":
    """测试代码"""
    print("=== RiskAttribution 测试 ===\n")

    # 创建模拟数据
    np.random.seed(42)
    n_samples = 100

    strategy_returns = {
        "strategy_1": pd.Series(np.random.randn(n_samples) * 0.01),
        "strategy_2": pd.Series(np.random.randn(n_samples) * 0.015),
        "strategy_3": pd.Series(np.random.randn(n_samples) * 0.008)
    }

    strategy_weights = {
        "strategy_1": 0.4,
        "strategy_2": 0.4,
        "strategy_3": 0.2
    }

    # 测试1: 贡献分解
    print("1. 策略贡献分解:")
    contributions = decompose_strategy_contributions(strategy_returns, strategy_weights)

    for strat_id, contrib in contributions.items():
        print(f"   {strat_id}:")
        print(f"     贡献比例: {contrib.contribution_ratio*100:.1f}%")
        print(f"     窗口数: {contrib.window_count}")

    # 测试2: 协同爆炸对识别
    print("\n2. 协同爆炸对识别:")
    # 创建相关的策略对
    strategy_returns_corr = {
        "strategy_a": pd.Series(np.random.randn(n_samples) * 0.01),
        "strategy_b": pd.Series(np.random.randn(n_samples) * 0.01 + 0.5 * np.random.randn(n_samples) * 0.01)
    }
    # 添加一些同时亏损的时期
    for i in range(20, 30):
        strategy_returns_corr["strategy_a"].iloc[i] = -0.02
        strategy_returns_corr["strategy_b"].iloc[i] = -0.015

    co_crash_pairs = identify_co_crash_pairs(strategy_returns_corr, correlation_threshold=0.5)

    for pair in co_crash_pairs[:3]:
        print(f"   {pair.strategy_1} - {pair.strategy_2}:")
        print(f"     相关性: {pair.correlation:.3f}")
        print(f"     协同爆炸率: {pair.co_crash_rate*100:.1f}%")

    # 测试3: 初始预算分配
    print("\n3. 初始预算分配:")
    from analysis.portfolio.risk_budget import create_moderate_budget

    budget = create_moderate_budget()

    # 创建模拟的风险画像
    profiles = {
        "strategy_1": StrategyRiskProfile(
            strategy_id="strategy_1",
            risk_score=45.0,
            envelope_score=20.0,
            structural_score=10.0,
            stability_score=10.0,
            regime_score=5.0,
            worst_return=-0.08,
            risk_pattern="single_outlier",
            stability_rating=70.0,
            regime_variance=0.05
        ),
        "strategy_2": StrategyRiskProfile(
            strategy_id="strategy_2",
            risk_score=65.0,
            envelope_score=30.0,
            structural_score=20.0,
            stability_score=10.0,
            regime_score=5.0,
            worst_return=-0.15,
            risk_pattern="structural",
            stability_rating=50.0,
            regime_variance=0.08
        )
    }

    allocation = allocate_initial_budget(profiles, budget)

    print(f"   总权重: {allocation.total_weight:.2f}")
    print(f"   活跃策略: {allocation.active_strategies}")
    for strat_id, strat_budget in allocation.strategy_budgets.items():
        print(f"   {strat_id}:")
        print(f"     权重: {strat_budget.allocated_weight*100:.1f}%")
        print(f"     风险评分: {strat_budget.risk_score:.1f}")

    print("\n✓ 测试完成")
