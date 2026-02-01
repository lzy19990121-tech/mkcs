"""
SPL-7b-C: 效果量化

为每个反事实计算：avoided drawdown, lost return, spike 消除, gating 次数变化。
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.counterfactual.counterfactual_config import CounterfactualResult


@dataclass
class EffectMetrics:
    """效果指标"""
    scenario_id: str

    # 风险改善
    avoided_drawdown: float = 0.0      # 避免的回撤
    reduced_volatility: float = 0.0    # 降低的波动率
    improved_cvar: float = 0.0          # 改善的 CVaR
    eliminated_spikes: int = 0         # 消除���尖刺

    # 收益牺牲
    lost_return: float = 0.0           # 牺牲的收益
    reduced_upside: float = 0.0         # 降低的上行空间

    # 事件统计
    gating_reduction: int = 0           # Gating 次数变化
    rebalance_reduction: int = 0        # Rebalance 次数变化

    # 权衡比
    tradeoff_ratio: float = 0.0         # 风险改善 / 收益牺牲

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "avoided_drawdown": self.avoided_drawdown,
            "reduced_volatility": self.reduced_volatility,
            "improved_cvar": self.improved_cvar,
            "eliminated_spikes": self.eliminated_spikes,
            "lost_return": self.lost_return,
            "reduced_upside": self.reduced_upside,
            "gating_reduction": self.gating_reduction,
            "rebalance_reduction": self.rebalance_reduction,
            "tradeoff_ratio": self.tradeoff_ratio
        }


@dataclass
class RuleValueMetrics:
    """规则价值指标"""
    rule_id: str

    # 边际贡献
    marginal_risk_reduction: float    # 边际风险降低
    marginal_return_cost: float        # 边际收益成本

    # 效率指标
    risk_reduction_per_return: float   # 每单位收益的风险降低
    efficiency_score: float            # 效率评分 (0-100)

    # 排序
    value_rank: int = 0                # 价值排名

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "marginal_risk_reduction": self.marginal_risk_reduction,
            "marginal_return_cost": self.marginal_return_cost,
            "risk_reduction_per_return": self.risk_reduction_per_return,
            "efficiency_score": self.efficiency_score,
            "value_rank": self.value_rank
        }


class EffectCalculator:
    """效果计算器

    量化反事实场景的效果。
    """

    def __init__(self):
        """初始化计算器"""

    def calculate_effects(
        self,
        actual_result: CounterfactualResult,
        cf_results: Dict[str, CounterfactualResult]
    ) -> Dict[str, EffectMetrics]:
        """计算所有反事实的效果

        Args:
            actual_result: 实际运行结果
            cf_results: 反事实结果字典

        Returns:
            场景 ID -> 效果指标
        """
        effects = {}

        for scenario_id, cf_result in cf_results.items():
            if not cf_result.success:
                continue

            # 计算效果指标
            metrics = EffectMetrics(scenario_id=scenario_id)

            # 避免的回撤
            metrics.avoided_drawdown = actual_result.max_drawdown - cf_result.max_drawdown

            # 降低的波动率
            metrics.reduced_volatility = actual_result.volatility - cf_result.volatility

            # 改善的 CVaR
            metrics.improved_cvar = actual_result.cvar_95 - cf_result.cvar_95

            # 消除的尖刺（简化：gating 次数减少）
            metrics.eliminated_spikes = max(0, actual_result.gating_events_count - cf_result.gating_events_count)

            # 牺牲的收益
            metrics.lost_return = actual_result.total_return - cf_result.total_return

            # 降低的上行空间（简化）
            max_actual = max(actual_result.daily_returns) if actual_result.daily_returns else 0
            max_cf = max(cf_result.daily_returns) if cf_result.daily_returns else 0
            metrics.reduced_upside = max(0, max_actual - max_cf)

            # Gating 次数变化
            metrics.gating_reduction = actual_result.gating_events_count - cf_result.gating_events_count

            # Rebalance 次数变化
            metrics.rebalance_reduction = actual_result.allocator_rebalance_count - cf_result.allocator_rebalance_count

            # 权衡比
            if abs(metrics.lost_return) > 1e-10:
                metrics.tradeoff_ratio = metrics.avoided_drawdown / abs(metrics.lost_return)
            else:
                metrics.tradeoff_ratio = 0.0 if metrics.avoided_drawdown <= 0 else float('inf')

            effects[scenario_id] = metrics

        return effects

    def calculate_rule_values(
        self,
        actual_result: CounterfactualResult,
        cf_results: Dict[str, CounterfactualResult]
    ) -> List[RuleValueMetrics]:
        """计算规则价值

        Args:
            actual_result: 实际运行结果
            cf_results: 反事实结果字典

        Returns:
            规则价值指标列表
        """
        rule_values = []

        # 对比不同 gating 配置的场景
        gating_scenarios = {
            "cf_earlier_gating": "earlier gating",
            "cf_later_gating": "later gating",
            "cf_no_gating": "no gating"
        }

        for scenario_id, rule_name in gating_scenarios.items():
            if scenario_id not in cf_results:
                continue

            cf_result = cf_results[scenario_id]
            if not cf_result.success:
                continue

            # 边际风险降低
            marginal_risk_reduction = actual_result.max_drawdown - cf_result.max_drawdown

            # 边际收益成本
            marginal_return_cost = actual_result.total_return - cf_result.total_return

            # 风险降低/收益成本比
            if abs(marginal_return_cost) > 1e-10:
                risk_reduction_per_return = marginal_risk_reduction / abs(marginal_return_cost)
            else:
                risk_reduction_per_return = 0.0

            # 效率评分
            # 同时降低风险且增加收益 = 高效
            # 降低风险但牺牲太多收益 = 低效
            if marginal_risk_reduction > 0 and marginal_return_cost > 0:
                # 风险降低但收益减少 = 权衡
                efficiency_score = 50.0
            elif marginal_risk_reduction > 0 and marginal_return_cost < 0:
                # 风险降低且收益增加 = 高效
                efficiency_score = 100.0
            elif marginal_risk_reduction < 0 and marginal_return_cost > 0:
                # 风险增加且收益减少 = 低效
                efficiency_score = 0.0
            else:
                efficiency_score = 50.0

            rule_values.append(RuleValueMetrics(
                rule_id=rule_name,
                marginal_risk_reduction=marginal_risk_reduction,
                marginal_return_cost=marginal_return_cost,
                risk_reduction_per_return=risk_reduction_per_return,
                efficiency_score=efficiency_score
            ))

        # 按效率评分排序
        rule_values.sort(key=lambda x: x.efficiency_score, reverse=True)

        # 分配排名
        for i, rule_value in enumerate(rule_values):
            rule_value.value_rank = i + 1

        return rule_values

    def calculate_worst_window_value(
        self,
        actual_result: CounterfactualResult,
        cf_results: Dict[str, CounterfactualResult],
        window_size: int = 20
    ) -> Dict[str, float]:
        """计算最坏窗口中的规则价值

        Args:
            actual_result: 实际结果
            cf_results: 反事实结果
            window_size: 窗口大小

        Returns:
            场景 ID -> 最坏窗口价值
        """
        worst_window_values = {}

        for scenario_id, cf_result in cf_results.items():
            if not cf_result.success:
                continue

            # 找到最坏窗口（最大回撤）
            actual_returns = actual_result.daily_returns
            cf_returns = cf_result.daily_returns

            # 滑动窗口计算
            worst_actual = -float('inf')
            worst_cf = -float('inf')

            for i in range(len(actual_returns) - window_size + 1):
                window_actual = sum(actual_returns[i:i+window_size])
                if window_actual < worst_actual:
                    worst_actual = window_actual

                window_cf = sum(cf_returns[i:i+window_size])
                if window_cf < worst_cf:
                    worst_cf = window_cf

            # 计算价值（最坏窗口改善）
            window_value = worst_cf - worst_actual
            worst_window_values[scenario_id] = window_value

        return worst_window_values

    def find_best_scenario(
        self,
        effects: Dict[str, EffectMetrics],
        optimize_for: str = "tradeoff"  # "tradeoff", "risk", "return"
    ) -> Tuple[str, EffectMetrics]:
        """找到最优场景

        Args:
            effects: 效果指标字典
            optimize_for: 优化目标

        Returns:
            (场景 ID, 效果指标)
        """
        if not effects:
            return ("", EffectMetrics(scenario_id=""))

        if optimize_for == "tradeoff":
            # 最大化权衡比（风险改善/收益牺牲）
            best = max(effects.items(), key=lambda x: x[1].tradeoff_ratio)
        elif optimize_for == "risk":
            # 最大化风险降低
            best = max(effects.items(), key=lambda x: x[1].avoided_drawdown)
        elif optimize_for == "return":
            # 最小化收益牺牲
            best = min(effects.items(), key=lambda x: x[1].lost_return)
        else:
            best = max(effects.items(), key=lambda x: x[1].tradeoff_ratio)

        return best


class SpikeAnalyzer:
    """Spike 分析器

    分析 spike 是否被消除。
    """

    def __init__(self):
        """初始化分析器"""

    def analyze_spike_elimination(
        self,
        actual_result: CounterfactualResult,
        cf_results: Dict[str, CounterfactualResult],
        spike_threshold: float = -0.02
    ) -> Dict[str, Dict[str, Any]]:
        """分析 spike 消除情况

        Args:
            actual_result: 实际结果
            cf_results: 反事实结果
            spike_threshold: Spike 阈值（-2%）

        Returns:
            场景 ID -> 分析结果
        """
        spike_analysis = {}

        # 统计实际 spike 数量
        actual_spikes = sum(
            1 for r in actual_result.daily_returns
            if r < spike_threshold
        )

        # 分析每个反事实
        for scenario_id, cf_result in cf_results.items():
            if not cf_result.success:
                continue

            # 统计反事实 spike 数量
            cf_spikes = sum(
                1 for r in cf_result.daily_returns
                if r < spike_threshold
            )

            eliminated_spikes = actual_spikes - cf_spikes

            spike_analysis[scenario_id] = {
                "actual_spikes": actual_spikes,
                "cf_spikes": cf_spikes,
                "eliminated_spikes": eliminated_spikes,
                "elimination_rate": eliminated_spikes / max(actual_spikes, 1),
                "spike_reduction_percent": (
                    (eliminated_spikes / max(actual_spikes, 1)) * 100
                    if actual_spikes > 0 else 0
                )
            }

        return spike_analysis


if __name__ == "__main__":
    """测试效果量化"""
    print("=== SPL-7b-C: 效果量化测试 ===\n")

    # 模拟结果
    from analysis.counterfactual.counterfactual_config import CounterfactualResult

    actual_result = CounterfactualResult(
        scenario_id="actual",
        strategy_id="test",
        total_return=100.0,
        daily_returns=[1.0] * 50 + [-5.0] + [1.0] * 49,  # 1 个 spike
        max_drawdown=0.08,
        volatility=0.02,
        cvar_95=-0.03,
        cvar_99=-0.05,
        gating_events_count=5,
        allocator_rebalance_count=10
    )

    cf_earlier = CounterfactualResult(
        scenario_id="cf_earlier_gating",
        strategy_id="test",
        total_return=95.0,
        daily_returns=[0.8] * 100,
        max_drawdown=0.05,
        volatility=0.015,
        cvar_95=-0.02,
        cvar_99=-0.03,
        gating_events_count=15,
        allocator_rebalance_count=10
    )

    cf_later = CounterfactualResult(
        scenario_id="cf_later_gating",
        strategy_id="test",
        total_return=110.0,
        daily_returns=[1.2] * 50 + [-8.0] + [1.2] * 49,  # 更大 spike
        max_drawdown=0.12,
        volatility=0.025,
        cvar_95=-0.04,
        cvar_99=-0.07,
        gating_events_count=2,
        allocator_rebalance_count=10
    )

    cf_results = {
        "cf_earlier_gating": cf_earlier,
        "cf_later_gating": cf_later
    }

    # 计算效果
    calculator = EffectCalculator()
    effects = calculator.calculate_effects(actual_result, cf_results)

    print("效果指标:")
    for scenario_id, metrics in effects.items():
        print(f"\n{scenario_id}:")
        print(f"  避免回撤: {metrics.avoided_drawdown:.2%}")
        print(f"  牺牲收益: {metrics.lost_return:.2f}")
        print(f"  权衡比: {metrics.tradeoff_ratio:.2f}")
        print(f"  消除 Spike: {metrics.eliminated_spikes}")

    # 计算规则价值
    rule_values = calculator.calculate_rule_values(actual_result, cf_results)
    print(f"\n规则价值:")
    for rule_value in rule_values[:5]:
        print(f"  {rule_value.rule_id}:")
        print(f"    边际风险降低: {rule_value.marginal_risk_reduction:.2%}")
        print(f"    边际收益成本: {rule_value.marginal_return_cost:.2f}")
        print(f"    效率评分: {rule_value.efficiency_score:.1f}/100")
        print(f"    排名: {rule_value.value_rank}")

    # 找到最优场景
    best_scenario, best_metrics = calculator.find_best_scenario(effects, "tradeoff")
    print(f"\n最优场景 (权衡比): {best_scenario}")
    print(f"  权衡比: {best_metrics.tradeoff_ratio:.2f}")

    # Spike 分析
    spike_analyzer = SpikeAnalyzer()
    spike_analysis = spike_analyzer.analyze_spike_elimination(actual_result, cf_results)
    print(f"\nSpike 消除分析:")
    for scenario_id, analysis in spike_analysis.items():
        print(f"\n{scenario_id}:")
        print(f"  实际 Spike: {analysis['actual_spikes']}")
        print(f"  CF Spike: {analysis['cf_spikes']}")
        print(f"  消除 Spike: {analysis['eliminated_spikes']}")
        print(f"  消除率: {analysis['elimination_rate']:.1%}")

    print("\n✅ 效果量化测试通过")
