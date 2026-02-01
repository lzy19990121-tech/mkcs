"""
SPL-5b-C: 动态预算分配器

实现基于规则的预算分配器，根据市场状态和策略状态动态调整预算分配。
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import numpy as np

from analysis.portfolio.risk_budget import PortfolioRiskBudget, StrategyBudget, BudgetAllocation
from analysis.regime_features import RegimeFeatures


class AllocationRuleType(Enum):
    """分配规则类型"""
    VOLATILITY_ADJUSTMENT = "volatility_adjustment"      # 波动率调整
    TREND_FILTER = "trend_filter"                        # 趋势过滤
    CO_CRASH_EXCLUSION = "co_crash_exclusion"            # 协同爆炸互斥
    DRAWDOWN_PROTECTION = "drawdown_protection"          # 回撤保护
    BUDGET_CONSTRAINT = "budget_constraint"              # 预算约束


@dataclass
class AllocationRule:
    """分配规则定义"""
    rule_id: str
    rule_name: str
    rule_type: AllocationRuleType
    priority: int                # 优先级（数字越小越优先）
    enabled: bool = True
    description: str = ""

    def apply(
        self,
        regime: RegimeFeatures,
        strategy_states: Dict[str, "StrategyState"],
        result: "AllocationResult"
    ) -> None:
        """应用规则

        Args:
            regime: 当前市场状态
            strategy_states: 策略状态
            result: 分配结果（会被修改）
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_type": self.rule_type.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "description": self.description
        }


@dataclass
class StrategyState:
    """策略状态"""
    strategy_id: str
    current_weight: float        # 当前权重
    current_return: float         # 当前收益
    drawdown: float               # 当前回撤
    is_in_drawdown: bool          # 是否处于回撤中
    risk_score: float             # 风险评分

    # 策略类型（用于规则判断）
    strategy_type: str = "neutral"  # "trend_following", "mean_reversion", "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AllocationResult:
    """分配结果"""
    target_weights: Dict[str, float]     # 目标权重
    weight_caps: Dict[str, float]        # 仓位上限
    disabled_strategies: List[str]        # 禁用列表

    # 元数据
    allocation_reasons: Dict[str, str]   # {strategy_id: reason}
    applied_rules: List[str]             # 应用的规则列表

    def __post_init__(self):
        if self.target_weights is None:
            self.target_weights = {}
        if self.weight_caps is None:
            self.weight_caps = {}
        if self.disabled_strategies is None:
            self.disabled_strategies = []
        if self.allocation_reasons is None:
            self.allocation_reasons = {}
        if self.applied_rules is None:
            self.applied_rules = []

    def normalize_weights(self) -> None:
        """归一化权重（使总和为1）"""
        active_weights = {
            k: v for k, v in self.target_weights.items()
            if k not in self.disabled_strategies
        }

        if not active_weights:
            return

        total = sum(active_weights.values())
        if total > 0:
            for strat_id in active_weights:
                self.target_weights[strat_id] /= total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_weights": self.target_weights,
            "weight_caps": self.weight_caps,
            "disabled_strategies": self.disabled_strategies,
            "allocation_reasons": self.allocation_reasons,
            "applied_rules": self.applied_rules
        }


# ========== 预定义规则 ==========

class VolatilityAdjustmentRule(AllocationRule):
    """波动率调整规则

    高波动市场降低总仓位。
    """

    def __init__(
        self,
        high_vol_cap: float = 0.8,
        medium_vol_cap: float = 0.9
    ):
        super().__init__(
            rule_id="volatility_adjustment",
            rule_name="高波动降低仓位",
            rule_type=AllocationRuleType.VOLATILITY_ADJUSTMENT,
            priority=1,
            description="在高波动市场降低总仓位"
        )
        self.high_vol_cap = high_vol_cap
        self.medium_vol_cap = medium_vol_cap

    def apply(
        self,
        regime: RegimeFeatures,
        strategy_states: Dict[str, StrategyState],
        result: AllocationResult
    ) -> None:
        if not self.enabled:
            return

        total_cap = 1.0

        if regime.vol_bucket == "high":
            total_cap = self.high_vol_cap
            result.applied_rules.append(f"{self.rule_id}:high")
        elif regime.vol_bucket == "med":
            total_cap = self.medium_vol_cap
            result.applied_rules.append(f"{self.rule_id}:med")

        # 应用总仓位限制
        current_total = sum(result.target_weights.values())
        if current_total > total_cap:
            scale = total_cap / current_total
            for strat_id in result.target_weights:
                result.target_weights[strat_id] *= scale


class TrendFilterRule(AllocationRule):
    """趋势过滤规则

    在震荡市场禁用趋势跟随策略。
    """

    def __init__(self, trend_following_strategies: List[str] = None):
        super().__init__(
            rule_id="trend_filter",
            rule_name="震荡市场禁用趋势策略",
            rule_type=AllocationRuleType.TREND_FILTER,
            priority=2,
            description="在震荡市场禁用趋势跟随策略"
        )
        self.trend_following_strategies = trend_following_strategies or []

    def apply(
        self,
        regime: RegimeFeatures,
        strategy_states: Dict[str, StrategyState],
        result: AllocationResult
    ) -> None:
        if not self.enabled:
            return

        if regime.trend_bucket == "weak":
            for strat_id in self.trend_following_strategies:
                if strat_id in result.target_weights:
                    result.target_weights[strat_id] = 0.0
                    result.disabled_strategies.append(strat_id)
                    result.allocation_reasons[strat_id] = "震荡市场禁用趋势策略"
            result.applied_rules.append(f"{self.rule_id}:weak_trend")


class CoCrashExclusionRule(AllocationRule):
    """协同爆炸互斥规则

    如果两个策略容易一起亏损，限制它们的总权重。
    """

    def __init__(
        self,
        co_crash_pairs: List[tuple] = None,
        max_combined_weight: float = 0.6
    ):
        super().__init__(
            rule_id="co_crash_exclusion",
            rule_name="协同爆炸对互斥",
            rule_type=AllocationRuleType.CO_CRASH_EXCLUSION,
            priority=3,
            description="限制协同爆炸对的组合权重"
        )
        self.co_crash_pairs = co_crash_pairs or []
        self.max_combined_weight = max_combined_weight

    def apply(
        self,
        regime: RegimeFeatures,
        strategy_states: Dict[str, StrategyState],
        result: AllocationResult
    ) -> None:
        if not self.enabled:
            return

        for s1, s2 in self.co_crash_pairs:
            if s1 not in result.target_weights or s2 not in result.target_weights:
                continue

            combined_weight = result.target_weights[s1] + result.target_weights[s2]

            if combined_weight > self.max_combined_weight:
                # 按比例降低
                scale = self.max_combined_weight / combined_weight
                result.target_weights[s1] *= scale
                result.target_weights[s2] *= scale

                result.weight_caps[s1] = min(
                    result.weight_caps.get(s1, 1.0),
                    self.max_combined_weight * 0.6
                )
                result.weight_caps[s2] = min(
                    result.weight_caps.get(s2, 1.0),
                    self.max_combined_weight * 0.6
                )

                result.applied_rules.append(f"{self.rule_id}:{s1}_{s2}")


class DrawdownProtectionRule(AllocationRule):
    """回撤保护规则

    如果策略深度回撤，降低其权重。
    """

    def __init__(
        self,
        drawdown_threshold: float = 0.10,
        weight_reduction: float = 0.5
    ):
        super().__init__(
            rule_id="drawdown_protection",
            rule_name="回撤保护",
            rule_type=AllocationRuleType.DRAWDOWN_PROTECTION,
            priority=4,
            description="降低深度回撤策略的权重"
        )
        self.drawdown_threshold = drawdown_threshold
        self.weight_reduction = weight_reduction

    def apply(
        self,
        regime: RegimeFeatures,
        strategy_states: Dict[str, StrategyState],
        result: AllocationResult
    ) -> None:
        if not self.enabled:
            return

        for strat_id, state in strategy_states.items():
            if state.is_in_drawdown and state.drawdown > self.drawdown_threshold:
                if strat_id in result.target_weights:
                    original_weight = result.target_weights[strat_id]
                    result.target_weights[strat_id] *= self.weight_reduction
                    result.allocation_reasons[strat_id] = (
                        f"回撤保护: {state.drawdown*100:.1f}%回撤，权重从{original_weight:.2f}降至{result.target_weights[strat_id]:.2f}"
                    )
                    result.applied_rules.append(f"{self.rule_id}:{strat_id}")


class BudgetConstraintRule(AllocationRule):
    """预算约束规则

    确保分配满足预算约束。
    """

    def __init__(self, budget: PortfolioRiskBudget):
        super().__init__(
            rule_id="budget_constraint",
            rule_name="预算约束",
            rule_type=AllocationRuleType.BUDGET_CONSTRAINT,
            priority=99,  # 最后执行
            description="确保分配满足预算约束"
        )
        self.budget = budget

    def apply(
        self,
        regime: RegimeFeatures,
        strategy_states: Dict[str, StrategyState],
        result: AllocationResult
    ) -> None:
        if not self.enabled:
            return

        # 检查最小权重
        for strat_id, weight in result.target_weights.items():
            if weight < self.budget.min_weight_per_strategy and weight > 0:
                result.target_weights[strat_id] = 0
                result.disabled_strategies.append(strat_id)

        # 检查最大权重
        for strat_id, weight in list(result.target_weights.items()):
            max_weight = result.weight_caps.get(strat_id, self.budget.max_weight_per_strategy)
            if weight > max_weight:
                result.target_weights[strat_id] = max_weight

        # 检查策略数量
        active_count = sum(
            1 for w in result.target_weights.values()
            if w > 0
        )
        if active_count < self.budget.min_strategies_count:
            # 启用一些被禁用的策略
            for strat_id in result.disabled_strategies[:self.budget.min_strategies_count - active_count]:
                result.target_weights[strat_id] = self.budget.min_weight_per_strategy
                result.disabled_strategies.remove(strat_id)

        result.applied_rules.append(f"{self.rule_id}:applied")


class RuleBasedAllocator:
    """基于规则的预算分配器"""

    def __init__(
        self,
        budget: PortfolioRiskBudget,
        rules: List[AllocationRule] = None
    ):
        """初始化分配器

        Args:
            budget: 风险预算
            rules: 分配规则列表（None则使用默认规则）
        """
        self.budget = budget
        self.rules = rules or self._create_default_rules()

        # 按优先级排序
        self.rules.sort(key=lambda r: r.priority)

    def _create_default_rules(self) -> List[AllocationRule]:
        """创建默认规则"""
        return [
            VolatilityAdjustmentRule(),
            TrendFilterRule(),
            CoCrashExclusionRule(),
            DrawdownProtectionRule(),
            BudgetConstraintRule(self.budget)
        ]

    def allocate(
        self,
        regime: RegimeFeatures,
        strategy_states: Dict[str, StrategyState],
        current_weights: Optional[Dict[str, float]] = None
    ) -> AllocationResult:
        """计算分配方案

        Args:
            regime: 当前市场状态
            strategy_states: 策略状态
            current_weights: 当前权重（可选）

        Returns:
            AllocationResult
        """
        # 初始化结果
        if current_weights:
            result = AllocationResult(
                target_weights=current_weights.copy(),
                weight_caps={},
                disabled_strategies=[],
                allocation_reasons={},
                applied_rules=[]
            )
        else:
            # 等权重初始化
            n_strategies = len(strategy_states)
            if n_strategies > 0:
                initial_weight = 1.0 / n_strategies
                result = AllocationResult(
                    target_weights={sid: initial_weight for sid in strategy_states},
                    weight_caps={},
                    disabled_strategies=[],
                    allocation_reasons={},
                    applied_rules=[]
                )
            else:
                result = AllocationResult()

        # 应用每条规则
        for rule in self.rules:
            rule.apply(regime, strategy_states, result)

        # 归一化权重
        result.normalize_weights()

        return result

    def add_rule(self, rule: AllocationRule) -> None:
        """添加规则"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def remove_rule(self, rule_id: str) -> None:
        """移除规则"""
        self.rules = [r for r in self.rules if r.rule_id != rule_id]

    def get_rule(self, rule_id: str) -> Optional[AllocationRule]:
        """获取规则"""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def enable_rule(self, rule_id: str) -> bool:
        """启用规则"""
        rule = self.get_rule(rule_id)
        if rule:
            rule.enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """禁用规则"""
        rule = self.get_rule(rule_id)
        if rule:
            rule.enabled = False
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "budget": self.budget.to_dict(),
            "rules": [rule.to_dict() for rule in self.rules],
            "num_rules": len(self.rules)
        }


if __name__ == "__main__":
    """测试代码"""
    print("=== BudgetAllocator 测试 ===\n")

    from analysis.portfolio.risk_budget import create_moderate_budget

    # 创建分配器
    budget = create_moderate_budget()
    allocator = RuleBasedAllocator(budget)

    # 创建模拟数据
    regime = RegimeFeatures(
        realized_vol=0.025,
        vol_bucket="high",
        adx=20.0,
        trend_bucket="weak",
        spread_proxy=0.001,
        cost_bucket="low",
        calculated_at=datetime.now()
    )

    strategy_states = {
        "strategy_1": StrategyState(
            strategy_id="strategy_1",
            current_weight=0.4,
            current_return=0.05,
            drawdown=0.02,
            is_in_drawdown=False,
            risk_score=45.0,
            strategy_type="trend_following"
        ),
        "strategy_2": StrategyState(
            strategy_id="strategy_2",
            current_weight=0.4,
            current_return=-0.08,
            drawdown=0.12,
            is_in_drawdown=True,
            risk_score=65.0,
            strategy_type="neutral"
        ),
        "strategy_3": StrategyState(
            strategy_id="strategy_3",
            current_weight=0.2,
            current_return=0.03,
            drawdown=0.01,
            is_in_drawdown=False,
            risk_score=35.0,
            strategy_type="mean_reversion"
        )
    }

    # 测试1: 基本分配
    print("1. 基本分配（高波动+震荡市场）:")
    result = allocator.allocate(regime, strategy_states)

    print(f"   应用的规则: {', '.join(result.applied_rules)}")
    print(f"   禁用的策略: {result.disabled_strategies}")
    print(f"   目标权重:")
    for strat_id, weight in result.target_weights.items():
        reason = result.allocation_reasons.get(strat_id, "")
        print(f"     {strat_id}: {weight*100:.1f}% - {reason}")

    # 测试2: 规则管理
    print("\n2. 规则管理:")
    print(f"   规则数量: {len(allocator.rules)}")
    for rule in allocator.rules:
        print(f"   - {rule.rule_name} (优先级={rule.priority}, 启用={rule.enabled})")

    # 测试3: 禁用波动率调整
    print("\n3. 禁用波动率调整规则:")
    allocator.disable_rule("volatility_adjustment")
    result2 = allocator.allocate(regime, strategy_states)

    print(f"   应用的规则: {', '.join(result2.applied_rules)}")
    print(f"   目标权重:")
    for strat_id, weight in result2.target_weights.items():
        print(f"     {strat_id}: {weight*100:.1f}%")

    print("\n✓ 测试完成")
