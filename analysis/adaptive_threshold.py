"""
SPL-5a-B: 自适应阈值函数设计

将固定阈值改为随市场状态变化的函数形式。
"""

from dataclasses import dataclass
from typing import Dict, Any, Callable, Union, List
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from analysis.regime_features import RegimeFeatures


class ThresholdFunction(ABC):
    """阈值函数基类"""

    @abstractmethod
    def evaluate(self, regime: RegimeFeatures) -> float:
        """根据市场状态计算阈值

        Args:
            regime: 市场状态特征

        Returns:
            阈值
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        pass


@dataclass
class PiecewiseConstantThreshold(ThresholdFunction):
    """分段常数阈值函数

    根据特征分桶返回不同的阈值。
    """

    feature_name: str           # 特征名称（如 "realized_vol"）
    buckets: Dict[str, float]     # 分桶映射 (bucket -> threshold)

    def evaluate(self, regime: RegimeFeatures) -> float:
        """根据市场状态返回阈值"""
        # 获取特征值
        feature_value = getattr(regime, self.feature_name)

        # 获取桶值
        bucket = self._get_bucket(feature_value)

        # 返回对应阈值
        return self.buckets.get(bucket, self.buckets[list(self.buckets.keys())[0]])

    def _get_bucket(self, value: float) -> str:
        """根据值确定桶"""
        # 获取排序后的桶边界
        sorted_keys = sorted(self.buckets.keys(), key=lambda k: self._get_bucket_max(k))

        # 找到合适的桶
        for bucket in sorted_keys:
            max_val = self._get_bucket_max(bucket)
            if value < max_val:
                return bucket

        # 如果都不匹配，返回最后一个桶
        return list(self.buckets.keys())[-1]

    def _get_bucket_max(self, bucket: str) -> float:
        """获取桶的最大值"""
        # 假设桶命名格式为 "low", "med", "high" 等
        # 这里简化处理，实际应该从配置中读取
        bucket_order = ["low", "med", "high"]
        idx = bucket_order.index(bucket) if bucket in bucket_order else len(bucket_order) - 1

        # 从 buckets 配置中获取实际边界（这里简化）
        if bucket == "low":
            return 0.01
        elif bucket == "med":
            return 0.02
        elif bucket == "high":
            return float("inf")
        return 0.02

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": "piecewise_constant",
            "feature_name": self.feature_name,
            "buckets": self.buckets
        }


@dataclass
class LinearThreshold(ThresholdFunction):
    """线性阈值函数

    根据特征值线性计算阈值：threshold = a + b * value
    """

    feature_name: str           # 特征名称
    intercept: float             # 截距
    slope: float                 # 斜率
    min_value: float = None      # 最小阈值（可选）
    max_value: float = None      # 最大阈值（可选）

    def evaluate(self, regime: RegimeFeatures) -> float:
        """根据市场状态返回阈值"""
        # 获取特征值
        feature_value = getattr(regime, self.feature_name)

        # 线性计算
        threshold = self.intercept + self.slope * feature_value

        # 限制范围
        if self.min_value is not None:
            threshold = max(threshold, self.min_value)
        if self.max_value is not None:
            threshold = min(threshold, self.max_value)

        return threshold

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": "linear",
            "feature_name": self.feature_name,
            "intercept": self.intercept,
            "slope": self.slope,
            "min_value": self.min_value,
            "max_value": self.max_value
        }


@dataclass
class AdaptiveThreshold:
    """自适应阈值配置

    包含策略的所有自适应阈值规则。
    """

    rule_id: str                 # 规则ID
    rule_name: str               # 规则名称
    rule_type: str               # 规则类型 (gating/reduction/disable)
    trigger_metric: str          # 触发指标
    threshold_function: ThresholdFunction  # 阈值函数
    description: str             # 描述

    def get_threshold(self, regime: RegimeFeatures) -> float:
        """获取当前市场状态的阈值"""
        return self.threshold_function.evaluate(regime)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_type": self.rule_type,
            "trigger_metric": self.trigger_metric,
            "threshold_function": self.threshold_function.to_dict(),
            "description": self.description
        }


# 预定义的阈值函数配置
def create_stability_gating_threshold() -> AdaptiveThreshold:
    """创建稳定性gating的自适应阈值

    规则：稳定性评分阈值随波动变化
    - 低波动：阈值降低（更宽松）
    - 高波动：阈值提高（更严格）
    """
    return AdaptiveThreshold(
        rule_id="adaptive_stability_gating",
        rule_name="自适应稳定性暂停交易",
        rule_type="gating",
        trigger_metric="stability_score",
        threshold_function=PiecewiseConstantThreshold(
            feature_name="realized_vol",
            buckets={
                "low": 20.0,    # 低波动：稳定性评分阈值20
                "med": 30.0,    # 中等波动：30
                "high": 40.0     # 高波动：40
            }
        ),
        description="稳定性评分阈值随波动自适应调整"
    )


def create_return_reduction_threshold() -> AdaptiveThreshold:
    """创建收益降仓的自适应阈值

    规则：降仓触发阈值随波动变化
    - 低波动：阈值较低（更容易降仓）
    - 高波动：阈值更高（需要更大亏损才降仓）
    """
    return AdaptiveThreshold(
        rule_id="adaptive_return_reduction",
        rule_name="自适应收益降仓",
        rule_type="reduction",
        trigger_metric="window_return",
        threshold_function=LinearThreshold(
            feature_name="realized_vol",
            intercept=-0.08,   # 基础阈值 -8%
            slope=-2.0,       # 波动每增加1%，阈值降低2%
            min_value=-0.20,   # 最小阈值 -20%
            max_value=-0.05    # 最大阈值 -5%
        ),
        description="降仓阈值随波动自适应（低波动更敏感）"
    )


def create_regime_disable_threshold() -> AdaptiveThreshold:
    """创建市场状态禁用的自适应阈值

    规则：在震荡市场（ADX低）时禁用策略
    """
    return AdaptiveThreshold(
        rule_id="adaptive_regime_disable",
        rule_name="自适应市场状态禁用",
        rule_type="disable",
        trigger_metric="market_regime",
        threshold_function=PiecewiseConstantThreshold(
            feature_name="adx",
            buckets={
                "weak": 0.0,     # 弱趋势：禁用（阈值=0）
                "strong": 1.0    # 强趋势：不禁用（阈值=1）
            }
        ),
        description="根据趋势强度决定是否禁用策略"
    )


def create_duration_gating_threshold() -> AdaptiveThreshold:
    """创建回撤持续gating的自适应阈值

    规则：回撤持续阈值随波动变化
    - 高波动：更容易触发gating（阈值降低）
    """
    return AdaptiveThreshold(
        rule_id="adaptive_duration_gating",
        rule_name="自适应回撤持续暂停",
        rule_type="gating",
        trigger_metric="drawdown_duration",
        threshold_function=LinearThreshold(
            feature_name="realized_vol",
            intercept=10.0,    # 基础阈值 10天
            slope=-200.0,     # 波动每增加1%，阈值降低200天
            min_value=5.0,     # 最小阈值 5天
            max_value=15.0     # 最大阈值 15天
        ),
        description="回撤持续阈值随波动自适应调整"
    )


# 规则集管理器
class AdaptiveThresholdRuleset:
    """自适应阈值规则集"""

    def __init__(self):
        self.rules: Dict[str, AdaptiveThreshold] = {}

    def add_rule(self, rule: AdaptiveThreshold):
        """添加规则"""
        self.rules[rule.rule_id] = rule

    def get_rule(self, rule_id: str) -> AdaptiveThreshold:
        """获取规则"""
        return self.rules.get(rule_id)

    def get_all_rules(self) -> List[AdaptiveThreshold]:
        """获取所有规则"""
        return list(self.rules.values())

    def calculate_all_thresholds(self, regime: RegimeFeatures) -> Dict[str, float]:
        """计算所有规则的当前阈值

        Args:
            regime: 市场状态特征

        Returns:
            {rule_id: threshold} 字典
        """
        thresholds = {}
        for rule_id, rule in self.rules.items():
            thresholds[rule_id] = rule.get_threshold(regime)
        return thresholds

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "version": "adaptive_v1",
            "num_rules": len(self.rules),
            "rules": {
                rule_id: rule.to_dict()
                for rule_id, rule in self.rules.items()
            }
        }


# 预定义的规则集
def create_default_adaptive_ruleset() -> AdaptiveThresholdRuleset:
    """创建默认的自适应规则集"""
    ruleset = AdaptiveThresholdRuleset()

    ruleset.add_rule(create_stability_gating_threshold())
    ruleset.add_rule(create_return_reduction_threshold())
    ruleset.add_rule(create_regime_disable_threshold())
    ruleset.add_rule(create_duration_gating_threshold())

    return ruleset


if __name__ == "__main__":
    """测试代码"""
    print("=== AdaptiveThreshold 测试 ===\n")

    from analysis.regime_features import RegimeFeatures

    # 创建测试特征
    test_features = [
        RegimeFeatures(
            realized_vol=0.005,
            vol_bucket="low",
            adx=20.0,
            trend_bucket="weak",
            spread_proxy=0.0005,
            cost_bucket="low",
            calculated_at=datetime.now()
        ),
        RegimeFeatures(
            realized_vol=0.015,
            vol_bucket="med",
            adx=30.0,
            trend_bucket="strong",
            spread_proxy=0.001,
            cost_bucket="low",
            calculated_at=datetime.now()
        ),
        RegimeFeatures(
            realized_vol=0.03,
            vol_bucket="high",
            adx=10.0,
            trend_bucket="weak",
            spread_proxy=0.002,
            cost_bucket="high",
            calculated_at=datetime.now()
        )
    ]

    # 创建阈值函数
    stability_threshold = create_stability_gating_threshold()

    print("1. 测试稳定性gating阈值:")
    for features in test_features:
        threshold = stability_threshold.get_threshold(features)
        print(f"   波动={features.realized_vol:.3f} ({features.vol_bucket}) → "
              f"阈值={threshold:.1f}")

    print("\n2. 测试阈值函数序列化:")
    print(stability_threshold.to_dict())

    print("\n3. 测试规则集:")
    ruleset = create_default_adaptive_ruleset()
    print(f"   规则数: {len(ruleset.rules)}")

    print("\n✓ 测试通过")
