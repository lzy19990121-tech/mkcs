"""
MetaStrategyAgent - 主裁判 / Brain

融合多个 Alpha 观点，决定最终目标仓位

这是系统中唯一的"下决定的人"
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from core.schema import (
    MarketState,
    AlphaOpinion,
    MetaDecision,
    SchemaVersion
)

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """融合配置"""
    # 共识阈值
    consensus_agreement_threshold: float = 0.7  # 70% 策略方向一致算共识

    # 冲突处理模式
    conflict_mode: str = "confidence"  # confidence / market_state / strongest

    # 无共识行为
    no_consensus_action: str = "reduce"  # reduce / flat / scale

    # 权重配置
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "ma": 1.0,
        "breakout": 1.0,
        "ml": 0.8
    })


class MetaStrategyAgent:
    """
    MetaStrategyAgent - 主裁判 / Brain

    职责：
    ✓ 融合多个 Alpha 观点
    ✓ 决定最终目标仓位
    ✓ 说明哪些策略被启用/禁用
    ✓ 处理策略冲突

    不做：
    ✗ 不直接下单
    ✗ 不做风险控制（交给 RiskGate）
    """

    def __init__(
        self,
        max_position_weight: float = 0.2,  # 单标的最大权重
        fusion_config: Optional[FusionConfig] = None
    ):
        """
        初始化 MetaStrategyAgent

        Args:
            max_position_weight: 单标的最大权重
            fusion_config: 融合配置
        """
        self.max_position_weight = max_position_weight
        self.fusion_config = fusion_config or FusionConfig()

        # 决策历史
        self._decision_history: List[MetaDecision] = []

    def decide(
        self,
        market_state: MarketState,
        opinions: List[AlphaOpinion],
        current_position: float = 0,
        available_capital: float = 100000
    ) -> MetaDecision:
        """
        做出交易决策

        Args:
            market_state: 市场状态
            opinions: 所有 Alpha 策略的观点
            current_position: 当前持仓
            available_capital: 可用资金

        Returns:
            MetaDecision
        """
        symbol = market_state.symbol
        timestamp = market_state.timestamp

        # 1. 分类观点
        enabled_opinions = [o for o in opinions if not o.is_disabled]
        disabled_strategies = {
            o.strategy_name: o.disabled_reason
            for o in opinions if o.is_disabled
        }

        # 2. 分析共识
        consensus_info = self._analyze_consensus(enabled_opinions)

        # 3. 应用融合规则
        decision_result = self._apply_fusion_rules(
            enabled_opinions,
            consensus_info,
            market_state
        )

        # 4. 计算目标仓位
        target_position, target_weight = self._calculate_target_position(
            decision_result,
            market_state,
            current_position,
            available_capital
        )

        # 5. 构建决策
        decision = MetaDecision(
            timestamp=timestamp,
            symbol=symbol,
            target_position=target_position,
            target_weight=target_weight,
            active_strategies=[o.strategy_name for o in enabled_opinions],
            disabled_strategies=disabled_strategies,
            decision_confidence=decision_result["confidence"],
            consensus_level=decision_result["consensus_level"],
            reasoning=decision_result["reasoning"]
        )

        self._decision_history.append(decision)
        return decision

    def _analyze_consensus(
        self,
        opinions: List[AlphaOpinion]
    ) -> Dict[str, Any]:
        """分析策略共识"""
        if not opinions:
            return {
                "has_consensus": False,
                "agreement_ratio": 0,
                "long_count": 0,
                "short_count": 0,
                "neutral_count": 0
            }

        long_opinions = [o for o in opinions if o.direction > 0]
        short_opinions = [o for o in opinions if o.direction < 0]
        neutral_opinions = [o for o in opinions if o.direction == 0]

        total = len(opinions)
        long_count = len(long_opinions)
        short_count = len(short_opinions)
        neutral_count = len(neutral_opinions)

        # 计算一致比例（最大同向比例）
        max_same_direction = max(long_count, short_count, neutral_count)
        agreement_ratio = max_same_direction / total if total > 0 else 0

        return {
            "has_consensus": agreement_ratio >= self.fusion_config.consensus_agreement_threshold,
            "agreement_ratio": agreement_ratio,
            "long_count": long_count,
            "short_count": short_count,
            "neutral_count": neutral_count,
            "long_opinions": long_opinions,
            "short_opinions": short_opinions
        }

    def _apply_fusion_rules(
        self,
        opinions: List[AlphaOpinion],
        consensus_info: Dict[str, Any],
        market_state: MarketState
    ) -> Dict[str, Any]:
        """
        应用融合规则

        规则优先级：
        1. 共识型：多策略同向加权
        2. 冲突型：强者优先 / 市场状态裁决
        3. 无共识：空仓或轻仓
        """
        if not opinions:
            return {
                "direction": 0,
                "strength": 0,
                "confidence": 0,
                "consensus_level": "NONE",
                "reasoning": "无启用策略"
            }

        # 共识型
        if consensus_info["has_consensus"]:
            return self._fuse_consensus(consensus_info, opinions)

        # 冲突型
        if consensus_info["long_count"] > 0 and consensus_info["short_count"] > 0:
            return self._fuse_conflict(consensus_info, market_state)

        # 单边但有中性
        return self._fuse_single_sided(consensus_info, opinions)

    def _fuse_consensus(
        self,
        consensus_info: Dict[str, Any],
        opinions: List[AlphaOpinion]
    ) -> Dict[str, Any]:
        """融合共识观点"""
        if consensus_info["long_count"] > 0:
            active_opinions = consensus_info["long_opinions"]
        else:
            active_opinions = consensus_info["short_opinions"]

        # 加权平均
        total_weight = 0
        weighted_signal = 0

        for opinion in active_opinions:
            weight = self.fusion_config.strategy_weights.get(
                opinion.strategy_name.lower(),
                1.0
            )
            signal = opinion.get_position_signal()
            weighted_signal += signal * weight
            total_weight += weight

        final_signal = weighted_signal / total_weight if total_weight > 0 else 0

        # 综合置信度
        avg_confidence = sum(o.confidence for o in active_opinions) / len(active_opinions)

        return {
            "direction": 1 if final_signal > 0 else (-1 if final_signal < 0 else 0),
            "strength": abs(final_signal),
            "confidence": avg_confidence,
            "consensus_level": "STRONG",
            "reasoning": f"{'做多' if final_signal > 0 else '做空'}共识 ({len(active_opinions)}个策略一致)"
        }

    def _fuse_conflict(
        self,
        consensus_info: Dict[str, Any],
        market_state: MarketState
    ) -> Dict[str, Any]:
        """处理冲突观点"""
        mode = self.fusion_config.conflict_mode

        if mode == "confidence":
            return self._resolve_by_confidence(consensus_info)
        elif mode == "market_state":
            return self._resolve_by_market_state(consensus_info, market_state)
        elif mode == "strongest":
            return self._resolve_by_strength(consensus_info)
        else:
            return self._no_consensus_decision(consensus_info)

    def _resolve_by_confidence(
        self,
        consensus_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于置信度解决冲突"""
        all_opinions = (
            consensus_info["long_opinions"] +
            consensus_info["short_opinions"]
        )

        # 选择置信度最高的观点
        best_opinion = max(all_opinions, key=lambda o: o.confidence)

        return {
            "direction": best_opinion.direction,
            "strength": best_opinion.strength,
            "confidence": best_opinion.confidence,
            "consensus_level": "WEAK",
            "reasoning": f"冲突中选择置信度最高的 {best_opinion.strategy_name} ({best_opinion.confidence:.0%} 置信度)"
        }

    def _resolve_by_market_state(
        self,
        consensus_info: Dict[str, Any],
        market_state: MarketState
    ) -> Dict[str, Any]:
        """基于市场状态解决冲突"""
        # 趋势市优先使用趋势策略
        if market_state.regime == "TREND":
            trend_strategies = [o for o in consensus_info["long_opinions"] if o.strategy_name == "MA"]
            if trend_strategies:
                best = max(trend_strategies, key=lambda o: o.confidence)
                return {
                    "direction": best.direction,
                    "strength": best.strength,
                    "confidence": best.confidence,
                    "consensus_level": "WEAK",
                    "reasoning": f"趋势市选择趋势策略 {best.strategy_name}"
                }

        # 震荡市倾向于保守
        if market_state.regime == "RANGE":
            return {
                "direction": 0,
                "strength": 0,
                "confidence": 0.5,
                "consensus_level": "NONE",
                "reasoning": "震荡市策略冲突，选择观望"
            }

        # 默认使用置信度
        return self._resolve_by_confidence(consensus_info)

    def _resolve_by_strength(
        self,
        consensus_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于信号强度解决冲突"""
        all_opinions = (
            consensus_info["long_opinions"] +
            consensus_info["short_opinions"]
        )

        best_opinion = max(all_opinions, key=lambda o: o.strength * o.confidence)

        return {
            "direction": best_opinion.direction,
            "strength": best_opinion.strength,
            "confidence": best_opinion.confidence,
            "consensus_level": "WEAK",
            "reasoning": f"冲突中选择信号最强的 {best_opinion.strategy_name}"
        }

    def _fuse_single_sided(
        self,
        consensus_info: Dict[str, Any],
        opinions: List[AlphaOpinion]
    ) -> Dict[str, Any]:
        """融合单边观点"""
        if consensus_info["long_count"] > 0:
            active = consensus_info["long_opinions"]
        else:
            active = consensus_info["short_opinions"]

        if not active:
            return {
                "direction": 0,
                "strength": 0,
                "confidence": 0,
                "consensus_level": "NONE",
                "reasoning": "所有策略均为中性"
            }

        # 取加权平均
        total_weight = 0
        weighted_signal = 0

        for opinion in active:
            weight = self.fusion_config.strategy_weights.get(opinion.strategy_name, 1.0)
            signal = opinion.get_position_signal()
            weighted_signal += signal * weight
            total_weight += weight

        final_signal = weighted_signal / total_weight if total_weight > 0 else 0
        avg_confidence = sum(o.confidence for o in active) / len(active)

        return {
            "direction": 1 if final_signal > 0 else -1,
            "strength": abs(final_signal),
            "confidence": avg_confidence,
            "consensus_level": "WEAK",
            "reasoning": f"单边信号 ({len(active)}个策略)"
        }

    def _no_consensus_decision(
        self,
        consensus_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """无共识决策"""
        action = self.fusion_config.no_consensus_action

        if action == "flat":
            return {
                "direction": 0,
                "strength": 0,
                "confidence": 0,
                "consensus_level": "NONE",
                "reasoning": "策略冲突且无共识，选择空仓"
            }
        elif action == "scale":
            # 降低仓位但保持方向
            return {
                "direction": 0,
                "strength": 0.3,
                "confidence": 0.3,
                "consensus_level": "NONE",
                "reasoning": "策略冲突，降低仓位"
            }
        else:  # reduce
            return {
                "direction": 0,
                "strength": 0,
                "confidence": 0,
                "consensus_level": "NONE",
                "reasoning": "策略冲突，保守处理"
            }

    def _calculate_target_position(
        self,
        decision_result: Dict[str, Any],
        market_state: MarketState,
        current_position: float,
        available_capital: float
    ) -> tuple[float, float]:
        """
        计算目标仓位

        Returns:
            (target_position, target_weight)
        """
        direction = decision_result["direction"]
        strength = decision_result["strength"]
        confidence = decision_result["confidence"]

        # 基础权重 = 方向 * 强度 * 置信度 * 最大权重
        base_weight = abs(direction) * strength * confidence * self.max_position_weight

        # 应用市场状态调整
        market_scale = market_state.position_scale_factor if hasattr(market_state, 'position_scale_factor') else 1.0
        adjusted_weight = base_weight * market_scale

        # 限制在合理范围内
        adjusted_weight = max(0, min(adjusted_weight, self.max_position_weight))

        # 计算股数（简化，实际需要价格数据）
        # 这里假设平均价格为 100
        avg_price = 100
        target_position = (available_capital * adjusted_weight) / avg_price

        # 考虑方向
        target_position = target_position if direction > 0 else -target_position

        return target_position, adjusted_weight

    def get_decision_history(self, limit: int = 100) -> List[MetaDecision]:
        """获取决策历史"""
        return self._decision_history[-limit:]

    def get_last_decision(self) -> Optional[MetaDecision]:
        """获取最新决策"""
        return self._decision_history[-1] if self._decision_history else None


if __name__ == "__main__":
    """测试代码"""
    print("=== MetaStrategyAgent 测试 ===\n")

    from datetime import timedelta

    # 创建测试数据
    bars = []
    for i in range(50):
        date = datetime(2024, 1, 1) + timedelta(days=i)
        price = 100 + i * 0.5 + (i % 5 - 2) * 0.3

        bars.append({
            "timestamp": date,
            "symbol": "AAPL",
            "close": price
        })

    # 创建 MarketState
    from core.schema import MarketState

    market_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime="TREND",
        regime_confidence=0.8,
        volatility_state="NORMAL",
        volatility_trend="STABLE",
        volatility_percentile=0.6,
        liquidity_state="NORMAL",
        volume_ratio=1.2,
        sentiment_state="NEUTRAL",
        sentiment_score=0.0
    )

    # 创建 Alpha 观点
    opinions = [
        AlphaOpinion(
            strategy_name="MA",
            timestamp=datetime.now(),
            symbol="AAPL",
            direction=1,
            strength=0.8,
            confidence=0.7,
            horizon="daily",
            reason="MA 金叉"
        ),
        AlphaOpinion(
            strategy_name="Breakout",
            timestamp=datetime.now(),
            symbol="AAPL",
            direction=1,
            strength=0.6,
            confidence=0.6,
            horizon="swing",
            reason="向上突破"
        ),
        AlphaOpinion(
            strategy_name="ML",
            timestamp=datetime.now(),
            symbol="AAPL",
            direction=1,
            strength=0.5,
            confidence=0.5,
            horizon="intraday",
            reason="ML 做多信号"
        )
    ]

    # 测试共识型融合
    print("1. 共识型融合:")
    meta = MetaStrategyAgent()
    decision = meta.decide(market_state, opinions)
    print(f"   Target position: {decision.target_position:.0f}")
    print(f"   Target weight: {decision.target_weight:.2%}")
    print(f"   Active strategies: {decision.active_strategies}")
    print(f"   Consensus level: {decision.consensus_level}")
    print(f"   Reasoning: {decision.reasoning}")

    # 测试冲突型融合
    print("\n2. 冲突型融合:")
    conflict_opinions = [
        AlphaOpinion(
            strategy_name="MA",
            timestamp=datetime.now(),
            symbol="AAPL",
            direction=1,
            strength=0.8,
            confidence=0.7,
            horizon="daily",
            reason="MA 做多"
        ),
        AlphaOpinion(
            strategy_name="Breakout",
            timestamp=datetime.now(),
            symbol="AAPL",
            direction=-1,
            strength=0.6,
            confidence=0.6,
            horizon="swing",
            reason="Breakout 做空"
        )
    ]

    decision = meta.decide(market_state, conflict_opinions)
    print(f"   Target position: {decision.target_position:.0f}")
    print(f"   Consensus level: {decision.consensus_level}")
    print(f"   Reasoning: {decision.reasoning}")

    # 测试有禁用策略的情况
    print("\n3. 包含禁用策略:")
    mixed_opinions = [
        AlphaOpinion(
            strategy_name="MA",
            timestamp=datetime.now(),
            symbol="AAPL",
            direction=1,
            strength=0.8,
            confidence=0.7,
            horizon="daily",
            reason="MA 金叉"
        ),
        AlphaOpinion(
            strategy_name="Breakout",
            timestamp=datetime.now(),
            symbol="AAPL",
            direction=1,
            strength=0.6,
            confidence=0.6,
            horizon="swing",
            is_disabled=True,
            disabled_reason="regime_RANGE"
        )
    ]

    decision = meta.decide(market_state, mixed_opinions)
    print(f"   Active strategies: {decision.active_strategies}")
    print(f"   Disabled strategies: {decision.disabled_strategies}")
    print(f"   Reasoning: {decision.reasoning}")

    print("\n✓ 所有测试通过")
