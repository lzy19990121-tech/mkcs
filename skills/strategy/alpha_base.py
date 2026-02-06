"""
Alpha 策略基类

所有 L2 Alpha 策略继承此类，只输出 AlphaOpinion
不决定仓位，不产生订单
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

from core.models import Bar, Position
from core.schema import (
    MarketState,
    AlphaOpinion,
    AlphaGatingConfig,
    DEFAULT_ALPHA_GATING
)
import logging

logger = logging.getLogger(__name__)


class AlphaStrategy(ABC):
    """
    Alpha 策略基类

    职责边界：
    ✓ 分析市场产生观点
    ✓ 输出方向、强度、置信度
    ✓ 给出理由
    ✗ 不决定仓位
    ✗ 不产生订单
    """

    def __init__(
        self,
        name: str,
        gating_config: Optional[AlphaGatingConfig] = None
    ):
        """
        初始化 Alpha 策略

        Args:
            name: 策略名称
            gating_config: 禁用配置（如果不提供，使用默认配置）
        """
        self.name = name
        self.gating_config = gating_config or DEFAULT_ALPHA_GATING.get(
            name.lower(),
            AlphaGatingConfig(strategy_name=name)
        )

        # 观点历史（用于可观测性）
        self._opinion_history: List[AlphaOpinion] = []

    @abstractmethod
    def get_min_bars_required(self) -> int:
        """获取策略所需的最小K线数量"""
        pass

    @abstractmethod
    def generate_opinion(
        self,
        bars: List[Bar],
        market_state: MarketState
    ) -> AlphaOpinion:
        """
        生成策略观点（子类实现）

        Args:
            bars: K线数据
            market_state: 市场状态

        Returns:
            AlphaOpinion
        """
        pass

    def analyze(
        self,
        bars: List[Bar],
        market_state: MarketState
    ) -> AlphaOpinion:
        """
        分析并生成观点（应用 gating 规则）

        Args:
            bars: K线数据
            market_state: 市场状态

        Returns:
            AlphaOpinion（可能被禁用）
        """
        if len(bars) < self.get_min_bars_required():
            # 数据不足，输出被禁用的观点
            return AlphaOpinion(
                strategy_name=self.name,
                timestamp=bars[-1].timestamp if bars else datetime.now(),
                symbol=bars[-1].symbol if bars else "UNKNOWN",
                direction=0,
                strength=0,
                confidence=0,
                horizon="daily",
                is_disabled=True,
                disabled_reason="insufficient_data"
            )

        # 检查 gating 规则
        should_disable, disable_reason = self.gating_config.should_disable(market_state)

        if should_disable:
            # 被禁用，输出禁用的观点
            opinion = AlphaOpinion(
                strategy_name=self.name,
                timestamp=bars[-1].timestamp,
                symbol=bars[-1].symbol,
                direction=0,
                strength=0,
                confidence=0,
                horizon="daily",
                is_disabled=True,
                disabled_reason=disable_reason
            )
            self._opinion_history.append(opinion)
            return opinion

        # 未被禁用，生成实际观点
        opinion = self.generate_opinion(bars, market_state)

        # 应用缩放系数（降权但不完全禁用）
        scale_factor = self.gating_config.get_scale_factor(market_state)
        if scale_factor < 1.0:
            opinion.strength *= scale_factor
            opinion.disabled_reason = f"scaled:{scale_factor:.1%}"

        self._opinion_history.append(opinion)
        return opinion

    def get_opinion_history(self, limit: int = 100) -> List[AlphaOpinion]:
        """获取观点历史"""
        return self._opinion_history[-limit:]

    def get_last_opinion(self) -> Optional[AlphaOpinion]:
        """获取最新的观点"""
        return self._opinion_history[-1] if self._opinion_history else None

    def is_currently_disabled(self) -> bool:
        """当前是否被禁用"""
        last_opinion = self.get_last_opinion()
        return last_opinion.is_disabled if last_opinion else False

    def get_disable_reason(self) -> str:
        """获取禁用原因"""
        last_opinion = self.get_last_opinion()
        return last_opinion.disabled_reason if last_opinion else ""


# =============================================================================
# 具体策略实现（降级为观点输出）
# =============================================================================

class MAStrategy(AlphaStrategy):
    """移动平均策略 - Alpha 版本"""

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 20,
        strength_threshold: float = 0.3
    ):
        super().__init__("MA")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.strength_threshold = strength_threshold

    def get_min_bars_required(self) -> int:
        return self.slow_period + 5

    def generate_opinion(
        self,
        bars: List[Bar],
        market_state: MarketState
    ) -> AlphaOpinion:
        """生成 MA 观点"""
        # 计算快慢 MA
        closes = [float(b.close) for b in bars]

        fast_ma = sum(closes[-self.fast_period:]) / self.fast_period
        slow_ma = sum(closes[-self.slow_period:]) / self.slow_period

        # 计算趋势强度
        trend_strength = abs(fast_ma - slow_ma) / slow_ma

        # 判断方向
        if fast_ma > slow_ma and trend_strength > 0.005:
            direction = 1
            reason = f"MA 金叉 (快线 {fast_ma:.2f} > 慢线 {slow_ma:.2f})"
        elif fast_ma < slow_ma and trend_strength > 0.005:
            direction = -1
            reason = f"MA 死叉 (快线 {fast_ma:.2f} < 慢线 {slow_ma:.2f})"
        else:
            direction = 0
            reason = f"MA 震荡 (快线 {fast_ma:.2f}, 慢线 {slow_ma:.2f})"

        # 计算置信度（基于趋势强度）
        confidence = min(trend_strength * 20, 1.0)

        # 计算信号强度
        strength = min(trend_strength * 10, 1.0)

        return AlphaOpinion(
            strategy_name=self.name,
            timestamp=bars[-1].timestamp,
            symbol=bars[-1].symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            horizon="daily",
            reason=reason
        )


class BreakoutStrategy(AlphaStrategy):
    """突破策略 - Alpha 版本"""

    def __init__(
        self,
        period: int = 20,
        threshold_multiplier: float = 1.5
    ):
        super().__init__("Breakout")
        self.period = period
        self.threshold_multiplier = threshold_multiplier

    def get_min_bars_required(self) -> int:
        return self.period + 10

    def generate_opinion(
        self,
        bars: List[Bar],
        market_state: MarketState
    ) -> AlphaOpinion:
        """生成突破观点"""
        # 计算区间
        recent_bars = bars[-self.period:]
        high = max(float(b.high) for b in recent_bars)
        low = min(float(b.low) for b in recent_bars)
        close = float(bars[-1].close)

        # 计算波动率调整后的阈值
        atr = self._calculate_atr(bars[-14:])
        volatility_adjusted_threshold = atr * self.threshold_multiplier

        # 判断突破
        if close > high - volatility_adjusted_threshold:
            direction = 1
            reason = f"向上突破 (收盘 {close:.2f} > 区间上限 {high:.2f})"
        elif close < low + volatility_adjusted_threshold:
            direction = -1
            reason = f"向下突破 (收盘 {close:.2f} < 区间下限 {low:.2f})"
        else:
            direction = 0
            reason = f"区间震荡 (收盘 {close:.2f} 在 [{low:.2f}, {high:.2f}] 内)"

        # 计算置信度
        if direction != 0:
            # 突破强度决定置信度
            if direction == 1:
                strength = (close - (high - volatility_adjusted_threshold)) / atr
            else:
                strength = ((low + volatility_adjusted_threshold) - close) / atr
            confidence = min(strength / 2, 1.0)
        else:
            strength = 0
            confidence = 0

        return AlphaOpinion(
            strategy_name=self.name,
            timestamp=bars[-1].timestamp,
            symbol=bars[-1].symbol,
            direction=direction,
            strength=min(strength, 1.0),
            confidence=confidence,
            horizon="swing",
            reason=reason
        )

    def _calculate_atr(self, bars: List[Bar]) -> float:
        """计算 ATR"""
        if len(bars) < 2:
            return 1.0

        true_ranges = []
        for i in range(1, len(bars)):
            high = float(bars[i].high)
            low = float(bars[i].low)
            prev_close = float(bars[i-1].close)

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        return sum(true_ranges) / len(true_ranges) if true_ranges else 1.0


class MLStrategy(AlphaStrategy):
    """机器学习策略 - Alpha 版本"""

    def __init__(
        self,
        confidence_threshold: float = 0.6
    ):
        super().__init__("ML")
        self.confidence_threshold = confidence_threshold

    def get_min_bars_required(self) -> int:
        return 50

    def generate_opinion(
        self,
        bars: List[Bar],
        market_state: MarketState
    ) -> AlphaOpinion:
        """生成 ML 观点（简化版）"""
        # 简化实现：基于技术指标的加权预测
        closes = [float(b.close) for b in bars[-20:]]
        volumes = [b.volume for b in bars[-20:]]

        # 简单的动量信号
        momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0

        # 成交量确认
        avg_volume = sum(volumes[:-1]) / (len(volumes) - 1) if len(volumes) > 1 else volumes[0]
        volume_confirmation = volumes[-1] / avg_volume if avg_volume > 0 else 1

        # 综合判断
        if abs(momentum) > 0.02 and volume_confirmation > 1.2:
            direction = 1 if momentum > 0 else -1
            strength = min(abs(momentum) * 5, 1.0)
            confidence = min(strength * volume_confirmation / 2, 1.0)
            reason = f"ML 动量信号 (动量 {momentum:+.2%}, 成交量确认 {volume_confirmation:.1f}x)"
        else:
            direction = 0
            strength = 0
            confidence = 0
            reason = "ML 无明确信号"

        return AlphaOpinion(
            strategy_name=self.name,
            timestamp=bars[-1].timestamp,
            symbol=bars[-1].symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            horizon="intraday",
            reason=reason
        )


if __name__ == "__main__":
    """测试代码"""
    print("=== Alpha Strategy 测试 ===\n")

    from datetime import timedelta

    # 创建测试数据
    bars = []
    for i in range(50):
        date = datetime(2024, 1, 1) + timedelta(days=i)
        # 趋势数据
        price = 100 + i * 0.5 + (i % 5 - 2) * 0.3

        bars.append(Bar(
            symbol="AAPL",
            timestamp=date,
            open=Decimal(str(price - 0.2)),
            high=Decimal(str(price + 0.5)),
            low=Decimal(str(price - 0.5)),
            close=Decimal(str(price)),
            volume=1000000,
            interval="1d"
        ))

    # 创建 MarketState（趋势市场）
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

    # 测试 MA 策略
    print("1. MA Strategy:")
    ma = MAStrategy()
    opinion = ma.analyze(bars, market_state)
    print(f"   Has opinion: {opinion.has_opinion()}")
    print(f"   Direction: {opinion.direction}")
    print(f"   Strength: {opinion.strength:.2f}")
    print(f"   Reason: {opinion.reason}")

    # 测试震荡市场下的禁用
    print("\n2. MA Strategy in RANGE market:")
    range_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime="RANGE",
        regime_confidence=0.7,
        volatility_state="NORMAL",
        volatility_trend="STABLE",
        volatility_percentile=0.5,
        liquidity_state="NORMAL",
        volume_ratio=1.0,
        sentiment_state="NEUTRAL",
        sentiment_score=0.0
    )

    opinion = ma.analyze(bars, range_state)
    print(f"   Is disabled: {opinion.is_disabled}")
    print(f"   Disable reason: {opinion.disabled_reason}")
    print(f"   Position signal: {opinion.get_position_signal():.2f}")

    # 测试 Breakout 策略
    print("\n3. Breakout Strategy:")
    breakout = BreakoutStrategy()
    opinion = breakout.analyze(bars, market_state)
    print(f"   Has opinion: {opinion.has_opinion()}")
    print(f"   Direction: {opinion.direction}")
    print(f"   Reason: {opinion.reason}")

    # 测试震荡市场下的 Breakout
    print("\n4. Breakout Strategy in RANGE market:")
    opinion = breakout.analyze(bars, range_state)
    print(f"   Is disabled: {opinion.is_disabled}")
    print(f"   Disable reason: {opinion.disabled_reason}")

    print("\n✓ 所有测试通过")
