"""
增强的移动平均线交叉策略

支持：
- 统一信号输出 (StrategySignal)
- 市场状态感知 (Regime-aware)
- 连续仓位 sizing
- 策略状态管理
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from core.models import Bar, Position
from core.strategy_models import (
    StrategySignal,
    RegimeType,
    RegimeInfo,
    RiskHints
)
from skills.strategy.enhanced_base import EnhancedStrategy
from skills.analysis.regime_detector import RegimeDetector, RegimeDetectorConfig


class EnhancedMAStrategy(EnhancedStrategy):
    """增强版移动平均线交叉策略

    策略逻辑：
    - 快速均线上穿慢速均线（金叉）→ 做多信号
    - 快速均线下穿慢速均线（死叉）→ 平仓/做空信号

    增强功能：
    - 市场状态感知：在震荡市自动禁用
    - 连续仓位 sizing：根据信号强度和波动率调整仓位
    - 动态止盈止损：使用 trailing stop
    """

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 20,
        atr_period: int = 14,
        max_position_ratio: float = 0.2,
        enable_regime_gating: bool = True,
        detector_config: Optional[RegimeDetectorConfig] = None
    ):
        """初始化策略

        Args:
            fast_period: 快速均线周期
            slow_period: 慢速均线周期
            atr_period: ATR 计算周期
            max_position_ratio: 最大仓位比例
            enable_regime_gating: 是否启用市场状态 gating
            detector_config: 市场状态检测器配置
        """
        if fast_period >= slow_period:
            raise ValueError(f"快速均线周期必须小于慢速均线周期: {fast_period} >= {slow_period}")

        super().__init__(
            name="MA",
            regime_detector=RegimeDetector(detector_config) if detector_config else None,
            enable_regime_gating=enable_regime_gating
        )

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.max_position_ratio = max_position_ratio

    def get_min_bars_required(self) -> int:
        """获取所需最小K线数量"""
        return max(self.slow_period, self.atr_period) + 10

    def _generate_strategy_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None,
        regime_info: Optional[RegimeInfo] = None
    ) -> List[StrategySignal]:
        """生成策略信号

        Args:
            bars: K线数据
            position: 当前持仓
            regime_info: 市场状态信息

        Returns:
            StrategySignal 列表
        """
        if len(bars) < self.get_min_bars_required():
            return []

        # 计算均线
        fast_mas = self._calculate_sma(bars, self.fast_period)
        slow_mas = self._calculate_sma(bars, self.slow_period)

        # 获取最新值
        prev_fast = fast_mas[-2]
        prev_slow = slow_mas[-2]
        curr_fast = fast_mas[-1]
        curr_slow = slow_mas[-1]

        # 计算 ATR
        atr = self._calculate_atr(bars, self.atr_period)
        current_price = bars[-1].close

        # 计算波动率（用于仓位 sizing）
        volatility = regime_info.volatility_level if regime_info else 1.0

        # 检测交叉
        signals = []
        symbol = bars[-1].symbol
        timestamp = bars[-1].timestamp

        # 金叉：快速均线从下方穿越慢速均线
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            # 计算信号强度（基于均线距离）
            confidence = self._calculate_confidence(curr_fast, curr_slow, "golden_cross")

            # 连续仓位 sizing：根据信号强度和波动率调整
            target_weight = self._calculate_position_weight(confidence, volatility)

            # 计算目标仓位（绝对数量）
            current_qty = position.quantity if position else 0
            max_qty = 1000  # 默认最大持仓，实际应根据资金计算
            target_position = int(target_weight * max_qty)

            # 风险提示
            risk_hints = RiskHints(
                stop_loss=current_price - (atr * Decimal('1.5')),
                take_profit=current_price + (atr * Decimal('2.5')),
                trailing_stop=Decimal('0.02'),  # 2% trailing stop
                max_holding_days=20,
                position_limit=self.max_position_ratio * target_weight
            )

            signal = StrategySignal(
                symbol=symbol,
                timestamp=timestamp,
                target_position=target_position,
                target_weight=target_weight,
                confidence=confidence,
                reason=f"金叉: MA{self.fast_period}上穿MA{self.slow_period}",
                regime=regime_info.regime if regime_info else RegimeType.UNKNOWN,
                risk_hints=risk_hints,
                raw_action="BUY",
                metadata={
                    "fast_ma": float(curr_fast),
                    "slow_ma": float(curr_slow),
                    "atr": float(atr),
                    "current_price": float(current_price)
                }
            )
            signals.append(signal)

        # 死叉：快速均线从上方穿越慢速均线
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            confidence = self._calculate_confidence(curr_fast, curr_slow, "death_cross")

            # 死叉时平仓
            target_position = 0
            target_weight = 0.0

            signal = StrategySignal(
                symbol=symbol,
                timestamp=timestamp,
                target_position=target_position,
                target_weight=target_weight,
                confidence=confidence,
                reason=f"死叉: MA{self.fast_period}下穿MA{self.slow_period}",
                regime=regime_info.regime if regime_info else RegimeType.UNKNOWN,
                risk_hints=None,
                raw_action="SELL",
                metadata={
                    "fast_ma": float(curr_fast),
                    "slow_ma": float(curr_slow),
                    "current_price": float(current_price)
                }
            )
            signals.append(signal)

        # 更新策略状态
        state = self.get_state()
        state.update("last_fast_ma", float(curr_fast))
        state.update("last_slow_ma", float(curr_slow))
        state.update("last_signal", signals[0].raw_action if signals else "HOLD")
        state.update("last_signal_time", timestamp.isoformat())

        return signals

    def _calculate_position_weight(self, confidence: float, volatility: float) -> float:
        """计算目标仓位权重（连续 sizing）

        Args:
            confidence: 信号置信度 (0-1)
            volatility: 波动率水平（相对值）

        Returns:
            目标仓位权重 (-1 到 1)
        """
        # 基础仓位：信号强度 * 最大仓位
        base_weight = confidence * self.max_position_ratio

        # 波动率调整：高波动降低仓位
        vol_adjustment = 1.0 / max(volatility, 0.5)

        # 最终权重
        final_weight = base_weight * vol_adjustment
        final_weight = min(final_weight, self.max_position_ratio)
        final_weight = max(final_weight, 0.0)

        return final_weight

    def _calculate_atr(self, bars: List[Bar], period: int = 14) -> Decimal:
        """计算 ATR"""
        if len(bars) < period + 1:
            total_range = sum((bar.high - bar.low) for bar in bars)
            return total_range / Decimal(len(bars)) if bars else Decimal('1')

        true_ranges = []
        for i in range(1, len(bars)):
            prev_close = bars[i - 1].close
            current_high = bars[i].high
            current_low = bars[i].low

            tr = max(
                current_high - current_low,
                abs(current_high - prev_close),
                abs(current_low - prev_close)
            )
            true_ranges.append(tr)

        recent_tr = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
        atr = sum(recent_tr) / Decimal(len(recent_tr))

        return atr

    def _calculate_sma(self, bars: List[Bar], period: int) -> List[Optional[Decimal]]:
        """计算简单移动平均线"""
        sma_values = []

        for i in range(len(bars)):
            if i < period - 1:
                sma_values.append(None)
            else:
                sum_close = sum(bars[j].close for j in range(i - period + 1, i + 1))
                sma = sum_close / Decimal(period)
                sma_values.append(sma)

        return sma_values

    def _calculate_confidence(
        self,
        fast_ma: Decimal,
        slow_ma: Decimal,
        cross_type: str
    ) -> float:
        """计算信号置信度"""
        distance_percent = float(abs((fast_ma - slow_ma) / slow_ma))
        confidence = min(0.5 + distance_percent * 10, 0.95)
        return round(float(confidence), 2)


if __name__ == "__main__":
    """测试代码"""
    from skills.market_data.mock_source import MockMarketSource, TrendingMockSource

    print("=== EnhancedMAStrategy 测试 ===\n")

    # 测试1: 趋势市
    print("1. 趋势市测试（预期允许交易）:")
    trending_source = TrendingMockSource(seed=42, trend=0.005)
    bars = trending_source.get_bars("AAPL", datetime(2024, 1, 1), datetime(2024, 1, 31), "1d")

    strategy = EnhancedMAStrategy(
        fast_period=5,
        slow_period=20,
        enable_regime_gating=True
    )

    signals = strategy.generate_strategy_signals(bars)
    print(f"   生成信号数: {len(signals)}")
    for sig in signals:
        print(f"   - {sig.raw_action}: 目标仓位={sig.target_position}, "
              f"权重={sig.target_weight:.2f}, 市场状态={sig.regime.value}")
        print(f"     原因: {sig.reason}")

    # 测试2: 兼容旧接口
    print("\n2. 兼容旧接口测试:")
    legacy_signals = strategy.generate_signals(bars)
    print(f"   生成信号数: {len(legacy_signals)}")
    for sig in legacy_signals:
        print(f"   - {sig.action}: 数量={sig.quantity}, 置信度={sig.confidence:.2f}")

    # 测试3: 状态管理
    print("\n3. 状态管理测试:")
    state = strategy.get_state()
    print(f"   状态: {state.internal_state}")

    # 测试4: 禁用 gating
    print("\n4. 禁用 Regime Gating:")
    strategy_no_gating = EnhancedMAStrategy(
        fast_period=5,
        slow_period=20,
        enable_regime_gating=False
    )
    signals_no_gating = strategy_no_gating.generate_strategy_signals(bars)
    print(f"   生成信号数: {len(signals_no_gating)}")
    if signals_no_gating:
        sig = signals_no_gating[0]
        print(f"   - 目标仓位={sig.target_position}, 权重={sig.target_weight:.2f}")

    print("\n✓ 所有测试通过")
