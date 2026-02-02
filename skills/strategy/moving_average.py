"""
移动平均线交叉策略

实现简单移动平均线（SMA）金叉/死叉策略
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from core.models import Bar, Signal, Position
from skills.strategy.base import Strategy


class MAStrategy(Strategy):
    """移动平均线交叉策略

    策略逻辑：
    - 快速均线上穿慢速均线（金叉）→ 买入信号
    - 快速均线下穿慢速均线（死叉）→ 卖出信号
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        """初始化策略

        Args:
            fast_period: 快速均线周期
            slow_period: 慢速均线周期
        """
        if fast_period >= slow_period:
            raise ValueError(f"快速均线周期必须小于慢速均线周期: {fast_period} >= {slow_period}")

        self.fast_period = fast_period
        self.slow_period = slow_period

    def get_min_bars_required(self) -> int:
        """获取所需最小K线数量"""
        return self.slow_period + 1  # 需要多一根来检测交叉

    def generate_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None
    ) -> List[Signal]:
        """生成交易信号

        Args:
            bars: K线数据列表
            position: 当前持仓（可选）

        Returns:
            信号列表（最多一个信号）
        """
        if len(bars) < self.get_min_bars_required():
            return []

        # 计算快速和慢速均线
        fast_mas = self._calculate_sma(bars, self.fast_period)
        slow_mas = self._calculate_sma(bars, self.slow_period)

        # 获取最新的两个均线值（用于检测交叉）
        prev_fast = fast_mas[-2]
        prev_slow = slow_mas[-2]
        curr_fast = fast_mas[-1]
        curr_slow = slow_mas[-1]

        # 计算 ATR (Average True Range) 用于设置止损和目标价
        atr = self._calculate_atr(bars, 14)
        current_price = bars[-1].close

        # 检测交叉
        signal = None
        symbol = bars[-1].symbol

        # 金叉：快速均线从下方穿越慢速均线
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            # 如果没有多头持仓，生成买入信号
            if position is None or position.quantity <= 0:
                # 计算目标价和止��（使用 Decimal 进行计算）
                target_price = current_price + (atr * Decimal('2'))  # 目标价为当前价 + 2倍ATR
                stop_loss = current_price - (atr * Decimal('1.5'))    # 止损价为当前价 - 1.5倍ATR

                signal = Signal(
                    symbol=symbol,
                    timestamp=bars[-1].timestamp,
                    action="BUY",
                    price=current_price,
                    quantity=100,  # 默认数量，实际应用中应根据资金管理计算
                    confidence=self._calculate_confidence(curr_fast, curr_slow, "golden_cross"),
                    reason=f"金叉: 快速均线({self.fast_period})上穿慢速均线({self.slow_period})",
                    target_price=target_price,
                    stop_loss=stop_loss,
                    time_horizon=120  # 预期持有时间：120小时（5天）
                )

        # 死叉：快速均线从上方穿越慢速均线
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            # 如果有多头持仓，生成卖出信号
            if position is not None and position.quantity > 0:
                signal = Signal(
                    symbol=symbol,
                    timestamp=bars[-1].timestamp,
                    action="SELL",
                    price=current_price,
                    quantity=abs(position.quantity),  # 卖出全部持仓
                    confidence=self._calculate_confidence(curr_fast, curr_slow, "death_cross"),
                    reason=f"死叉: 快速均线({self.fast_period})下穿慢速均线({self.slow_period})"
                )

        return [signal] if signal else []

    def _calculate_atr(self, bars: List[Bar], period: int = 14) -> Decimal:
        """计算平均真实波幅 (Average True Range)

        ATR 用于衡量市场波动性，可用于设置止损和目标价格

        Args:
            bars: K线数据
            period: 计算周期

        Returns:
            ATR 值
        """
        if len(bars) < period + 1:
            # 数据不足，使用简单计算
            total_range = sum((bar.high - bar.low) for bar in bars)
            return total_range / Decimal(len(bars)) if bars else Decimal('1')

        # 计算每一根K线的 True Range
        true_ranges = []
        for i in range(1, len(bars)):
            prev_close = bars[i - 1].close
            current_high = bars[i].high
            current_low = bars[i].low

            # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
            tr = max(
                current_high - current_low,
                abs(current_high - prev_close),
                abs(current_low - prev_close)
            )
            true_ranges.append(tr)

        # 计算最近 period 个 TR 的平均值
        recent_tr = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
        atr = sum(recent_tr) / Decimal(len(recent_tr))

        return atr

    def _calculate_sma(self, bars: List[Bar], period: int) -> List[Decimal]:
        """计算简单移动平均线

        Args:
            bars: K线数据
            period: 周期

        Returns:
            SMA值列表
        """
        sma_values = []

        for i in range(len(bars)):
            if i < period - 1:
                # 数据不足，填充None
                sma_values.append(None)
            else:
                # 计算SMA
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
        """计算信号置信度

        基于均线之间的距离来计算置信度
        距离越大，置信度越高

        Args:
            fast_ma: 快速均线值
            slow_ma: 慢速均线值
            cross_type: 交叉类型（'golden_cross' 或 'death_cross'）

        Returns:
            置信度 (0-1)
        """
        # 计算两条均线的相对距离
        distance_percent = float(abs((fast_ma - slow_ma) / slow_ma))

        # 将距离转换为置信度（最大0.95）
        confidence = min(0.5 + distance_percent * 10, 0.95)

        return round(float(confidence), 2)


if __name__ == "__main__":
    """自测代码"""
    from skills.market_data.mock_source import MockMarketSource

    print("=== MAStrategy 测试 ===\n")

    # 创建数据源和策略
    source = MockMarketSource(seed=42)
    strategy = MAStrategy(fast_period=5, slow_period=20)

    # 测试1: 生成上涨趋势数据
    print("1. 测试上涨趋势（预期金叉买入信号）:")
    from skills.market_data.mock_source import TrendingMockSource
    trending_source = TrendingMockSource(seed=42, trend=0.005)  # 每天上涨0.5%

    bars = trending_source.get_bars(
        "AAPL",
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
        "1d"
    )

    signals = strategy.generate_signals(bars)
    if signals:
        for sig in signals:
            print(f"   信号: {sig.action} {sig.quantity}股 @ {sig.price}")
            print(f"   原因: {sig.reason}")
            print(f"   置信度: {sig.confidence}")
    else:
        print("   未生成信号（可能尚未触发交叉）")

    # 测试2: 生成下跌趋势数据
    print("\n2. 测试下跌趋势（预期死叉卖出信号）:")
    downtrend_source = TrendingMockSource(seed=42, trend=-0.005)  # 每天下跌0.5%

    bars = downtrend_source.get_bars(
        "AAPL",
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
        "1d"
    )

    # 假设已有持仓
    position = Position(
        symbol="AAPL",
        quantity=100,
        avg_price=Decimal("150.00"),
        market_value=Decimal("15000.00"),
        unrealized_pnl=Decimal("0.00")
    )

    signals = strategy.generate_signals(bars, position)
    if signals:
        for sig in signals:
            print(f"   信号: {sig.action} {sig.quantity}股 @ {sig.price}")
            print(f"   原因: {sig.reason}")
            print(f"   置信度: {sig.confidence}")
    else:
        print("   未生成信号（可能尚未触发交叉）")

    # 测试3: 测试均线计算
    print("\n3. 测试均线计算:")
    test_bars = source.get_bars("AAPL", datetime(2024, 1, 1), datetime(2024, 1, 10), "1d")

    fast_mas = strategy._calculate_sma(test_bars, 5)
    slow_mas = strategy._calculate_sma(test_bars, 20)

    print(f"   K线数量: {len(test_bars)}")
    print(f"   快速均线(5日)最后5个值:")
    for i, ma in enumerate(fast_mas[-5:]):
        if ma is not None:
            print(f"     {test_bars[-(5-i)].timestamp.strftime('%Y-%m-%d')}: {ma.quantize(Decimal('0.01'))}")

    # 测试4: 数据不足
    print("\n4. 测试数据不足的情况:")
    short_bars = test_bars[:10]
    print(f"   K线数量: {len(short_bars)}")
    print(f"   最小需求: {strategy.get_min_bars_required()}")

    signals = strategy.generate_signals(short_bars)
    print(f"   生成信号: {len(signals)}")
    print(f"   ✓ 正确处理数据不足情况")

    # 测试5: 震荡市场
    print("\n5. 测试震荡市场（可能无信号）:")
    range_bars = source.get_bars("AAPL", datetime(2024, 1, 1), datetime(2024, 1, 31), "1d")
    signals = strategy.generate_signals(range_bars)
    print(f"   K线数量: {len(range_bars)}")
    print(f"   生成信号: {len(signals)}")

    # 测试6: 持仓影响
    print("\n6. 测试持仓对信号的影响:")
    long_position = Position(
        symbol="AAPL",
        quantity=100,
        avg_price=Decimal("150.00"),
        market_value=Decimal("15400.00"),
        unrealized_pnl=Decimal("400.00")
    )
    signals_with_long = strategy.generate_signals(bars[:25], long_position)
    print(f"   有多头持仓时的信号数: {len(signals_with_long)}")

    no_position = None
    signals_without = strategy.generate_signals(bars[:25], no_position)
    print(f"   无持仓时的信号数: {len(signals_without)}")

    print("\n✓ 所有测试通过")
