"""
突破策略

当价格突破N日高点时买入，突破N日低点时卖出
"""

from decimal import Decimal
from typing import List

from core.models import Bar, Signal
from skills.strategy.base import Strategy


class BreakoutStrategy(Strategy):
    """突破策略

    价格突破N日高点买入，突破N日低点卖出
    """

    def __init__(self, period: int = 20, threshold: float = 0.01):
        """初始化突破策略

        Args:
            period: 突破周期（默认20日）
            threshold: 突破阈值（默认1%）
        """
        self.period = period
        self.threshold = threshold

    def get_min_bars_required(self) -> int:
        """获取所需的最小K线数量"""
        return self.period + 1

    def generate_signals(self, bars: List[Bar], position=None) -> List[Signal]:
        """生成交易信号

        Args:
            bars: K线列表
            position: 当前持仓（可选）

        Returns:
            信号列表
        """
        signals = []

        if len(bars) < self.get_min_bars_required():
            return signals

        current_bar = bars[-1]
        symbol = current_bar.symbol

        # 计算N日高低点
        lookback_bars = bars[-(self.period + 1):-1]
        high_n = max(b.high for b in lookback_bars)
        low_n = min(b.low for b in lookback_bars)

        # 突破检测
        close = current_bar.close
        threshold = Decimal(str(self.threshold))

        # 向上突破
        if close > high_n * (Decimal("1") + threshold):
            # 检查是否已有持仓
            if position is None or position.quantity == 0:
                signals.append(Signal(
                    symbol=symbol,
                    timestamp=current_bar.timestamp,
                    action="BUY",
                    price=close,
                    quantity=100,  # 固定数量
                    confidence=min(1.0, float((close - high_n) / high_n * 10)),
                    reason=f"向上突破{self.period}日高点 {high_n:.2f}"
                ))

        # 向下突破
        elif close < low_n * (Decimal("1") - threshold):
            # 检查是否是否有多头持仓
            if position is not None and position.quantity > 0:
                signals.append(Signal(
                    symbol=symbol,
                    timestamp=current_bar.timestamp,
                    action="SELL",
                    price=close,
                    quantity=position.quantity,
                    confidence=min(1.0, float((low_n - close) / low_n * 10)),
                    reason=f"向下突破{self.period}日低点 {low_n:.2f}"
                ))

        return signals


if __name__ == "__main__":
    """测试代码"""
    from datetime import datetime, timedelta

    print("=== BreakoutStrategy 测试 ===\n")

    # 生成测试数据
    base_price = Decimal("100.0")
    bars = []
    base_date = datetime(2024, 1, 1)

    # 前20天：震荡
    for i in range(20):
        date = base_date + timedelta(days=i)
        bars.append(Bar(
            symbol="TEST",
            timestamp=date,
            open=base_price + Decimal(str(i % 5)),
            high=base_price + Decimal(str(5 + i % 10)),
            low=base_price - Decimal(str(5 + i % 10)),
            close=base_price,
            volume=1000000,
            interval="1d"
        ))

    # 第21天：向上突破
    bars.append(Bar(
        symbol="TEST",
        timestamp=base_date + timedelta(days=20),
        open=base_price,
        high=base_price + Decimal("15"),
        low=base_price - Decimal("5"),
        close=base_price + Decimal("12"),
        volume=1000000,
        interval="1d"
    ))

    # 创建策略
    strategy = BreakoutStrategy(period=20, threshold=0.01)

    # 生成信号
    signals = strategy.generate_signals(bars)

    print(f"生成信号数: {len(signals)}")
    for sig in signals:
        print(f"  {sig.action}: {sig.reason} (置信度: {sig.confidence:.2%})")

    print("\n✓ 测试通过")
