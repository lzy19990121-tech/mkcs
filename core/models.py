"""
核心数据模型定义

使用dataclass定义所有核心数据结构，确保类型安全和不可变性。
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, Dict, List, Optional


@dataclass(frozen=True)
class Bar:
    """K线数据

    Attributes:
        symbol: 标的代码
        timestamp: 时间戳
        open: 开盘价
        high: 最高价
        low: 最低价
        close: 收盘价
        volume: 成交量
        interval: 时间周期（1m, 5m, 1h, 1d）
    """
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    interval: str

    def __post_init__(self):
        """验证数据有效性"""
        if self.high < self.low:
            raise ValueError(f"最高价不能低于最低价: {self.high} < {self.low}")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"最高价必须>=开盘价和收盘价")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"最低价必须<=开盘价和收盘价")
        if self.volume < 0:
            raise ValueError(f"成交量不能为负数: {self.volume}")
        if self.interval not in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
            raise ValueError(f"无效的时间周期: {self.interval}")


@dataclass(frozen=True)
class Quote:
    """实时报价

    Attributes:
        symbol: 标的代码
        timestamp: 时间戳
        bid_price: 买一价
        ask_price: 卖一价
        bid_size: 买一量
        ask_size: 卖一量
    """
    symbol: str
    timestamp: datetime
    bid_price: Decimal
    ask_price: Decimal
    bid_size: int
    ask_size: int
    prev_close: Optional[Decimal] = None

    def __post_init__(self):
        """验证数据有效性"""
        if self.bid_price <= 0 or self.ask_price <= 0:
            raise ValueError(f"价格必须为正数: bid={self.bid_price}, ask={self.ask_price}")
        if self.ask_price < self.bid_price:
            raise ValueError(f"卖价不能低于买价: ask={self.ask_price} < bid={self.bid_price}")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError(f"委托量不能为负数: bid_size={self.bid_size}, ask_size={self.ask_size}")

    @property
    def spread(self) -> Decimal:
        """买卖价差"""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> Decimal:
        """中间价"""
        return (self.bid_price + self.ask_price) / 2


@dataclass(frozen=True)
class Signal:
    """交易信号

    Attributes:
        symbol: 标的代码
        timestamp: 时间戳
        action: 操作类型（BUY/SELL/HOLD）
        price: 建议价格
        quantity: 建议数量
        confidence: 信号强度 0-1
        reason: 信号原因
        target_price: 目标价格（止盈价，可选）
        stop_loss: 止损价（可选）
        time_horizon: 预期持有时间（小时，可选）
    """
    symbol: str
    timestamp: datetime
    action: Literal['BUY', 'SELL', 'HOLD']
    price: Decimal
    quantity: int
    confidence: float
    reason: str
    target_price: Decimal = None
    stop_loss: Decimal = None
    time_horizon: int = None

    def __post_init__(self):
        """验证数据有效性"""
        if self.action not in ['BUY', 'SELL', 'HOLD']:
            raise ValueError(f"无效的操作类型: {self.action}")
        if self.price <= 0:
            raise ValueError(f"价格必须为正数: {self.price}")
        if self.quantity <= 0:
            raise ValueError(f"数量必须为正数: {self.quantity}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"信号强度必须在0-1之间: {self.confidence}")
        # 验证目标价格和止损价
        if self.target_price is not None and self.target_price <= 0:
            raise ValueError(f"目标价格必须为正数: {self.target_price}")
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError(f"止损价必须为正数: {self.stop_loss}")
        # 买入信号：目标价 > 当前价 > 止损价
        if self.action == 'BUY':
            if self.target_price is not None and self.target_price <= self.price:
                raise ValueError(f"买入信号的目标价必须高于当前价: {self.target_price} <= {self.price}")
            if self.stop_loss is not None and self.stop_loss >= self.price:
                raise ValueError(f"买入信号的止损价必须低于当前价: {self.stop_loss} >= {self.price}")
        # 卖出信号：当前价 > 目标价 > 止损价（如果是做空）
        elif self.action == 'SELL':
            if self.target_price is not None and self.target_price >= self.price:
                raise ValueError(f"卖出信号的目标价必须低于当前价: {self.target_price} >= {self.price}")


@dataclass(frozen=True)
class OrderIntent:
    """订单意图（风险检查后）

    Attributes:
        signal: 原始交易信号
        timestamp: 时间戳
        approved: 是否通过风控
        risk_reason: 风控说明
    """
    signal: Signal
    timestamp: datetime
    approved: bool
    risk_reason: str

    @property
    def can_execute(self) -> bool:
        """是否可以执行"""
        return self.approved and self.signal.action != 'HOLD'


TradeSide = Literal['BUY', 'SELL']


@dataclass(frozen=True)
class Trade:
    """成交记录

    Attributes:
        trade_id: 成交ID
        symbol: 标的代码
        side: 方向（BUY/SELL）
        price: 成交价
        quantity: 成交量
        timestamp: 成交时间
        commission: 手续费
        realized_pnl: 已实现盈亏（可选）
    """
    trade_id: str
    symbol: str
    side: TradeSide
    price: Decimal
    quantity: int
    timestamp: datetime
    commission: Decimal
    realized_pnl: Decimal = Decimal("0")

    def __post_init__(self):
        """验证数据有效性"""
        if self.side not in ['BUY', 'SELL']:
            raise ValueError(f"无效的交易方向: {self.side}")
        if self.price <= 0:
            raise ValueError(f"价格必须为正数: {self.price}")
        if self.quantity <= 0:
            raise ValueError(f"数量必须为正数: {self.quantity}")
        if self.commission < 0:
            raise ValueError(f"手续费不能为负数: {self.commission}")

    @property
    def notional_value(self) -> Decimal:
        """成交金额"""
        return self.price * Decimal(self.quantity)


@dataclass(frozen=True)
class Position:
    """持仓信息

    Attributes:
        symbol: 标的代码
        quantity: 持仓数量（正数为多头，负数为空头）
        avg_price: 平均成本价
        market_value: 市值
        unrealized_pnl: 浮动盈亏
    """
    symbol: str
    quantity: int
    avg_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal

    def __post_init__(self):
        """验证数据有效性"""
        if self.quantity == 0:
            raise ValueError(f"持仓数量不能为0: {self.quantity}")
        if self.avg_price <= 0:
            raise ValueError(f"平均成本价必须为正数: {self.avg_price}")

    @property
    def is_long(self) -> bool:
        """是否多头"""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """是否空头"""
        return self.quantity < 0


# 类型别名
PositionDict = Dict[str, Position]
BarList = List[Bar]
SignalList = List[Signal]
TradeList = List[Trade]


if __name__ == "__main__":
    """自测代码"""
    from decimal import Decimal

    # 测试Bar
    print("测试 Bar:")
    bar = Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 1, 9, 30),
        open=Decimal("150.0"),
        high=Decimal("155.0"),
        low=Decimal("149.0"),
        close=Decimal("154.0"),
        volume=1000000,
        interval="1d"
    )
    print(f"  {bar}")
    print(f"  开盘价: {bar.open}, 收盘价: {bar.close}")

    # 测试Quote
    print("\n测试 Quote:")
    quote = Quote(
        symbol="AAPL",
        timestamp=datetime.now(),
        bid_price=Decimal("150.00"),
        ask_price=Decimal("150.05"),
        bid_size=1000,
        ask_size=1500
    )
    print(f"  {quote}")
    print(f"  价差: {quote.spread}, 中间价: {quote.mid_price}")

    # 测试Signal
    print("\n测试 Signal:")
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.85,
        reason="金叉买入信号"
    )
    print(f"  {signal}")
    print(f"  操作: {signal.action}, 数量: {signal.quantity}, 置信度: {signal.confidence}")

    # 测试OrderIntent
    print("\n测试 OrderIntent:")
    intent = OrderIntent(
        signal=signal,
        timestamp=datetime.now(),
        approved=True,
        risk_reason="符合风控规则"
    )
    print(f"  {intent}")
    print(f"  可执行: {intent.can_execute}")

    # 测试Trade
    print("\n测试 Trade:")
    trade = Trade(
        trade_id="T001",
        symbol="AAPL",
        side="BUY",
        price=Decimal("150.00"),
        quantity=100,
        timestamp=datetime.now(),
        commission=Decimal("1.00")
    )
    print(f"  {trade}")
    print(f"  成交金额: {trade.notional_value}")

    # 测试Position
    print("\n测试 Position:")
    position = Position(
        symbol="AAPL",
        quantity=100,
        avg_price=Decimal("150.00"),
        market_value=Decimal("15400.00"),
        unrealized_pnl=Decimal("400.00")
    )
    print(f"  {position}")
    print(f"  多头: {position.is_long}, 空头: {position.is_short}")

    # 测试异常情况
    print("\n测试异常检测:")
    try:
        Bar(
            symbol="INVALID",
            timestamp=datetime.now(),
            open=Decimal("150"),
            high=Decimal("140"),  # 最高价 < 最低价
            low=Decimal("145"),
            close=Decimal("142"),
            volume=1000,
            interval="1d"
        )
    except ValueError as e:
        print(f"  捕获到预期异常: {e}")

    print("\n✓ 所有数据模型测试通过")
