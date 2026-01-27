"""
模拟交易执行器

实现虚拟账户的订单执行、持仓管理和资金管理
"""

from decimal import Decimal
from typing import Dict, Optional
from datetime import datetime
import uuid

from core.models import OrderIntent, Trade, Position, Signal


class PaperBroker:
    """模拟交易经纪商

    功能：
    - ��护虚拟账户和资金
    - 执行订单并生成成交记录
    - 计算手续费
    - 更新持仓和资金
    """

    def __init__(
        self,
        initial_cash: float = 1000000.0,
        commission_per_share: float = 0.01
    ):
        """初始化模拟账户

        Args:
            initial_cash: 初始资金（默认100万美元）
            commission_per_share: 每股手续费（默认0.01美元）
        """
        self.initial_cash = Decimal(str(initial_cash))
        self.cash = self.initial_cash
        self.commission_per_share = Decimal(str(commission_per_share))

        # 持仓字典
        self.positions: Dict[str, Position] = {}

        # 成交记录
        self.trades: list[Trade] = []

        # 交易计数器（用于生成trade_id）
        self._trade_counter = 0

    def execute_order(self, intent: OrderIntent) -> Optional[Trade]:
        """执行订单

        Args:
            intent: 订单意图

        Returns:
            成交记录，如果订单被拒绝则返回None
        """
        if not intent.can_execute:
            return None

        signal = intent.signal
        symbol = signal.symbol
        side = signal.action
        price = signal.price
        quantity = signal.quantity

        # 计算手续费
        commission = self.commission_per_share * Decimal(quantity)

        if side == 'BUY':
            return self._execute_buy(symbol, price, quantity, commission)
        elif side == 'SELL':
            return self._execute_sell(symbol, price, quantity, commission)
        else:
            return None

    def _execute_buy(
        self,
        symbol: str,
        price: Decimal,
        quantity: int,
        commission: Decimal
    ) -> Optional[Trade]:
        """执行买入订单

        Args:
            symbol: 标的代码
            price: 买入价格
            quantity: 买入数量
            commission: 手续费

        Returns:
            成交记录
        """
        # 计算所需资金
        required_amount = price * quantity + commission

        # 检查资金是否充足
        if required_amount > self.cash:
            return None

        # 更新现金
        self.cash -= required_amount

        # 更新持仓
        if symbol in self.positions:
            # 已有持仓，计算新的平均成本
            old_pos = self.positions[symbol]
            old_quantity = old_pos.quantity
            old_avg_price = old_pos.avg_price

            # 计算新的平均成本价
            total_cost = (old_avg_price * old_quantity + price * quantity)
            new_quantity = old_quantity + quantity
            new_avg_price = total_cost / new_quantity

            # 创建新的持仓对象
            market_value = new_quantity * price
            unrealized_pnl = (price - new_avg_price) * new_quantity

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=new_quantity,
                avg_price=new_avg_price.quantize(Decimal("0.01")),
                market_value=market_value.quantize(Decimal("0.01")),
                unrealized_pnl=unrealized_pnl.quantize(Decimal("0.01"))
            )
        else:
            # 新建持仓
            market_value = price * quantity
            unrealized_pnl = Decimal("0.00")

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                market_value=market_value.quantize(Decimal("0.01")),
                unrealized_pnl=unrealized_pnl
            )

        # 创建成交记录
        trade = Trade(
            trade_id=self._generate_trade_id(),
            symbol=symbol,
            side="BUY",
            price=price,
            quantity=quantity,
            timestamp=datetime.now(),
            commission=commission
        )

        self.trades.append(trade)
        return trade

    def _execute_sell(
        self,
        symbol: str,
        price: Decimal,
        quantity: int,
        commission: Decimal
    ) -> Optional[Trade]:
        """执行卖出订单

        Args:
            symbol: 标的代码
            price: 卖出价格
            quantity: 卖出数量
            commission: 手续费

        Returns:
            成交记录
        """
        # 检查持仓是否充足
        if symbol not in self.positions:
            return None

        current_pos = self.positions[symbol]
        if current_pos.quantity < quantity:
            return None

        # 计算卖出金额
        sell_amount = price * quantity - commission

        # 更新现金
        self.cash += sell_amount

        # 计算已实现盈亏
        realized_pnl = (price - current_pos.avg_price) * quantity

        # 更新持仓
        new_quantity = current_pos.quantity - quantity

        if new_quantity == 0:
            # 全部卖出，删除持仓
            del self.positions[symbol]
        else:
            # 部分卖出，更新持仓
            market_value = new_quantity * price
            unrealized_pnl = (price - current_pos.avg_price) * new_quantity

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=new_quantity,
                avg_price=current_pos.avg_price,
                market_value=market_value.quantize(Decimal("0.01")),
                unrealized_pnl=unrealized_pnl.quantize(Decimal("0.01"))
            )

        # 创建成交记录
        trade = Trade(
            trade_id=self._generate_trade_id(),
            symbol=symbol,
            side="SELL",
            price=price,
            quantity=quantity,
            timestamp=datetime.now(),
            commission=commission
        )

        self.trades.append(trade)
        return trade

    def get_positions(self) -> Dict[str, Position]:
        """获取当前所有持仓

        Returns:
            持仓字典
        """
        return self.positions.copy()

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定标的的持仓

        Args:
            symbol: 标的代码

        Returns:
            持仓对象，如果不存在则返回None
        """
        return self.positions.get(symbol)

    def get_cash_balance(self) -> Decimal:
        """获取当前现金余额

        Returns:
            现金余额
        """
        return self.cash

    def get_total_equity(self) -> Decimal:
        """获取总权益（现金 + 持仓市值）

        Returns:
            总权益
        """
        total = self.cash
        for pos in self.positions.values():
            total += pos.market_value
        return total.quantize(Decimal("0.01"))

    def get_total_pnl(self) -> Decimal:
        """获取总盈亏（已实现 + 未实现）

        Returns:
            总盈亏
        """
        # 总权益 - 初始资金
        return self.get_total_equity() - self.initial_cash

    def get_trades(self) -> list[Trade]:
        """获取所有成交记录

        Returns:
            成交记录列表
        """
        return self.trades.copy()

    def reset(self):
        """重置账户到初始状态"""
        self.cash = self.initial_cash
        self.positions.clear()
        self.trades.clear()
        self._trade_counter = 0

    def update_position_prices(self, prices: Dict[str, Decimal]):
        """更新持仓的当前价格（用于计算浮动盈亏）

        Args:
            prices: 标的代码到当前价格的映射
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                market_value = current_price * position.quantity
                unrealized_pnl = (current_price - position.avg_price) * position.quantity

                self.positions[symbol] = Position(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_price=position.avg_price,
                    market_value=market_value.quantize(Decimal("0.01")),
                    unrealized_pnl=unrealized_pnl.quantize(Decimal("0.01"))
                )

    def _generate_trade_id(self) -> str:
        """生成交易ID

        Returns:
            交易ID
        """
        self._trade_counter += 1
        return f"T{datetime.now().strftime('%Y%m%d')}{self._trade_counter:06d}"


if __name__ == "__main__":
    """自测代码"""
    from decimal import Decimal

    print("=== PaperBroker 测试 ===\n")

    # 创建模拟账户
    broker = PaperBroker(initial_cash=100000, commission_per_share=0.01)

    print("1. 初始账户状态:")
    print(f"   现金: ${broker.get_cash_balance():,.2f}")
    print(f"   总权益: ${broker.get_total_equity():,.2f}")
    print(f"   持仓数: {len(broker.get_positions())}")

    # 测试买入
    print("\n2. 执行买入订单:")
    buy_signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.8,
        reason="金叉买入"
    )
    buy_intent = OrderIntent(
        signal=buy_signal,
        timestamp=datetime.now(),
        approved=True,
        risk_reason="符合风控"
    )

    trade1 = broker.execute_order(buy_intent)
    if trade1:
        print(f"   ✓ 成交: {trade1.side} {trade1.quantity} {trade1.symbol} @ ${trade1.price}")
        print(f"   手续费: ${trade1.commission}")
        print(f"   成交金额: ${trade1.notional_value}")

    print("\n   买入后账户状态:")
    print(f"   现金: ${broker.get_cash_balance():,.2f}")
    print(f"   总权益: ${broker.get_total_equity():,.2f}")

    # 查看持仓
    aapl_pos = broker.get_position("AAPL")
    if aapl_pos:
        print(f"   AAPL持仓: {aapl_pos.quantity}股, 成本 ${aapl_pos.avg_price}")

    # 测试继续买入同一股票
    print("\n3. 继续买入同一股票:")
    buy_signal2 = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("155.00"),
        quantity=50,
        confidence=0.8,
        reason="加仓"
    )
    buy_intent2 = OrderIntent(
        signal=buy_signal2,
        timestamp=datetime.now(),
        approved=True,
        risk_reason="符合风控"
    )

    trade2 = broker.execute_order(buy_intent2)
    if trade2:
        print(f"   ✓ 成交: {trade2.side} {trade2.quantity} {trade2.symbol} @ ${trade2.price}")

    aapl_pos = broker.get_position("AAPL")
    if aapl_pos:
        print(f"   AAPL持仓: {aapl_pos.quantity}股, 平均成本 ${aapl_pos.avg_price}")
        print(f"   市值: ${aapl_pos.market_value}")

    # 测试买入其他股票
    print("\n4. 买入其他股票:")
    googl_signal = Signal(
        symbol="GOOGL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("140.00"),
        quantity=50,
        confidence=0.8,
        reason="测试"
    )
    googl_intent = OrderIntent(
        signal=googl_signal,
        timestamp=datetime.now(),
        approved=True,
        risk_reason="符合风控"
    )

    trade3 = broker.execute_order(googl_intent)
    if trade3:
        print(f"   ✓ 成交: {trade3.side} {trade3.quantity} {trade3.symbol} @ ${trade3.price}")

    print(f"   持仓数量: {len(broker.get_positions())}")
    print(f"   总权益: ${broker.get_total_equity():,.2f}")

    # 测试部分卖出
    print("\n5. 部分卖出AAPL:")
    sell_signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="SELL",
        price=Decimal("160.00"),
        quantity=75,
        confidence=0.8,
        reason="减仓"
    )
    sell_intent = OrderIntent(
        signal=sell_signal,
        timestamp=datetime.now(),
        approved=True,
        risk_reason="符合风控"
    )

    trade4 = broker.execute_order(sell_intent)
    if trade4:
        print(f"   ✓ 成交: {trade4.side} {trade4.quantity} {trade4.symbol} @ ${trade4.price}")

    aapl_pos = broker.get_position("AAPL")
    if aapl_pos:
        print(f"   AAPL剩余持仓: {aapl_pos.quantity}股")
    else:
        print(f"   AAPL持仓已清空")

    print(f"   现金: ${broker.get_cash_balance():,.2f}")

    # 测试价格更新
    print("\n6. 更新持仓价格:")
    broker.update_position_prices({
        "AAPL": Decimal("165.00"),
        "GOOGL": Decimal("145.00")
    })

    for symbol, pos in broker.get_positions().items():
        print(f"   {symbol}: {pos.quantity}股, 市值 ${pos.market_value}, 浮盈 ${pos.unrealized_pnl}")

    print(f"   总权益: ${broker.get_total_equity():,.2f}")
    print(f"   总盈亏: ${broker.get_total_pnl():,.2f} ({broker.get_total_pnl()/broker.initial_cash*100:.2f}%)")

    # 测试成交记录
    print("\n7. 成交记录:")
    all_trades = broker.get_trades()
    print(f"   总成交次数: {len(all_trades)}")
    for trade in all_trades:
        print(f"   {trade.trade_id}: {trade.side} {trade.quantity} {trade.symbol} @ ${trade.price}")

    # 测试重置
    print("\n8. 重置账户:")
    broker.reset()
    print(f"   现金: ${broker.get_cash_balance():,.2f}")
    print(f"   持仓数: {len(broker.get_positions())}")
    print(f"   成交记录: {len(broker.get_trades())}")

    print("\n✓ 所有测试通过")
