"""
模拟交易执行器

实现虚拟账户的订单执行、持仓管理和资金管理
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, List, Literal, Any
from datetime import datetime

from core.models import OrderIntent, Trade, Position
from typing import cast


@dataclass(frozen=True)
class Order:
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    submit_ts: datetime


@dataclass(frozen=True)
class OrderFill:
    order: Order
    trade: Trade


@dataclass(frozen=True)
class OrderReject:
    order: Order
    reason: str


class PaperBroker:
    """模拟交易经纪商

    功能：
        - 维护虚拟账户和资金
        - 接受订单并在下一根bar撮合
        - 计算手续费
        - 更新持仓和资金
        - 可配置的BPS滑点模型
    """

    def __init__(
        self,
        initial_cash: float = 1000000.0,
        commission_per_share: float = 0.01,
        slippage_bps: float = 0.0
    ):
        """初始化模拟账户

        Args:
            initial_cash: 初始资金（默认100万美元）
            commission_per_share: 每股手续费（默认0.01美元）
            slippage_bps: 滑点（基点，1 BPS = 0.01%，默认0表示无滑点）
                          例如：slippage_bps=5 表示0.05%的滑点
        """
        self.initial_cash = Decimal(str(initial_cash))
        self.cash = self.initial_cash
        self.commission_per_share = Decimal(str(commission_per_share))
        self.slippage_bps = Decimal(str(slippage_bps))  # BPS滑点

        # 持仓字典
        self.positions: Dict[str, Position] = {}

        # 成交记录
        self.trades: List[Trade] = []

        # 订单计数器
        self._order_counter = 0
        self._trade_counter = 0

        # 挂单队列
        self.pending_orders: Dict[str, List[Order]] = {}

    def submit_order(self, intent: OrderIntent) -> Optional[Order]:
        """提交订单（市价单，在下一根bar成交）"""
        if not intent.can_execute:
            return None

        signal = intent.signal
        if signal.action not in ["BUY", "SELL"]:
            return None

        order = Order(
            order_id=self._generate_order_id(),
            symbol=signal.symbol,
            side=cast(Literal["BUY", "SELL"], signal.action),
            quantity=signal.quantity,
            submit_ts=intent.timestamp
        )

        self.pending_orders.setdefault(order.symbol, []).append(order)
        return order

    def on_bar(self, bar) -> tuple[List[OrderFill], List[OrderReject]]:
        """撮合挂单：仅允许在下一根bar成交"""
        fills: List[OrderFill] = []
        rejects: List[OrderReject] = []

        symbol = bar.symbol
        if symbol not in self.pending_orders:
            return fills, rejects

        remaining_orders: List[Order] = []
        for order in self.pending_orders[symbol]:
            if order.submit_ts >= bar.timestamp:
                remaining_orders.append(order)
                continue

            trade, reject_reason = self._fill_order(order, bar)
            if trade:
                fills.append(OrderFill(order=order, trade=trade))
            else:
                rejects.append(OrderReject(order=order, reason=reject_reason))

        if remaining_orders:
            self.pending_orders[symbol] = remaining_orders
        else:
            del self.pending_orders[symbol]

        return fills, rejects

    def _fill_order(self, order: Order, bar) -> tuple[Optional[Trade], str]:
        # 计算执行价格（BPS滑点）
        price = bar.open

        # 应用滑点：买入价格偏高，卖出价格偏低
        if self.slippage_bps > 0:
            slippage_amount = price * self.slippage_bps / Decimal("10000")
            if order.side == "BUY":
                price = price + slippage_amount  # 买入滑点使价格更高
            else:
                price = price - slippage_amount  # 卖出滑点使价格更低

        commission = self.commission_per_share * Decimal(order.quantity)

        if order.side == "BUY":
            required_amount = price * order.quantity + commission
            if required_amount > self.cash:
                return None, "insufficient_cash"

            self.cash -= required_amount
            self._update_position_buy(order.symbol, price, order.quantity)
        else:
            if order.symbol not in self.positions:
                return None, "no_position"
            current_pos = self.positions[order.symbol]
            if current_pos.quantity < order.quantity:
                return None, "insufficient_position"

            sell_amount = price * order.quantity - commission
            self.cash += sell_amount
            self._update_position_sell(order.symbol, price, order.quantity)

        trade = Trade(
            trade_id=self._generate_trade_id(bar.timestamp),
            symbol=order.symbol,
            side=order.side,
            price=price,
            quantity=order.quantity,
            timestamp=bar.timestamp,
            commission=commission
        )

        self.trades.append(trade)
        return trade, ""

    def _update_position_buy(self, symbol: str, price: Decimal, quantity: int):
        if symbol in self.positions:
            old_pos = self.positions[symbol]
            old_quantity = old_pos.quantity
            old_avg_price = old_pos.avg_price

            total_cost = (old_avg_price * old_quantity + price * quantity)
            new_quantity = old_quantity + quantity
            new_avg_price = total_cost / new_quantity

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
            market_value = price * quantity
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                market_value=market_value.quantize(Decimal("0.01")),
                unrealized_pnl=Decimal("0.00")
            )

    def _update_position_sell(self, symbol: str, price: Decimal, quantity: int):
        current_pos = self.positions[symbol]
        new_quantity = current_pos.quantity - quantity

        if new_quantity == 0:
            del self.positions[symbol]
        else:
            market_value = new_quantity * price
            unrealized_pnl = (price - current_pos.avg_price) * new_quantity

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=new_quantity,
                avg_price=current_pos.avg_price,
                market_value=market_value.quantize(Decimal("0.01")),
                unrealized_pnl=unrealized_pnl.quantize(Decimal("0.01"))
            )

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

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """撤销订单

        Args:
            order_id: 订单 ID
            client_order_id: 客户端订单 ID
            symbol: 撤销该品种的所有订单

        Returns:
            撤销结果
        """
        cancelled_orders = []

        if order_id:
            # 按 order_id 撤销 - 需要遍历所有品种的挂单
            for sym, orders in self.pending_orders.items():
                for i, order in enumerate(orders):
                    if order.order_id == order_id:
                        # 找到了，移除
                        self.pending_orders[sym].pop(i)
                        # 如果该品种没有挂单了，删除key
                        if not self.pending_orders[sym]:
                            del self.pending_orders[sym]
                        return {
                            "status": "success",
                            "reason": f"Order {order_id} cancelled",
                            "cancelled_orders": [order_id],
                            "updated_order": {
                                "order_id": order.order_id,
                                "symbol": order.symbol,
                                "side": order.side,
                                "quantity": order.quantity,
                                "status": "cancelled"
                            }
                        }
            return {
                "status": "error",
                "reason": f"Order {order_id} not found or already filled"
            }

        elif symbol:
            # 按 symbol 撤销所有订单
            symbol_upper = symbol.upper()
            if symbol_upper in self.pending_orders:
                orders = self.pending_orders.pop(symbol_upper)
                cancelled_ids = [o.order_id for o in orders]
                return {
                    "status": "success",
                    "reason": f"Cancelled {len(orders)} orders for {symbol_upper}",
                    "cancelled_orders": cancelled_ids
                }
            else:
                return {
                    "status": "error",
                    "reason": f"No pending orders found for {symbol_upper}"
                }

        elif client_order_id:
            # 按 client_order_id 撤销
            for sym, orders in list(self.pending_orders.items()):
                remaining = []
                cancelled = []
                for order in orders:
                    if getattr(order, 'client_order_id', None) == client_order_id:
                        cancelled.append(order.order_id)
                    else:
                        remaining.append(order)
                if cancelled:
                    self.pending_orders[sym] = remaining
                    if not remaining:
                        del self.pending_orders[sym]
                    return {
                        "status": "success",
                        "reason": f"Cancelled {len(cancelled)} orders with client_order_id={client_order_id}",
                        "cancelled_orders": cancelled
                    }
            return {
                "status": "error",
                "reason": f"No pending orders found with client_order_id={client_order_id}"
            }

        else:
            return {
                "status": "error",
                "reason": "At least one of order_id, client_order_id, or symbol must be provided"
            }

    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """获取所有挂单

        Returns:
            挂单列表
        """
        pending = []
        for symbol, orders in self.pending_orders.items():
            for order in orders:
                pending.append({
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "submit_ts": order.submit_ts.isoformat(),
                    "status": "pending"
                })
        return pending

    def reset(self):
        """重置账户到初始状态"""
        self.cash = self.initial_cash
        self.positions.clear()
        self.trades.clear()
        self.pending_orders.clear()
        self._order_counter = 0
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

    def _generate_order_id(self) -> str:
        self._order_counter += 1
        return f"O{self._order_counter:06d}"

    def _generate_trade_id(self, ts: datetime) -> str:
        self._trade_counter += 1
        return f"T{ts.strftime('%Y%m%d')}{self._trade_counter:06d}"


if __name__ == "__main__":
    """自测代码"""
    from decimal import Decimal
    from core.models import Signal

    print("=== PaperBroker 简单测试 ===\n")

    broker = PaperBroker(initial_cash=100000, commission_per_share=0.01)

    signal = Signal(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 2, 9, 30),
        action="BUY",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.8,
        reason="测试"
    )
    intent = OrderIntent(signal=signal, timestamp=signal.timestamp, approved=True, risk_reason="OK")
    order = broker.submit_order(intent)

    from core.models import Bar

    bar_t0 = Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 2, 9, 30),
        open=Decimal("150.00"),
        high=Decimal("151.00"),
        low=Decimal("149.00"),
        close=Decimal("150.50"),
        volume=100000,
        interval="1d"
    )
    fills, rejects = broker.on_bar(bar_t0)
    print(f"同bar撮合: fills={len(fills)}, rejects={len(rejects)}")

    bar_t1 = Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 3, 9, 30),
        open=Decimal("151.00"),
        high=Decimal("152.00"),
        low=Decimal("150.00"),
        close=Decimal("151.50"),
        volume=100000,
        interval="1d"
    )
    fills, rejects = broker.on_bar(bar_t1)
    print(f"下一根bar撮合: fills={len(fills)}, rejects={len(rejects)}")
