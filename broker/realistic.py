"""
订单执行模拟器

模拟真实的交易执行环境，包括滑点、市场冲击、部分成交等
"""

import logging
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

from core.models import Bar, OrderIntent, Signal, Trade, TradeSide

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"      # 市价单
    LIMIT = "limit"        # 限价单
    STOP_MARKET = "stop_market"  # 止损市价单


class ExecutionQuality(Enum):
    """执行质量"""
    EXCELLENT = "excellent"  # 优秀（滑点 < 0.01%）
    GOOD = "good"           # 良好（滑点 < 0.05%）
    FAIR = "fair"           # 一般（滑点 < 0.1%）
    POOR = "poor"           # 较差（滑点 < 0.5%）
    BAD = "bad"             # 很差（滑点 >= 0.5%）


@dataclass
class ExecutionResult:
    """执行结果"""
    trade: Optional[Trade]          # 成交记录
    filled_quantity: int            # 成交数量
    filled_price: Decimal           # 成交价格
    slippage: Decimal               # 滑点（绝对值）
    slippage_percent: Decimal       # 滑点百分比
    market_impact: Decimal          # 市场冲击
    execution_quality: ExecutionQuality
    reason: str                     # 未成交原因


class SlippageModel:
    """滑点模型"""

    @staticmethod
    def calculate_slippage(
        order_price: Decimal,
        side: str,
        quantity: int,
        current_bar: Bar,
        order_type: OrderType = OrderType.MARKET
    ) -> Decimal:
        """计算滑点

        Args:
            order_price: 订单价格
            side: 方向 (BUY/SELL)
            quantity: 订单数量
            current_bar: 当前 K 线
            order_type: 订单类型

        Returns:
            滑点（绝对值）
        """
        # 基础滑点（基于价格波动率）
        price_range = float(current_bar.high - current_bar.low)
        base_slippage = Decimal(str(price_range * 0.001))  # 0.1% 的波动范围

        # 数量相关的滑点（订单越大，滑点越大）
        volume = current_bar.volume
        volume_ratio = quantity / volume if volume > 0 else 0

        # 大单额外滑点（非线性增长）
        large_order_penalty = Decimal(str((volume_ratio ** 2) * price_range * 0.5))

        # 市价单 vs 限价单
        if order_type == OrderType.MARKET:
            multiplier = Decimal("1.0")
        else:
            multiplier = Decimal("0.5")  # 限价单滑点较小

        total_slippage = (base_slippage + large_order_penalty) * multiplier

        return max(total_slippage, Decimal("0"))

    @staticmethod
    def calculate_market_impact(
        side: str,
        quantity: int,
        current_bar: Bar,
        order_book_depth: Optional[Dict] = None
    ) -> Decimal:
        """计算市场冲击

        大单会对价格产生影响，使成交价格向不利方向移动

        Args:
            side: 方向
            quantity: 数量
            current_bar: 当前 K 线
            order_book_depth: 订单簿深度（可选）

        Returns:
            市场冲击（价格变动）
        """
        typical_price = (current_bar.high + current_bar.low + current_bar.close) / 3
        notional_value = float(typical_price * quantity)

        # 简化模型：冲击与交易金额的平方根成正比
        impact = Decimal(str((notional_value ** 0.5) * 0.0001))

        # 根据成交量调整（成交量越大，冲击越小）
        volume_factor = (100000 / current_bar.volume) ** 0.5
        impact = impact * Decimal(str(volume_factor))

        return max(impact, Decimal("0"))


class OrderBookSimulator:
    """订单簿模拟器"""

    def __init__(self, spread_percent: float = 0.001):
        """初始化订单簿

        Args:
            spread_percent: 买卖价差百分比（默认 0.1%）
        """
        self.spread_percent = spread_percent

    def get_bid_ask(self, current_bar: Bar) -> Tuple[Decimal, Decimal]:
        """获取买卖价

        Args:
            current_bar: 当前 K 线

        Returns:
            (bid_price, ask_price)
        """
        mid_price = current_bar.close
        half_spread = mid_price * Decimal(str(self.spread_percent / 2))

        bid_price = mid_price - half_spread
        ask_price = mid_price + half_spread

        return bid_price, ask_price

    def simulate_execution(
        self,
        side: str,
        quantity: int,
        price: Decimal,
        current_bar: Bar,
        order_type: OrderType
    ) -> ExecutionResult:
        """模拟订单执行

        Args:
            side: 买卖方向
            quantity: 数量
            price: 价格
            current_bar: 当前 K 线
            order_type: 订单类型

        Returns:
            执行结果
        """
        # 计算滑点
        slippage = SlippageModel.calculate_slippage(
            price, side, quantity, current_bar, order_type
        )

        # 计算市场冲击
        market_impact = SlippageModel.calculate_market_impact(
            side, quantity, current_bar
        )

        # 确定执行价格
        if side == "BUY":
            execution_price = price + slippage + market_impact
        else:  # SELL
            execution_price = price - slippage - market_impact

        # 确保价格合理（不能超出当天的高低点）
        execution_price = max(
            min(execution_price, current_bar.high),
            current_bar.low
        )

        # 计算滑点百分比
        if side == "BUY":
            slippage_percent = (execution_price - price) / price * 100
        else:
            slippage_percent = (price - execution_price) / price * 100

        # 判断执行质量
        quality = self._evaluate_execution_quality(slippage_percent)

        # 检查部分成交（大订单可能部分成交）
        filled_quantity = self._check_partial_fill(
            quantity, current_bar.volume, order_type
        )

        # 创建成交记录
        if filled_quantity > 0:
            commission = Decimal(str(filled_quantity * 0.01))  # $0.01 每股
            trade = Trade(
                trade_id=f"T_{random.randint(10000, 99999)}",
                symbol=current_bar.symbol,
                side=side,  # 直接使用字符串
                price=execution_price.quantize(Decimal("0.01")),
                quantity=filled_quantity,
                timestamp=current_bar.timestamp,
                commission=commission
            )
        else:
            trade = None

        return ExecutionResult(
            trade=trade,
            filled_quantity=filled_quantity,
            filled_price=execution_price,
            slippage=slippage + market_impact,
            slippage_percent=abs(slippage_percent),
            market_impact=market_impact,
            execution_quality=quality,
            reason="" if filled_quantity == quantity else "部分成交"
        )

    def _check_partial_fill(
        self,
        order_quantity: int,
        volume: int,
        order_type: OrderType
    ) -> int:
        """检查部分成交

        Args:
            order_quantity: 订单数量
            volume: 成交量
            order_type: 订单类型

        Returns:
            实际成交数量
        """
        # 简化逻辑：大部分订单都能成交，只是大单会有部分成交
        if order_quantity <= volume * 0.01:  # 订单小于成交量的 1%
            return order_quantity
        elif order_quantity <= volume * 0.1:  # 订单小于成交量的 10%
            fill_ratio = 0.95 if order_type == OrderType.MARKET else 0.85
        else:  # 大订单
            fill_ratio = 0.3 if order_type == OrderType.MARKET else 0.15

        filled = int(order_quantity * fill_ratio)

        # 确保至少成交 1 股
        filled = max(filled, 1)
        filled = min(filled, order_quantity)

        return filled

    def _evaluate_execution_quality(self, slippage_percent: float) -> ExecutionQuality:
        """评估执行质量

        Args:
            slippage_percent: 滑点百分比

        Returns:
            执行质量等级
        """
        sp = abs(slippage_percent)

        if sp < 0.01:
            return ExecutionQuality.EXCELLENT
        elif sp < 0.05:
            return ExecutionQuality.GOOD
        elif sp < 0.1:
            return ExecutionQuality.FAIR
        elif sp < 0.5:
            return ExecutionQuality.POOR
        else:
            return ExecutionQuality.BAD


class RealisticBroker:
    """更真实的模拟经纪商（带滑点和市场冲击）"""

    def __init__(
        self,
        initial_cash: Decimal = Decimal("100000"),
        commission_per_share: Decimal = Decimal("0.01"),
        enable_slippage: bool = True,
        enable_market_impact: bool = True,
        enable_partial_fill: bool = True
    ):
        """初始化真实经纪商

        Args:
            initial_cash: 初始资金
            commission_per_share: 每股手续费
            enable_slippage: 是否启用滑点
            enable_market_impact: 是否启用市场冲击
            enable_partial_fill: 是否启用部分成交
        """
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.commission_per_share = commission_per_share

        self.enable_slippage = enable_slippage
        self.enable_market_impact = enable_market_impact
        self.enable_partial_fill = enable_partial_fill

        self.positions: Dict[str, int] = {}  # symbol -> quantity
        self.trades: List[Trade] = []
        self.order_book = OrderBookSimulator()

        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'partial_fills': 0,
            'rejected_orders': 0,
            'avg_slippage': 0.0,
            'execution_quality_distribution': {}
        }

    def submit_order(
        self,
        intent: OrderIntent,
        current_bar: Bar,
        order_type: OrderType = OrderType.MARKET
    ) -> ExecutionResult:
        """提交订单（带滑点和市场冲击）

        Args:
            intent: 订单意图
            current_bar: 当前 K 线
            order_type: 订单类型

        Returns:
            执行结果
        """
        self.execution_stats['total_orders'] += 1

        signal = intent.signal
        side = signal.action
        quantity = signal.quantity
        price = signal.price

        # 检查资金
        required_cash = price * quantity * Decimal("1.01")  # 预留 1%
        if side == "BUY" and self.cash < required_cash:
            self.execution_stats['rejected_orders'] += 1
            return ExecutionResult(
                trade=None,
                filled_quantity=0,
                filled_price=Decimal("0"),
                slippage=Decimal("0"),
                slippage_percent=Decimal("0"),
                market_impact=Decimal("0"),
                execution_quality=ExecutionQuality.BAD,
                reason="资金不足"
            )

        # 模拟执行
        if self.enable_slippage or self.enable_market_impact:
            result = self.order_book.simulate_execution(
                side, quantity, price, current_bar, order_type
            )
        else:
            # 无滑点直接成交
            commission = self.commission_per_share * quantity
            trade = Trade(
                trade_id=f"T_{random.randint(10000, 99999)}",
                symbol=current_bar.symbol,
                side=TradeSide(side),
                price=price,
                quantity=quantity,
                timestamp=current_bar.timestamp,
                commission=commission
            )
            result = ExecutionResult(
                trade=trade,
                filled_quantity=quantity,
                filled_price=price,
                slippage=Decimal("0"),
                slippage_percent=Decimal("0"),
                market_impact=Decimal("0"),
                execution_quality=ExecutionQuality.EXCELLENT,
                reason=""
            )

        # 更新统计
        if result.trade:
            self.execution_stats['filled_orders'] += 1
            self.trades.append(result.trade)

            # 更新持仓和资金
            if result.trade.side == "BUY":
                self.positions[result.trade.symbol] = \
                    self.positions.get(result.trade.symbol, 0) + result.trade.quantity
                self.cash -= result.trade.notional_value + result.trade.commission
            else:
                self.positions[result.trade.symbol] -= result.trade.quantity
                if self.positions[result.trade.symbol] == 0:
                    del self.positions[result.trade.symbol]
                self.cash += result.trade.notional_value - result.trade.commission

            # 更新滑点统计
            self._update_slippage_stats(result)

            if result.filled_quantity < quantity:
                self.execution_stats['partial_fills'] += 1

        return result

    def _update_slippage_stats(self, result: ExecutionResult):
        """更新滑点统计"""
        sp = float(result.slippage_percent)
        n = self.execution_stats['filled_orders']

        # 更新平均滑点
        old_avg = self.execution_stats['avg_slippage']
        self.execution_stats['avg_slippage'] = \
            (old_avg * (n - 1) + sp) / n

        # 更新质量分布
        quality = result.execution_quality.value
        self.execution_stats['execution_quality_distribution'][quality] = \
            self.execution_stats['execution_quality_distribution'].get(quality, 0) + 1

    def get_execution_stats(self) -> Dict:
        """获取执行统计"""
        stats = self.execution_stats.copy()

        if stats['total_orders'] > 0:
            stats['fill_rate'] = stats['filled_orders'] / stats['total_orders']
        else:
            stats['fill_rate'] = 0.0

        return stats

    def get_cash_balance(self) -> Decimal:
        """获取现金余额"""
        return self.cash

    def get_total_equity(self) -> Decimal:
        """获取总权益（简化版）"""
        return self.cash

    def get_positions(self) -> Dict[str, int]:
        """获取持仓"""
        return self.positions.copy()


if __name__ == "__main__":
    """测试代码"""
    print("=== 订单执行模拟器测试 ===\n")

    from skills.market_data.mock_source import MockMarketSource
    from datetime import datetime, timedelta

    # 生成测试数据
    source = MockMarketSource(seed=42)
    bars = source.get_bars(
        "AAPL",
        datetime.now() - timedelta(days=100),
        datetime.now(),
        "1d"
    )

    print(f"生成 {len(bars)} 根 K 线\n")

    # 创建真实经纪商
    broker = RealisticBroker(
        initial_cash=Decimal("100000"),
        enable_slippage=True,
        enable_market_impact=True,
        enable_partial_fill=True
    )

    # 测试订单执行
    test_bar = bars[-1]
    print(f"测试执行 - 当前价格: ${test_bar.close}")

    from core.models import Signal

    signal = Signal(
        symbol="AAPL",
        timestamp=test_bar.timestamp,
        action="BUY",
        price=test_bar.close,
        quantity=100,  # 小订单
        confidence=0.8,
        reason="测试"
    )

    from core.models import OrderIntent
    intent = OrderIntent(
        signal=signal,
        timestamp=datetime.now(),
        approved=True,
        risk_reason="测试"
    )

    result = broker.submit_order(intent, test_bar, OrderType.MARKET)

    print(f"\n执行结果:")
    print(f"  成交数量: {result.filled_quantity}")
    print(f"  成交价格: ${result.filled_price}")
    print(f"  滑点: {result.slippage_percent:.4f}%")
    print(f"  市场冲击: {result.market_impact}")
    print(f"  执行质量: {result.execution_quality.value}")
    if result.trade:
        print(f"  手续费: ${result.trade.commission}")

    # 显示统计
    print(f"\n执行统计:")
    stats = broker.get_execution_stats()
    print(f"  总订单: {stats['total_orders']}")
    print(f"  成交订单: {stats['filled_orders']}")
    print(f"  成交率: {stats['fill_rate']:.2%}")
    print(f"  平均滑点: {stats['avg_slippage']:.4f}%")

    print("\n✓ 测试完成")
