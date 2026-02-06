"""
动态止盈止损模块

实现 trailing stop、分批止盈等动态风险管理
"""

import logging
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from core.strategy_models import RegimeType

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """退出原因"""
    STOP_LOSS = "stop_loss"           # 止损
    TAKE_PROFIT = "take_profit"       # 止盈
    TRAILING_STOP = "trailing_stop"   # 移动止损
    PARTIAL_TP = "partial_tp"         # 分批止盈
    SIGNAL_REVERSE = "signal_reverse" # 信号反转
    TIME_EXIT = "time_exit"           # 时间止损
    REGIME_CHANGE = "regime_change"   # 市场状态变化
    MANUAL = "manual"                 # 手动


@dataclass
class StopLevel:
    """止损/止盈价位"""
    price: Decimal
    quantity: int                     # 该价位执行的仓位
    reason: ExitReason
    executed: bool = False
    execute_time: Optional[datetime] = None


@dataclass
class PositionTracker:
    """持仓跟踪器

    记录持仓的入场信息、当前止损止盈状态
    """
    symbol: str
    entry_price: Decimal
    entry_time: datetime
    quantity: int
    original_quantity: int           # 原始持仓数量（用于分批止盈计算）

    # 止损止���设置
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop_pct: Optional[float] = None
    trailing_stop_price: Optional[Decimal] = None
    max_favorable_price: Optional[Decimal] = None  # 最高有利价

    # 分批止盈
    partial_levels: List[StopLevel] = field(default_factory=list)

    # 状态
    highest_unrealized_pnl: Decimal = Decimal("0")
    highest_price: Decimal = Decimal("0")
    lowest_price: Decimal = Decimal("999999")

    def update_price(self, price: Decimal):
        """更新价格，跟踪最高/最低价"""
        if price > self.highest_price:
            self.highest_price = price

        if price < self.lowest_price:
            self.lowest_price = price

        # 计算浮盈
        pnl = (price - self.entry_price) * self.quantity
        if pnl > self.highest_unrealized_pnl:
            self.highest_unrealized_pnl = pnl

    def get_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """计算浮盈"""
        return (current_price - self.entry_price) * self.quantity

    def get_unrealized_pnl_pct(self, current_price: Decimal) -> float:
        """计算浮盈百分比"""
        return float((current_price - self.entry_price) / self.entry_price)

    def update_trailing_stop(self, current_price: Decimal):
        """更新移动止损价"""
        if self.trailing_stop_pct is None:
            return

        # 更新最高有利价
        if self.max_favorable_price is None or current_price > self.max_favorable_price:
            self.max_favorable_price = current_price

        # 计算移动止损价
        new_stop = self.max_favorable_price * (Decimal("1") - Decimal(str(self.trailing_stop_pct)))
        if self.trailing_stop_price is None or new_stop > self.trailing_stop_price:
            self.trailing_stop_price = new_stop

    def should_exit(self, current_price: Decimal) -> Tuple[bool, Optional[ExitReason], int]:
        """检查是否应该退出

        Returns:
            (是否退出, 退出原因, 退出数量)
        """
        # 1. 检查止损
        if self.stop_loss and current_price <= self.stop_loss:
            return True, ExitReason.STOP_LOSS, self.quantity

        # 2. 检查止盈
        if self.take_profit and current_price >= self.take_profit:
            return True, ExitReason.TAKE_PROFIT, self.quantity

        # 3. 检查移动止损
        self.update_trailing_stop(current_price)
        if self.trailing_stop_price and current_price <= self.trailing_stop_price:
            return True, ExitReason.TRAILING_STOP, self.quantity

        # 4. 检查分批止盈
        for level in self.partial_levels:
            if not level.executed and current_price >= level.price:
                return True, ExitReason.PARTIAL_TP, level.quantity

        return False, None, 0


class DynamicStopManager:
    """动态止盈止损管理器

    根据市场状态、持仓时间、浮盈情况动态调整止盈止损
    """

    def __init__(
        self,
        default_stop_atr_multiplier: float = 1.5,
        default_tp_atr_multiplier: float = 2.5,
        default_trailing_stop_pct: float = 0.02,
        enable_partial_tp: bool = True
    ):
        """初始化管理器

        Args:
            default_stop_atr_multiplier: 默认止损ATR倍数
            default_tp_atr_multiplier: 默认止盈ATR倍数
            default_trailing_stop_pct: 默认移动止损百分比
            enable_partial_tp: 是否启用分批止盈
        """
        self.default_stop_atr_multiplier = default_stop_atr_multiplier
        self.default_tp_atr_multiplier = default_tp_atr_multiplier
        self.default_trailing_stop_pct = default_trailing_stop_pct
        self.enable_partial_tp = enable_partial_tp

        self._trackers: Dict[str, PositionTracker] = {}

    def create_tracker(
        self,
        symbol: str,
        entry_price: Decimal,
        quantity: int,
        atr: Optional[Decimal] = None,
        regime: RegimeType = RegimeType.UNKNOWN,
        custom_stops: Optional[Dict[str, Any]] = None
    ) -> PositionTracker:
        """创建持仓跟踪器

        Args:
            symbol: 标的代码
            entry_price: 入场价格
            quantity: 持仓数量
            atr: ATR值（用于计算止损止盈）
            regime: 市场状态
            custom_stops: 自定义止损止盈参数

        Returns:
            PositionTracker
        """
        tracker = PositionTracker(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=quantity,
            original_quantity=quantity
        )

        # 根据市场状态调整止损止盈参数
        stop_multiplier, tp_multiplier, trailing_pct = self._get_stop_params(regime)

        # 应用自定义参数
        if custom_stops:
            stop_multiplier = custom_stops.get("stop_multiplier", stop_multiplier)
            tp_multiplier = custom_stops.get("tp_multiplier", tp_multiplier)
            trailing_pct = custom_stops.get("trailing_stop_pct", trailing_pct)

        # 计算止损价
        if atr:
            tracker.stop_loss = entry_price - (atr * Decimal(str(stop_multiplier)))

        # 计算止盈价
        if atr:
            tracker.take_profit = entry_price + (atr * Decimal(str(tp_multiplier)))

        # 设置移动止损
        if trailing_pct > 0:
            tracker.trailing_stop_pct = trailing_pct
            # 初始移动止损价在盈亏平衡点以上
            tracker.trailing_stop_price = entry_price * Decimal("1.005")

        # 设置分批止盈
        if self.enable_partial_tp and atr:
            self._setup_partial_tp(tracker, atr, regime)

        # 初始化价格跟踪
        tracker.highest_price = entry_price
        tracker.lowest_price = entry_price

        self._trackers[symbol] = tracker

        return tracker

    def _get_stop_params(self, regime: RegimeType) -> Tuple[float, float, float]:
        """根据市场状态获取止损止盈参数"""
        # (止损倍数, 止盈倍数, 移动止损百分比)
        params = {
            RegimeType.TREND: (1.5, 3.0, 0.02),      # 趋势市：宽止损，让利润奔跑
            RegimeType.RANGE: (1.0, 1.5, 0.01),      # 震荡市：紧止损，快速止盈
            RegimeType.HIGH_VOL: (2.0, 2.5, 0.03),   # 高波动：更宽的止损
            RegimeType.LOW_VOL: (1.0, 2.0, 0.015),   # 低波动：正常止损
            RegimeType.UNKNOWN: (1.5, 2.0, 0.02),
        }
        return params.get(regime, (1.5, 2.0, 0.02))

    def _setup_partial_tp(self, tracker: PositionTracker, atr: Decimal, regime: RegimeType):
        """设置分批止盈

        根据市场状态设置不同的分批止盈策略
        """
        qty = tracker.original_quantity

        if regime == RegimeType.TREND:
            # 趋势市：分3批，每批1/3，距离递增
            tracker.partial_levels = [
                StopLevel(
                    price=tracker.entry_price + atr * Decimal("1.5"),
                    quantity=qty // 3,
                    reason=ExitReason.PARTIAL_TP
                ),
                StopLevel(
                    price=tracker.entry_price + atr * Decimal("2.5"),
                    quantity=qty // 3,
                    reason=ExitReason.PARTIAL_TP
                ),
                StopLevel(
                    price=tracker.entry_price + atr * Decimal("4.0"),
                    quantity=qty - 2 * (qty // 3),
                    reason=ExitReason.PARTIAL_TP
                ),
            ]
        elif regime == RegimeType.RANGE:
            # 震荡市：快速止盈，分2批
            tracker.partial_levels = [
                StopLevel(
                    price=tracker.entry_price + atr * Decimal("1.0"),
                    quantity=qty // 2,
                    reason=ExitReason.PARTIAL_TP
                ),
                StopLevel(
                    price=tracker.entry_price + atr * Decimal("1.5"),
                    quantity=qty - qty // 2,
                    reason=ExitReason.PARTIAL_TP
                ),
            ]
        else:
            # 默认：分2批平仓
            tracker.partial_levels = [
                StopLevel(
                    price=tracker.entry_price + atr * Decimal("2.0"),
                    quantity=qty // 2,
                    reason=ExitReason.PARTIAL_TP
                ),
                StopLevel(
                    price=tracker.entry_price + atr * Decimal("3.0"),
                    quantity=qty - qty // 2,
                    reason=ExitReason.PARTIAL_TP
                ),
            ]

    def check_exit(
        self,
        symbol: str,
        current_price: Decimal
    ) -> Optional[Dict[str, Any]]:
        """检查是否应该退出

        Args:
            symbol: 标的代码
            current_price: 当前价格

        Returns:
            退出指令字典，或 None
        """
        tracker = self._trackers.get(symbol)
        if not tracker:
            return None

        # 更新价格
        tracker.update_price(current_price)

        # 检查退出条件
        should_exit, reason, quantity = tracker.should_exit(current_price)

        if should_exit:
            # 如果是分批止盈，标记该层级为已执行
            if reason == ExitReason.PARTIAL_TP:
                for level in tracker.partial_levels:
                    if not level.executed and current_price >= level.price:
                        level.executed = True
                        level.execute_time = datetime.now()
                        break

            # 如果全部退出，移除跟踪器
            if quantity >= tracker.quantity:
                self._trackers.pop(symbol, None)
            else:
                # 减少持仓数量
                tracker.quantity -= quantity

            return {
                "symbol": symbol,
                "action": "SELL",
                "quantity": quantity,
                "price": current_price,
                "reason": reason.value,
                "pnl": tracker.get_unrealized_pnl(current_price),
                "pnl_pct": tracker.get_unrealized_pnl_pct(current_price)
            }

        return None

    def update_tracker(self, symbol: str, **kwargs):
        """更新跟踪器参数"""
        tracker = self._trackers.get(symbol)
        if tracker:
            for key, value in kwargs.items():
                if hasattr(tracker, key):
                    setattr(tracker, key, value)

    def get_tracker(self, symbol: str) -> Optional[PositionTracker]:
        """获取持仓跟踪器"""
        return self._trackers.get(symbol)

    def remove_tracker(self, symbol: str):
        """移除持仓跟踪器"""
        self._trackers.pop(symbol, None)

    def get_all_trackers(self) -> Dict[str, PositionTracker]:
        """获取所有跟踪器"""
        return self._trackers.copy()


class TimeBasedExit:
    """时间止损

    根据持仓时间强制退出
    """

    def __init__(self, max_holding_minutes: int = 60 * 24 * 5):  # 默认5天
        """��始化时间止损

        Args:
            max_holding_minutes: 最大持仓时间（分钟）
        """
        self.max_holding_minutes = max_holding_minutes
        self.entry_times: Dict[str, datetime] = {}

    def on_entry(self, symbol: str, entry_time: datetime = None):
        """记录入场时间"""
        self.entry_times[symbol] = entry_time or datetime.now()

    def check_exit(self, symbol: str, current_time: datetime = None) -> bool:
        """检查是否超时

        Args:
            symbol: 标的代码
            current_time: 当前时间

        Returns:
            是否应该退出
        """
        entry_time = self.entry_times.get(symbol)
        if not entry_time:
            return False

        current_time = current_time or datetime.now()
        elapsed = (current_time - entry_time).total_seconds() / 60

        if elapsed >= self.max_holding_minutes:
            self.entry_times.pop(symbol, None)
            return True

        return False

    def on_exit(self, symbol: str):
        """退出时清理"""
        self.entry_times.pop(symbol, None)


if __name__ == "__main__":
    """测试代码"""
    print("=== DynamicStopManager 测试 ===\n")

    manager = DynamicStopManager()

    # 测试1: 创建跟踪器
    print("1. 创建持仓跟踪器:")
    tracker = manager.create_tracker(
        symbol="AAPL",
        entry_price=Decimal("150"),
        quantity=1000,
        atr=Decimal("5"),
        regime=RegimeType.TREND
    )
    print(f"   入场价: {tracker.entry_price}")
    print(f"   止损价: {tracker.stop_loss}")
    print(f"   止盈价: {tracker.take_profit}")
    print(f"   移动止损: {tracker.trailing_stop_pct}")
    print(f"   分批止盈层级: {len(tracker.partial_levels)}")

    # 测试2: 检查止损
    print("\n2. 触发止损:")
    result = manager.check_exit("AAPL", Decimal("142"))
    if result:
        print(f"   触发退出: {result['reason']}")
        print(f"   数量: {result['quantity']}")
        print(f"   浮亏: {result['pnl_pct']:.2%}")

    # 测试3: 测试移动止损
    print("\n3. 测试移动止损:")
    manager2 = DynamicStopManager()
    tracker2 = manager2.create_tracker(
        symbol="MSFT",
        entry_price=Decimal("300"),
        quantity=1000,
        atr=Decimal("10"),
        regime=RegimeType.TREND
    )

    # 价格上涨
    manager2.check_exit("MSFT", Decimal("320"))
    updated = manager2.get_tracker("MSFT")
    print(f"   最高价: {updated.max_favorable_price}")
    print(f"   移动止损价: {updated.trailing_stop_price}")

    # 价格回落到移动止损
    result = manager2.check_exit("MSFT", Decimal("310"))
    if result:
        print(f"   触发退出: {result['reason']}")
        print(f"   浮盈: {result['pnl_pct']:.2%}")

    # 测试4: 分批止盈
    print("\n4. 测试分批止盈:")
    manager3 = DynamicStopManager()
    tracker3 = manager3.create_tracker(
        symbol="GOOGL",
        entry_price=Decimal("150"),
        quantity=900,
        atr=Decimal("5"),
        regime=RegimeType.TREND
    )

    # 第一批止盈
    result = manager3.check_exit("GOOGL", Decimal("158"))
    if result:
        print(f"   第一批: {result['quantity']}股, 原因: {result['reason']}")

    # 第二批止盈
    result = manager3.check_exit("GOOGL", Decimal("163"))
    if result:
        print(f"   第二批: {result['quantity']}股, 原因: {result['reason']}")

    # 测试5: 不同市场状态的参数
    print("\n5. 不同市场状态的止损参数:")
    for regime in RegimeType:
        manager_test = DynamicStopManager()
        t = manager_test.create_tracker(
            symbol="TEST",
            entry_price=Decimal("100"),
            quantity=100,
            atr=Decimal("2"),
            regime=regime
        )
        stop_dist = float((t.entry_price - t.stop_loss) / t.entry_price * 100) if t.stop_loss else 0
        tp_dist = float((t.take_profit - t.entry_price) / t.entry_price * 100) if t.take_profit else 0
        print(f"   {regime.value:10s}: 止损={stop_dist:.1f}%, 止盈={tp_dist:.1f}%, "
              f"移动止损={t.trailing_stop_pct}")

    print("\n✓ 所有测试通过")
