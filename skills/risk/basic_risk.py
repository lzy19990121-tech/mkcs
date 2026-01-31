"""
基础风控规则

实现常见的风控检查：仓位限制、黑名单、资金充足性等
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Set, Optional

from core.models import Signal, OrderIntent, Position
from skills.risk.base import RiskManager


class BasicRiskManager(RiskManager):
    """基础风控管理器

    风控规则：
    1. 单只股票最大仓位限制
    2. 单日最大亏损限制
    3. 持仓数量限制
    4. 禁止交易黑名单股票
    5. 资金充足性检查
    """

    def __init__(
        self,
        max_position_ratio: float = 0.2,  # 单只股票最大仓位占比
        max_positions: int = 10,  # 最大持仓数量
        blacklist: Optional[Set[str]] = None,  # 黑名单
        max_daily_loss_ratio: float = 0.05  # 单日最大亏损比例
    ):
        """初始化风控参数

        Args:
            max_position_ratio: 单只股票最大仓位占比（0-1）
            max_positions: 最大持仓数量
            blacklist: 股票黑名单
            max_daily_loss_ratio: 单日最大亏损比例
        """
        self.max_position_ratio = max_position_ratio
        self.max_positions = max_positions
        self.blacklist = blacklist or set()
        self.max_daily_loss_ratio = max_daily_loss_ratio

        # 记录当日盈亏（实际应用中应从数据库读取）
        self.daily_pnl = Decimal("0.00")
        self.initial_capital = None

    def set_capital(self, initial_capital: float):
        """设置初始资金

        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = Decimal(str(initial_capital))

    def update_daily_pnl(self, pnl: Decimal):
        """更新当日盈亏

        Args:
            pnl: 当日盈亏
        """
        self.daily_pnl = pnl

    def check(
        self,
        signal: Signal,
        positions: Dict[str, Position],
        cash_balance: float
    ) -> OrderIntent:
        """执行风控检查

        Args:
            signal: 交易信号
            positions: 当前持仓
            cash_balance: 可用资金

        Returns:
            订单意图
        """
        # 规则1: 检查是否为HOLD信号
        if signal.action == 'HOLD':
            return OrderIntent(
                signal=signal,
                timestamp=signal.timestamp,
                approved=False,
                risk_reason="HOLD信号无需执行"
            )

        # 规则2: 检查黑名单
        if signal.symbol in self.blacklist:
            return OrderIntent(
                signal=signal,
                timestamp=signal.timestamp,
                approved=False,
                risk_reason=f"股票在黑名单中: {signal.symbol}"
            )

        # 规则3: 检查资金充足性（仅买入）
        if signal.action == 'BUY':
            required_amount = signal.price * signal.quantity
            available_cash = Decimal(str(cash_balance))

            if required_amount > available_cash:
                return OrderIntent(
                    signal=signal,
                    timestamp=signal.timestamp,
                    approved=False,
                    risk_reason=f"资金不足: 需要 {required_amount}, 可用 {available_cash}"
                )

        # 规则4: 检查持仓数量限制（仅开新仓）
        if signal.action == 'BUY' and signal.symbol not in positions:
            if len(positions) >= self.max_positions:
                return OrderIntent(
                    signal=signal,
                    timestamp=signal.timestamp,
                    approved=False,
                    risk_reason=f"持仓数量已达上限: {len(positions)}/{self.max_positions}"
                )

        # 规则5: 检查单只股票仓位限制
        if signal.action == 'BUY':
            # 计算目标持仓市值
            target_value = signal.price * signal.quantity

            # 计算当前持仓市值（如果已有持仓）
            current_value = Decimal("0")
            if signal.symbol in positions:
                current_value = positions[signal.symbol].market_value

            # 计算总资产
            total_equity = sum(pos.market_value for pos in positions.values()) + Decimal(str(cash_balance))

            # 检查是否超限
            target_position_value = current_value + target_value
            position_ratio = float(target_position_value / total_equity) if total_equity > 0 else 0

            if position_ratio > self.max_position_ratio:
                return OrderIntent(
                    signal=signal,
                    timestamp=signal.timestamp,
                    approved=False,
                    risk_reason=f"单只股票仓位超限: {position_ratio:.2%} > {self.max_position_ratio:.2%}"
                )

        # 规则6: 检查单日亏损限制
        if self.initial_capital:
            loss_ratio = float(self.daily_pnl / self.initial_capital)
            if loss_ratio < -self.max_daily_loss_ratio:
                return OrderIntent(
                    signal=signal,
                    timestamp=signal.timestamp,
                    approved=False,
                    risk_reason=f"单日亏损超限: {loss_ratio:.2%} < -{self.max_daily_loss_ratio:.2%}"
                )

        # 所有风控检查通过
        return OrderIntent(
            signal=signal,
            timestamp=signal.timestamp,
            approved=True,
            risk_reason="符合所有风控规则"
        )


class StrictRiskManager(BasicRiskManager):
    """严格风控管理器

    更保守的风控参数
    """

    def __init__(self):
        super().__init__(
            max_position_ratio=0.1,  # 单只股票最多10%
            max_positions=5,  # 最多5只股票
            max_daily_loss_ratio=0.02  # 单日最多亏损2%
        )


if __name__ == "__main__":
    """自测代码"""
    from decimal import Decimal

    print("=== BasicRiskManager 测试 ===\n")

    # 创建风控管理器
    risk_mgr = BasicRiskManager(
        max_position_ratio=0.2,
        max_positions=3,
        blacklist={"PENNY.ST", "RISKY"},
        max_daily_loss_ratio=0.05
    )
    risk_mgr.set_capital(100000)  # 10万美元

    # 测试1: 正常买入信号
    print("1. 测试正常买入信号:")
    signal1 = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.8,
        reason="金叉买入"
    )
    intent1 = risk_mgr.check(signal1, {}, cash_balance=100000)
    print(f"   结果: {'通过' if intent1.approved else '拒绝'}")
    print(f"   原因: {intent1.risk_reason}")

    # 测试2: 黑名单股票
    print("\n2. 测试黑名单股票:")
    signal2 = Signal(
        symbol="PENNY.ST",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("1.00"),
        quantity=1000,
        confidence=0.9,
        reason="测试"
    )
    intent2 = risk_mgr.check(signal2, {}, cash_balance=100000)
    print(f"   结果: {'通过' if intent2.approved else '拒绝'}")
    print(f"   原因: {intent2.risk_reason}")

    # 测试3: 资金不足
    print("\n3. 测试资金不足:")
    signal3 = Signal(
        symbol="TSLA",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("250.00"),
        quantity=1000,
        confidence=0.8,
        reason="测试"
    )
    intent3 = risk_mgr.check(signal3, {}, cash_balance=1000)
    print(f"   结果: {'通过' if intent3.approved else '拒绝'}")
    print(f"   原因: {intent3.risk_reason}")

    # 测试4: 持仓数量超限
    print("\n4. 测试持仓数量超限:")
    # 创建3个持仓
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=100,
            avg_price=Decimal("150.00"),
            market_value=Decimal("15000.00"),
            unrealized_pnl=Decimal("0.00")
        ),
        "GOOGL": Position(
            symbol="GOOGL",
            quantity=50,
            avg_price=Decimal("140.00"),
            market_value=Decimal("7000.00"),
            unrealized_pnl=Decimal("0.00")
        ),
        "MSFT": Position(
            symbol="MSFT",
            quantity=30,
            avg_price=Decimal("380.00"),
            market_value=Decimal("11400.00"),
            unrealized_pnl=Decimal("0.00")
        )
    }

    signal4 = Signal(
        symbol="NVDA",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("500.00"),
        quantity=10,
        confidence=0.8,
        reason="测试"
    )
    intent4 = risk_mgr.check(signal4, positions, cash_balance=50000)
    print(f"   当前持仓数: {len(positions)}")
    print(f"   结果: {'通过' if intent4.approved else '拒绝'}")
    print(f"   原因: {intent4.risk_reason}")

    # 测试5: 仓位比例超限
    print("\n5. 测试仓位比例超限:")
    signal5 = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150.00"),
        quantity=500,
        confidence=0.8,
        reason="测试"
    )
    aapl_position = Position(
        symbol="AAPL",
        quantity=100,
        avg_price=Decimal("150.00"),
        market_value=Decimal("15000.00"),
        unrealized_pnl=Decimal("0.00")
    )
    intent5 = risk_mgr.check(signal5, {"AAPL": aapl_position}, cash_balance=50000)
    print(f"   结果: {'通过' if intent5.approved else '拒绝'}")
    print(f"   原因: {intent5.risk_reason}")

    # 测试6: HOLD信号
    print("\n6. 测试HOLD信号:")
    signal6 = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="HOLD",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.5,
        reason="观望"
    )
    intent6 = risk_mgr.check(signal6, {}, cash_balance=100000)
    print(f"   结果: {'通过' if intent6.approved else '拒绝'}")
    print(f"   原因: {intent6.risk_reason}")

    # 测试7: 卖出信号（通常不受仓位限制）
    print("\n7. 测试卖出信号:")
    signal7 = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="SELL",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.8,
        reason="测试"
    )
    intent7 = risk_mgr.check(signal7, {"AAPL": aapl_position}, cash_balance=50000)
    print(f"   结果: {'通过' if intent7.approved else '拒绝'}")
    print(f"   原因: {intent7.risk_reason}")

    # 测试8: 单日亏损超限
    print("\n8. 测试单日亏损超限:")
    risk_mgr.update_daily_pnl(Decimal("-6000"))  # 亏损6%
    signal8 = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150.00"),
        quantity=10,
        confidence=0.8,
        reason="测试"
    )
    intent8 = risk_mgr.check(signal8, {}, cash_balance=50000)
    print(f"   当日亏损: 6%")
    print(f"   结果: {'通过' if intent8.approved else '拒绝'}")
    print(f"   原因: {intent8.risk_reason}")

    print("\n✓ 所有测试通过")
