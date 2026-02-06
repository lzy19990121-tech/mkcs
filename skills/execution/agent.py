"""
ExecutionAgent - 执行层

处理 paper / replay / live 三种执行模式
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Literal, List
from decimal import Decimal

from core.schema import (
    RiskDecision,
    ExecutionResult,
    MarketState,
    SchemaVersion
)

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    ExecutionAgent - 执行层

    职责：
    ✓ 执行交易指令（paper / replay / live）
    ✓ 记录执行结果
    ✓ 记录滑点和成交价
    ✓ dry-run 模式（默认）

    不做：
    ✗ 不做任何交易决策
    """

    def __init__(
        self,
        mode: Literal["paper", "replay", "live", "dry_run"] = "dry_run",
        initial_capital: float = 100000
    ):
        """
        初始化 ExecutionAgent

        Args:
            mode: 执行模式
            initial_capital: 初始资金
        """
        self.mode = mode
        self.initial_capital = initial_capital

        # 当前持仓
        self._positions: Dict[str, float] = {}  # symbol -> quantity
        self._cash_balance = initial_capital

        # 历史记录
        self._execution_history: List[ExecutionResult] = []
        self._trade_history: List[Dict[str, Any]] = []

        # 配置
        self._is_live_unlocked = False  # live 模式需要显式解锁
        self._slippage_rate = 0.001  # 默认 0.1% 滑点

    def execute(
        self,
        risk_decision: RiskDecision,
        current_price: float,
        market_state: Optional[MarketState] = None
    ) -> ExecutionResult:
        """
        执行交易

        Args:
            risk_decision: 风控决策
            current_price: 当前价格
            market_state: 市场状态（可选，用于记录）

        Returns:
            ExecutionResult
        """
        symbol = risk_decision.symbol
        target_position = risk_decision.scaled_target_position

        # 检查 live 模式是否解锁
        if self.mode == "live" and not self._is_live_unlocked:
            return ExecutionResult(
                timestamp=risk_decision.timestamp,
                symbol=symbol,
                execution_mode=self.mode,
                target_position=target_position,
                actual_position=0,
                target_price=current_price,
                status="FAILED",
                error_message="Live 模式未解锁，显式调用 unlock_live() 以使用真实交易"
            )

        # 检查是否允许交易
        if risk_decision.risk_action in ["PAUSE", "DISABLE"]:
            return ExecutionResult(
                timestamp=risk_decision.timestamp,
                symbol=symbol,
                execution_mode=self.mode,
                target_position=target_position,
                actual_position=self._positions.get(symbol, 0),
                target_price=current_price,
                status="SKIPPED",
                error_message=f"风控拒绝: {risk_decision.risk_action}"
            )

        # 计算交易数量
        current_position = self._positions.get(symbol, 0)
        trade_quantity = target_position - current_position

        if abs(trade_quantity) < 0.01:  # 数量太小，跳过
            return ExecutionResult(
                timestamp=risk_decision.timestamp,
                symbol=symbol,
                execution_mode=self.mode,
                target_position=target_position,
                actual_position=current_position,
                target_price=current_price,
                status="SKIPPED",
                fill_quantity=0
            )

        # 计算成交价（模拟滑点）
        fill_price = self._calculate_fill_price(current_price, trade_quantity)

        # 检查资金（paper/replay 模式）
        if self.mode in ["paper", "replay"]:
            required_cash = abs(trade_quantity) * fill_price
            if trade_quantity > 0:  # 买入
                if required_cash > self._cash_balance:
                    return ExecutionResult(
                        timestamp=risk_decision.timestamp,
                        symbol=symbol,
                        execution_mode=self.mode,
                        target_position=target_position,
                        actual_position=current_position,
                        target_price=current_price,
                        status="FAILED",
                        error_message=f"资金不足 (需要 {required_cash:.0f}, 可用 {self._cash_balance:.0f})"
                    )

        # 执行交易
        self._positions[symbol] = target_position

        # 更新现金余额
        if self.mode in ["paper", "replay"]:
            self._cash_balance -= trade_quantity * fill_price

        # 计算滑点
        slippage = abs(fill_price - current_price) / current_price if current_price > 0 else 0

        # 构建结果
        result = ExecutionResult(
            timestamp=risk_decision.timestamp,
            symbol=symbol,
            execution_mode=self.mode,
            target_position=target_position,
            actual_position=target_position,
            target_price=current_price,
            fill_price=fill_price,
            status="FILLED",
            fill_quantity=abs(trade_quantity),
            slippage=slippage,
            current_positions=dict(self._positions),
            cash_balance=self._cash_balance
        )

        self._execution_history.append(result)

        # 记录交易历史
        self._trade_history.append({
            "timestamp": risk_decision.timestamp.isoformat(),
            "symbol": symbol,
            "action": "BUY" if trade_quantity > 0 else "SELL",
            "quantity": abs(trade_quantity),
            "price": fill_price,
            "mode": self.mode
        })

        logger.info(
            f"执行[{self.mode}]: {symbol} {trade_quantity:+.0f} @ {fill_price:.2f} "
            f"(目标 {target_position:.0f}, 滑点 {slippage:.2%})"
        )

        return result

    def _calculate_fill_price(
        self,
        current_price: float,
        trade_quantity: float
    ) -> float:
        """计算成交价（模拟滑点）"""
        # 简单的滑点模型
        if trade_quantity > 0:  # 买入，价格向上滑
            slippage = self._slippage_rate
        else:  # 卖出，价格向下滑
            slippage = -self._slippage_rate

        return current_price * (1 + slippage)

    def unlock_live(self):
        """解锁 live 模式

        警告：这将允许真实交易！
        """
        import warnings

        warnings.warn(
            "⚠️ 警告：正在解锁 LIVE 模式！真实交易将发生！\n"
            "如果您确定，请调用 confirm_unlock_live() 以确认。",
            stacklevel=2
        )
        self._live_unlock_requested = True

    def confirm_unlock_live(self):
        """确认解锁 live 模式"""
        if not hasattr(self, "_live_unlock_requested") or not self._live_unlock_requested:
            raise RuntimeError("必须先调用 unlock_live() 以请求解锁")

        self._is_live_unlocked = True
        self._live_unlock_requested = False
        logger.warning("🔓 Live 模式已解锁 - 真实交易将发生")

    def get_positions(self) -> Dict[str, float]:
        """获取当前持仓"""
        return dict(self._positions)

    def get_cash_balance(self) -> float:
        """获取现金余额"""
        return self._cash_balance

    def get_equity(self) -> float:
        """获取总权益（简化计算）"""
        # 简化：假设所有持仓的市值为仓位 * 100
        positions_value = sum(abs(q) * 100 for q in self._positions.values())
        return self._cash_balance + positions_value

    def get_execution_history(self, limit: int = 100) -> List[ExecutionResult]:
        """获取执行历史"""
        return self._execution_history[-limit:]

    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取交易历史"""
        return self._trade_history[-limit:]

    def reset(self, preserve_config: bool = False):
        """重置状态

        Args:
            preserve_config: 是否保留配置（live 解锁状态等）
        """
        live_unlocked = self._is_live_unlocked if preserve_config else False

        self._positions = {}
        self._cash_balance = self.initial_capital
        self._execution_history = []
        self._trade_history = []

        if preserve_config:
            self._is_live_unlocked = live_unlocked

    def get_portfolio_value(self) -> Dict[str, Any]:
        """获取组合价值"""
        return {
            "cash": self._cash_balance,
            "positions": dict(self._positions),
            "total_equity": self.get_equity(),
            "mode": self.mode
        }


if __name__ == "__main__":
    """测试代码"""
    print("=== ExecutionAgent 测试 ===\n")

    from core.schema import RiskDecision, MarketState

    # 创建 ExecutionAgent
    executor = ExecutionAgent(mode="paper", initial_capital=100000)

    # 测试数据
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

    # 测试1: 正常执行
    print("1. 正常执行:")
    risk_decision = RiskDecision(
        timestamp=datetime.now(),
        symbol="AAPL",
        scaled_target_position=100,
        scale_factor=1.0,
        risk_action="APPROVE",
        risk_reason="风控通过"
    )

    result = executor.execute(risk_decision, current_price=150)
    print(f"   Status: {result.status}")
    print(f"   Fill price: {result.fill_price:.2f}")
    print(f"   Cash balance: {result.cash_balance:.0f}")
    print(f"   Positions: {result.current_positions}")

    # 测试2: 风控拒绝
    print("\n2. 风控拒绝:")
    risk_decision = RiskDecision(
        timestamp=datetime.now(),
        symbol="AAPL",
        scaled_target_position=200,
        scale_factor=0.5,
        risk_action="SCALE_DOWN",
        risk_reason="高波动降低仓位"
    )

    result = executor.execute(risk_decision, current_price=150)
    print(f"   Target position: {result.target_position:.0f}")
    print(f"   Actual position: {result.actual_position:.0f}")
    print(f"   Status: {result.status}")

    # 测试3: live 模式未解锁
    print("\n3. Live 模式未解锁:")
    live_executor = ExecutionAgent(mode="live", initial_capital=100000)
    result = live_executor.execute(risk_decision, current_price=150)
    print(f"   Status: {result.status}")
    print(f"   Error: {result.error_message}")

    # 测试4: 解锁 live 模式
    print("\n4. Live 模式解锁:")
    try:
        live_executor.confirm_unlock_live()
        result = live_executor.execute(risk_decision, current_price=150)
        print(f"   Status: {result.status}")
        print(f"   Mode: {result.execution_mode}")
    except RuntimeError as e:
        print(f"   Error: {e}")

    print("\n✓ 所有测试通过")
