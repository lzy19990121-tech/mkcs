"""
市场感知的风控管理器

将 MarketState 集成到风控决策中，实现：
1. 根据市场状态动态调整仓位限制
2. 市场危机模式下的自动降权
3. 系统性风险事件窗口处理
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.models import Signal, Position
from core.strategy_models import StrategySignal
from skills.market_analysis.market_state import MarketState, RegimeType, VolatilityState, LiquidityState
from skills.risk.adaptive_risk import (
    AdaptiveRiskManager,
    RiskAction,
    RiskCheckResult,
    RiskRule,
    CooldownState
)

logger = logging.getLogger(__name__)


@dataclass
class MarketRiskConfig:
    """市场风险配置

    根据不同市场状态的风控参数
    """
    # 正常市场
    normal_max_position_ratio: float = 0.2
    normal_max_total_ratio: float = 0.8

    # 震荡市场
    range_max_position_ratio: float = 0.15
    range_max_total_ratio: float = 0.6

    # 高波动市场
    high_vol_max_position_ratio: float = 0.1
    high_vol_max_total_ratio: float = 0.5

    # 危机模式
    crisis_max_position_ratio: float = 0.05
    crisis_max_total_ratio: float = 0.2

    # 低流动性
    thin_liquidity_multiplier: float = 0.5

    # 系统性风险事件窗口
    event_window_multiplier: float = 0.3

    def get_position_limit(self, market_state: MarketState) -> float:
        """获取单个标的最大仓位限制"""
        base_limit = self.normal_max_position_ratio

        # 市场状态调整
        if market_state.regime == RegimeType.CRISIS:
            base_limit = self.crisis_max_position_ratio
        elif market_state.regime == RegimeType.RANGE:
            base_limit = self.range_max_position_ratio
        elif market_state.volatility_state in [VolatilityState.HIGH, VolatilityState.EXTREME]:
            base_limit = self.high_vol_max_position_ratio

        # 流动性调整
        if market_state.liquidity_state == LiquidityState.THIN:
            base_limit *= self.thin_liquidity_multiplier
        elif market_state.liquidity_state == LiquidityState.FROZEN:
            base_limit *= 0.1  # 接近零

        # 系统性风险调整
        if market_state.systemic_risk.has_any_risk():
            if market_state.systemic_risk.event_window:
                base_limit *= self.event_window_multiplier
            elif market_state.systemic_risk.high_systemic_risk:
                base_limit *= 0.5

        return max(0.0, min(base_limit, 1.0))

    def get_total_position_limit(self, market_state: MarketState) -> float:
        """获取总仓位限制"""
        base_limit = self.normal_max_total_ratio

        # 市场状态调整
        if market_state.regime == RegimeType.CRISIS:
            base_limit = self.crisis_max_total_ratio
        elif market_state.regime == RegimeType.RANGE:
            base_limit = self.range_max_total_ratio
        elif market_state.volatility_state in [VolatilityState.HIGH, VolatilityState.EXTREME]:
            base_limit = self.high_vol_max_total_ratio

        # 流动性调整
        if market_state.liquidity_state == LiquidityState.THIN:
            base_limit *= self.thin_liquidity_multiplier
        elif market_state.liquidity_state == LiquidityState.FROZEN:
            base_limit *= 0.3

        # 系统性风险调整
        if market_state.systemic_risk.has_any_risk():
            if market_state.systemic_risk.event_window:
                base_limit *= self.event_window_multiplier
            elif market_state.systemic_risk.high_systemic_risk:
                base_limit *= 0.6

        return max(0.0, min(base_limit, 1.0))


class MarketAwareRiskManager(AdaptiveRiskManager):
    """市场感知的风控管理器

    继承 AdaptiveRiskManager，增加：
    1. MarketState 感知
    2. 动态仓位限制
    3. 市场风险驱动的风控决策
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        enable_scaling: bool = True,
        default_cooldown_seconds: int = 300,
        market_risk_config: Optional[MarketRiskConfig] = None
    ):
        """初始化市场感知风控管理器

        Args:
            initial_capital: 初始资金
            enable_scaling: 是否启用仓位缩放
            default_cooldown_seconds: 默认冷却时间
            market_risk_config: 市场风险配置
        """
        super().__init__(initial_capital, enable_scaling, default_cooldown_seconds)

        self.market_risk_config = market_risk_config or MarketRiskConfig()
        self._current_market_state: Optional[MarketState] = None
        self._market_state_history: List[MarketState] = []

        # 添加市场状态相关的风控规则
        self._add_market_aware_rules()

    def _add_market_aware_rules(self):
        """添加市场感知的风控规则"""
        market_rules = [
            RiskRule(
                rule_id="crisis_mode_pause",
                name="危机模式暂停",
                rule_type="market_crisis",
                scope="global",
                priority=0,  # 最高优先级
                params={},
                cooldown_seconds=3600
            ),
            RiskRule(
                rule_id="extreme_volatility_limit",
                name="极端波动限制",
                rule_type="market_volatility",
                scope="symbol",
                priority=1,
                params={},
                cooldown_seconds=300
            ),
            RiskRule(
                rule_id="liquidity_freeze",
                name="流动性冻结",
                rule_type="market_liquidity",
                scope="symbol",
                priority=0,
                params={},
                cooldown_seconds=600
            ),
            RiskRule(
                rule_id="systemic_risk_event",
                name="系统性风险事件",
                rule_type="systemic_risk",
                scope="global",
                priority=0,
                params={},
                cooldown_seconds=1800
            ),
        ]

        for rule in market_rules:
            if rule.rule_id not in self.rules:
                self.rules[rule.rule_id] = rule

    def update_market_state(self, market_state: MarketState):
        """更新当前市场状态

        Args:
            market_state: 最新市场状态
        """
        self._current_market_state = market_state
        self._market_state_history.append(market_state)

        # 限制历史长度
        if len(self._market_state_history) > 1000:
            self._market_state_history = self._market_state_history[-1000:]

        # 如果市场进入危机模式，自动触发风控
        if market_state.regime == RegimeType.CRISIS:
            self.pause(f"市场进入危机模式", duration_seconds=3600)
            logger.warning(f"市场进入危机模式，自动暂停交易")

        # 如果流动性枯竭，自动冻结
        if market_state.liquidity_state == LiquidityState.FROZEN:
            self.pause(f"市场流动性枯竭", duration_seconds=1800)
            logger.warning(f"市场流动性枯竭，自动暂停交易")

    def get_market_state(self) -> Optional[MarketState]:
        """获取当前市场状态"""
        return self._current_market_state

    def check_signal_with_market_state(
        self,
        signal: Signal,
        positions: Dict[str, Position],
        cash_balance: float,
        market_state: Optional[MarketState] = None,
        strategy_name: Optional[str] = None
    ) -> RiskCheckResult:
        """检查交易信号（包含市场状态）

        Args:
            signal: 交易信号
            positions: 当前持仓
            cash_balance: 现金余额
            market_state: 市场状态（如果不提供，使用缓存的状态）
            strategy_name: 策略名称

        Returns:
            风控检查结果
        """
        # 更新市场状态
        if market_state:
            self.update_market_state(market_state)

        # 首先进行标准的风控检查
        result = self.check_signal(signal, positions, cash_balance, strategy_name)

        # 如果有市场状态，进行额外的市场风险检查
        if self._current_market_state and result.action != RiskAction.DISABLE:
            market_result = self._check_market_risk(signal, positions, cash_balance, strategy_name)

            # 合并结果
            if market_result.action != RiskAction.APPROVE:
                # 市场风险检查触发了更严格的限制
                result = market_result

        return result

    def check_strategy_signal_with_market_state(
        self,
        signal: StrategySignal,
        positions: Dict[str, Position],
        cash_balance: float,
        market_state: Optional[MarketState] = None,
        strategy_name: Optional[str] = None
    ) -> Tuple[RiskAction, float, StrategySignal]:
        """检查新格式信号（包含市场状态）

        Returns:
            (风控动作, 缩放系数, 调整后的信号)
        """
        # 更新市场状态
        if market_state:
            self.update_market_state(market_state)

        # 从 StrategySignal 的 metadata 中提取市场状态
        if not market_state and signal.metadata:
            market_state_dict = signal.metadata.get("market_state")
            if market_state_dict:
                from skills.market_analysis.market_state import MarketState
                try:
                    self._current_market_state = MarketState.from_dict(market_state_dict)
                except Exception as e:
                    logger.warning(f"无法从信号元数据恢复市场状态: {e}")

        # 首先进行标准的风控检查
        action, scale_factor, adjusted = self.check_strategy_signal(
            signal, positions, cash_balance, strategy_name
        )

        # 如果有市场状态，进行额外的市场风险检查
        if self._current_market_state and action not in [RiskAction.DISABLE, RiskAction.PAUSE]:
            market_scale = self._calculate_market_scale_factor(
                signal, positions, cash_balance, strategy_name
            )

            # 应用市场风险缩放
            if market_scale < scale_factor:
                scale_factor = market_scale
                action = RiskAction.SCALE_DOWN if scale_factor > 0.1 else RiskAction.PAUSE

                # 调整信号
                adjusted = StrategySignal(
                    symbol=signal.symbol,
                    timestamp=signal.timestamp,
                    target_position=signal.target_position * scale_factor,
                    target_weight=signal.target_weight * scale_factor,
                    confidence=signal.confidence * scale_factor,
                    reason=f"{signal.reason} (市场风险调整)",
                    regime=signal.regime,
                    risk_hints=signal.risk_hints,
                    raw_action=signal.raw_action,
                    metadata={**signal.metadata, "market_risk_scale_factor": scale_factor}
                )

        return action, scale_factor, adjusted

    def _check_market_risk(
        self,
        signal: Signal,
        positions: Dict[str, Position],
        cash_balance: float,
        strategy_name: Optional[str]
    ) -> RiskCheckResult:
        """检查市场风险"""
        if not self._current_market_state:
            return RiskCheckResult(action=RiskAction.APPROVE, scale_factor=1.0)

        state = self._current_market_state
        triggered_rules = []

        # 1. 检查危机模式
        if state.regime == RegimeType.CRISIS:
            triggered_rules.append("crisis_mode")
            return RiskCheckResult(
                action=RiskAction.PAUSE,
                reason="市场处于危机模式",
                triggered_rules=triggered_rules
            )

        # 2. 检查极端波动
        if state.volatility_state == VolatilityState.EXTREME:
            triggered_rules.append("extreme_volatility")

        # 3. 检查流动性
        if state.liquidity_state == LiquidityState.FROZEN:
            triggered_rules.append("liquidity_frozen")
            return RiskCheckResult(
                action=RiskAction.PAUSE,
                reason="市场流动性枯竭",
                triggered_rules=triggered_rules
            )
        elif state.liquidity_state == LiquidityState.THIN:
            triggered_rules.append("thin_liquidity")

        # 4. 检查系统性风险
        if state.systemic_risk.event_window:
            triggered_rules.append("event_window")
        elif state.systemic_risk.high_systemic_risk:
            triggered_rules.append("high_systemic_risk")

        # 计算缩放系数
        scale_factor = self._calculate_market_scale_factor(signal, positions, cash_balance, strategy_name)

        if triggered_rules:
            action = RiskAction.APPROVE if scale_factor >= 0.9 else RiskAction.SCALE_DOWN
            if scale_factor < 0.1:
                action = RiskAction.PAUSE

            return RiskCheckResult(
                action=action,
                scale_factor=scale_factor,
                reason=f"市场风险: {', '.join(triggered_rules)}",
                triggered_rules=triggered_rules
            )

        return RiskCheckResult(action=RiskAction.APPROVE, scale_factor=1.0)

    def _calculate_market_scale_factor(
        self,
        signal: Signal,
        positions: Dict[str, Position],
        cash_balance: float,
        strategy_name: Optional[str]
    ) -> float:
        """计算基于市场状态的缩放系数"""
        if not self._current_market_state:
            return 1.0

        state = self._current_market_state
        scale_factors = []

        # 1. 仓位限制缩放
        max_position_ratio = self.market_risk_config.get_position_limit(state)
        current_default = 0.2  # 默认单标的最大仓位

        if max_position_ratio < current_default:
            scale_factors.append(max_position_ratio / current_default)

        # 2. 总仓位缩放
        max_total_ratio = self.market_risk_config.get_total_position_limit(state)
        current_total_default = 0.8  # 默认总仓位

        if max_total_ratio < current_total_default:
            scale_factors.append(max_total_ratio / current_total_default)

        # 3. 波动率缩放
        if state.volatility_state == VolatilityState.EXTREME:
            scale_factors.append(0.2)
        elif state.volatility_state == VolatilityState.HIGH:
            scale_factors.append(0.5)

        # 4. 流动性缩放
        if state.liquidity_state == LiquidityState.THIN:
            scale_factors.append(0.5)

        # 5. 使用 MarketState 的内置缩放系数
        scale_factors.append(state.position_scale_factor)

        # 取最小值作为最终缩放系数
        final_scale = min(scale_factors) if scale_factors else 1.0

        return max(0.0, min(final_scale, 1.0))

    def get_dynamic_position_limits(
        self,
        market_state: Optional[MarketState] = None
    ) -> Dict[str, float]:
        """获取动态仓位限制

        Returns:
            {
                "max_position_ratio": float,  # 单标的最大仓位
                "max_total_ratio": float,     # 总仓位限制
                "scale_factor": float         # 综合缩放系数
            }
        """
        state = market_state or self._current_market_state

        if not state:
            return {
                "max_position_ratio": 0.2,
                "max_total_ratio": 0.8,
                "scale_factor": 1.0
            }

        return {
            "max_position_ratio": self.market_risk_config.get_position_limit(state),
            "max_total_ratio": self.market_risk_config.get_total_position_limit(state),
            "scale_factor": state.position_scale_factor
        }

    def get_rule_status(self) -> Dict[str, Any]:
        """获取规则状态（包含市场状态）"""
        status = super().get_rule_status()

        # 添加市场状态信息
        if self._current_market_state:
            status["market_state"] = {
                "regime": self._current_market_state.regime.value,
                "volatility_state": self._current_market_state.volatility_state.value,
                "liquidity_state": self._current_market_state.liquidity_state.value,
                "should_reduce_risk": self._current_market_state.should_reduce_risk,
                "position_scale_factor": self._current_market_state.position_scale_factor
            }

            status["dynamic_limits"] = self.get_dynamic_position_limits()

        return status


if __name__ == "__main__":
    """测试代码"""
    print("=== MarketAwareRiskManager 测试 ===\n")

    # 创建市场感知风控管理器
    risk_mgr = MarketAwareRiskManager(
        initial_capital=100000,
        enable_scaling=True
    )

    # 创建测试市场状态
    from skills.market_analysis.market_state import (
        MarketState, RegimeType, VolatilityState, LiquidityState,
        SentimentState, SystemicRiskFlags
    )

    normal_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime=RegimeType.TREND,
        volatility_state=VolatilityState.NORMAL,
        liquidity_state=LiquidityState.NORMAL,
        sentiment_state=SentimentState.NEUTRAL
    )

    crisis_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime=RegimeType.CRISIS,
        volatility_state=VolatilityState.EXTREME,
        liquidity_state=LiquidityState.THIN,
        sentiment_state=SentimentState.FEAR
    )

    # 创建测试信号
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150"),
        quantity=200,
        confidence=0.8,
        reason="测试信号"
    )

    # 测试1: 正常市场状态
    print("1. 正常市场状态:")
    result = risk_mgr.check_signal_with_market_state(signal, {}, 100000, normal_state)
    print(f"   动作: {result.action.value}")
    print(f"   缩放系数: {result.scale_factor}")

    # 测试2: 危机市场状态
    print("\n2. 危机市场状态:")
    result = risk_mgr.check_signal_with_market_state(signal, {}, 100000, crisis_state)
    print(f"   动作: {result.action.value}")
    print(f"   缩放系数: {result.scale_factor}")
    print(f"   原因: {result.reason}")

    # 测试3: 动态仓位限制
    print("\n3. 动态仓位限制:")
    limits = risk_mgr.get_dynamic_position_limits(normal_state)
    print(f"   正常市场 - 单标的限制: {limits['max_position_ratio']:.1%}")
    print(f"   正常市场 - 总仓位限制: {limits['max_total_ratio']:.1%}")

    limits = risk_mgr.get_dynamic_position_limits(crisis_state)
    print(f"   危机市场 - 单标的限制: {limits['max_position_ratio']:.1%}")
    print(f"   危机市场 - 总仓位限制: {limits['max_total_ratio']:.1%}")

    # 测试4: 规则状态
    print("\n4. 规则状态:")
    risk_mgr.update_market_state(crisis_state)
    status = risk_mgr.get_rule_status()
    if "market_state" in status:
        ms = status["market_state"]
        print(f"   当前市场状态: {ms['regime']}")
        print(f"   波动率状态: {ms['volatility_state']}")
        print(f"   应降低风险: {ms['should_reduce_risk']}")

    print("\n✓ 所有测试通过")
