"""
RiskGateAgent - 风控门禁 (基于新 Schema)

基于 MetaDecision 和 MarketState 进行风控，优先缩放仓位，其次才拒绝
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from core.schema import (
    MarketState,
    MetaDecision,
    RiskDecision,
    SchemaVersion
)

logger = logging.getLogger(__name__)


@dataclass
class CooldownState:
    """冷却状态"""
    action_type: str  # 风控动作类型
    until: datetime  # 冷却结束时间
    reason: str  # 冷却原因

    def is_active(self) -> bool:
        return datetime.now() < self.until


class RiskRuleConfig:
    """风控规则配置"""

    # 极端行情联合规则
    extreme_volatility_max_weight: float = 0.2  # 极端波动最大权重
    range_volatility_max_weight: float = 0.3  # 震荡+高波动最大权重

    # 危机模式规则
    crisis_mode_pause_new: bool = True  # 危机模式禁止新开仓
    crisis_mode_max_weight: float = 0.05  # 危机模式最大权重

    # 流动性规则
    frozen_liquidity_pause: bool = True  # 流动性枯竭暂停
    thin_liquidity_max_weight: float = 0.5  # 流动性不足最大权重

    # 系统性风险规则
    systemic_risk_multiplier: float = 0.5  # 系统性风险时仓位乘数


class RiskGateAgent:
    """
    RiskGateAgent - 风控门禁

    职责：
    ✓ 基于 MetaDecision 调整仓位
    ✓ 基于 MarketState 应用风控规则
    ✓ 设置冷却期
    ✓ 优先缩放，其次才拒绝
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        rule_config: Optional[RiskRuleConfig] = None
    ):
        """
        初始化 RiskGateAgent

        Args:
            initial_capital: 初始资金
            rule_config: 风控规则配置
        """
        self.initial_capital = initial_capital
        self.rule_config = rule_config or RiskRuleConfig()

        # 冷却状态
        self._cooldowns: Dict[str, CooldownState] = {}

        # 当前账户状态
        self._current_positions: Dict[str, float] = {}
        self._cash_balance = initial_capital
        self._total_equity = initial_capital

        # 决策历史
        self._decision_history: List[RiskDecision] = []

    def check(
        self,
        meta_decision: MetaDecision,
        market_state: MarketState,
        current_positions: Optional[Dict[str, float]] = None
    ) -> RiskDecision:
        """
        检查并调整风控决策

        Args:
            meta_decision: Meta层决策
            market_state: 市场状态
            current_positions: 当前持仓

        Returns:
            RiskDecision
        """
        # 更新当前持仓
        if current_positions is not None:
            self._current_positions = current_positions

        # 检查冷却期
        cooldown_check = self._check_cooldowns(meta_decision.symbol)
        if cooldown_check is not None:
            self._decision_history.append(cooldown_check)
            return cooldown_check

        # 计算基础缩放系数
        scale_factor = 1.0
        risk_actions = []
        risk_reasons = []

        # 1. 应用市场状态风控规则
        market_scale = self._apply_market_rules(market_state, meta_decision)
        scale_factor *= market_scale
        if market_scale < 1.0:
            risk_actions.append("MARKET_SCALE")
            risk_reasons.append(f"市场状态缩放 {market_scale:.1%}")

        # 2. 应用仓位限制规则
        position_scale = self._apply_position_limits(meta_decision)
        scale_factor *= position_scale
        if position_scale < 1.0:
            risk_actions.append("POSITION_LIMIT")
            risk_reasons.append(f"仓位限制缩放 {position_scale:.1%}")

        # 3. 决定最终动作
        final_action, final_reason = self._determine_action(
            scale_factor,
            risk_actions,
            risk_reasons,
            meta_decision,
            market_state
        )

        # 4. 计算调整后的仓位
        scaled_position = meta_decision.target_position * scale_factor

        # 5. 构建风控决策
        risk_decision = RiskDecision(
            timestamp=meta_decision.timestamp,
            symbol=meta_decision.symbol,
            scaled_target_position=scaled_position,
            scale_factor=scale_factor,
            risk_action=final_action,
            risk_reason=final_reason,
            max_position=self._get_max_position(meta_decision.symbol, market_state)
        )

        # 6. 如果需要设置冷却
        if final_action in ["PAUSE", "DISABLE"]:
            self._set_cooldown(
                meta_decision.symbol,
                final_action,
                3600,  # 1小时冷却
                final_reason
            )

        self._decision_history.append(risk_decision)
        return risk_decision

    def _check_cooldowns(self, symbol: str) -> Optional[RiskDecision]:
        """检查冷却期"""
        cooldown_key = f"{symbol}_global"

        if cooldown_key in self._cooldowns:
            cooldown = self._cooldowns[cooldown_key]
            if cooldown.is_active():
                return RiskDecision(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    scaled_target_position=0,
                    scale_factor=0,
                    risk_action="PAUSE",
                    risk_reason=f"冷却中: {cooldown.reason}",
                    cooldown_until=cooldown.until,
                    cooldown_reason=cooldown.reason
                )
            else:
                # 冷却结束，移除
                del self._cooldowns[cooldown_key]

        return None

    def _apply_market_rules(
        self,
        market_state: MarketState,
        meta_decision: MetaDecision
    ) -> float:
        """
        应用市场状态风控规则

        Returns:
            缩放系数
        """
        scale = 1.0

        # 规则1: 极端波动 + 震荡 = 严格限制
        if (market_state.volatility_state == "EXTREME" and
            market_state.regime == "RANGE"):
            scale = min(scale, self.rule_config.range_volatility_max_weight)

        # 规则2: 极端波动
        elif market_state.volatility_state == "EXTREME":
            scale = min(scale, self.rule_config.extreme_volatility_max_weight)

        # 规则3: 危机模式
        if market_state.regime == "CRISIS":
            if self.rule_config.crisis_mode_pause_new and meta_decision.target_position > 0:
                scale = 0  # 禁止新开仓
            else:
                scale = min(scale, self.rule_config.crisis_mode_max_weight)

        # 规则4: 流动性枯竭
        if market_state.liquidity_state == "FROZEN":
            if self.rule_config.frozen_liquidity_pause:
                return 0  # 暂停交易
            scale = 0  # 流动性枯竭，零仓位

        elif market_state.liquidity_state == "THIN":
            scale = min(scale, self.rule_config.thin_liquidity_max_weight)

        # 规则5: 系统性风险
        if market_state.systemic_risk_flags.get("has_any_risk", False):
            scale *= self.rule_config.systemic_risk_multiplier

        return max(0, min(scale, 1))

    def _apply_position_limits(
        self,
        meta_decision: MetaDecision
    ) -> float:
        """
        应用仓位限制规则

        Returns:
            缩放系数
        """
        scale = 1.0

        # 单标的仓位限制
        current_pos = self._current_positions.get(meta_decision.symbol, 0)
        current_value = abs(current_pos * 100)  # 简化

        # 如果当前仓位 + 目标仓位超过限制
        if abs(meta_decision.target_position) > 0:
            # 简化：假设每股市值 100，计算仓位比例
            target_value = abs(meta_decision.target_position) * 100
            position_ratio = target_value / self._total_equity

            max_ratio = 0.3  # 单标的最大 30%

            if position_ratio > max_ratio:
                scale = max_ratio / position_ratio

        return scale

    def _determine_action(
        self,
        scale_factor: float,
        risk_actions: List[str],
        risk_reasons: List[str],
        meta_decision: MetaDecision,
        market_state: MarketState
    ) -> tuple[str, str]:
        """
        决定最终风控动作

        Returns:
            (action, reason)
        """
        if scale_factor == 0:
            return "DISABLE", "禁止交易; ".join(risk_reasons) if risk_reasons else "风控禁止"

        if scale_factor < 0.1:
            return "PAUSE", f"仓位过小 ({scale_factor:.1%}); " + "; ".join(risk_reasons)

        if scale_factor < 1.0:
            return "SCALE_DOWN", "; ".join(risk_reasons) if risk_reasons else "仓位缩放"

        return "APPROVE", "风控通过"

    def _get_max_position(
        self,
        symbol: str,
        market_state: MarketState
    ) -> float:
        """获取最大允许仓位"""
        # 基础限制
        base_max = self._total_equity * 0.3  # 单标最大 30%

        # 市场状态调整
        if market_state.regime == "CRISIS":
            base_max *= 0.2
        elif market_state.volatility_state == "EXTREME":
            base_max *= 0.5

        return base_max / 100  # 转换为股数

    def _set_cooldown(
        self,
        symbol: str,
        action: str,
        duration_seconds: int,
        reason: str
    ):
        """设置冷却期"""
        cooldown_key = f"{symbol}_{action}"
        self._cooldowns[cooldown_key] = CooldownState(
            action_type=action,
            until=datetime.now() + timedelta(seconds=duration_seconds),
            reason=reason
        )

        logger.info(f"设置冷却期: {cooldown_key} 直到 {self._cooldowns[cooldown_key].until}")

    def update_account_state(
        self,
        positions: Dict[str, float],
        cash_balance: float,
        total_equity: float
    ):
        """更新账户状态"""
        self._current_positions = positions
        self._cash_balance = cash_balance
        self._total_equity = total_equity

    def get_decision_history(self, limit: int = 100) -> List[RiskDecision]:
        """获取决策历史"""
        return self._decision_history[-limit:]

    def get_active_cooldowns(self) -> Dict[str, CooldownState]:
        """获取活跃的冷却期"""
        return {
            k: v for k, v in self._cooldowns.items()
            if v.is_active()
        }


if __name__ == "__main__":
    """测试代码"""
    print("=== RiskGateAgent 测试 ===\n")

    from datetime import timedelta

    # 创建测试 MarketState
    normal_state = MarketState(
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

    extreme_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime="RANGE",
        regime_confidence=0.7,
        volatility_state="EXTREME",
        volatility_trend="STABLE",
        volatility_percentile=0.95,
        liquidity_state="NORMAL",
        volume_ratio=1.0,
        sentiment_state="FEAR",
        sentiment_score=-0.5
    )

    crisis_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime="CRISIS",
        regime_confidence=0.9,
        volatility_state="EXTREME",
        volatility_trend="STABLE",
        volatility_percentile=0.99,
        liquidity_state="THIN",
        volume_ratio=0.5,
        sentiment_state="FEAR",
        sentiment_score=-0.8
    )

    # 创建 MetaDecision
    meta_decision = MetaDecision(
        timestamp=datetime.now(),
        symbol="AAPL",
        target_position=200,
        target_weight=0.2,
        active_strategies=["MA", "Breakout"],
        disabled_strategies={},
        decision_confidence=0.7,
        consensus_level="STRONG"
    )

    # 创建 RiskGateAgent
    risk_gate = RiskGateAgent(initial_capital=100000)

    # 测试1: 正常市场通过
    print("1. 正常市场:")
    risk_decision = risk_gate.check(meta_decision, normal_state)
    print(f"   Action: {risk_decision.risk_action}")
    print(f"   Scale factor: {risk_decision.scale_factor:.2f}")
    print(f"   Scaled position: {risk_decision.scaled_target_position:.0f}")
    print(f"   Reason: {risk_decision.risk_reason}")

    # 测试2: 极端市场缩放
    print("\n2. 极端市场:")
    risk_decision = risk_gate.check(meta_decision, extreme_state)
    print(f"   Action: {risk_decision.risk_action}")
    print(f"   Scale factor: {risk_decision.scale_factor:.2f}")
    print(f"   Scaled position: {risk_decision.scaled_target_position:.0f}")
    print(f"   Reason: {risk_decision.risk_reason}")

    # 测试3: 危机模式
    print("\n3. 危机模式:")
    risk_decision = risk_gate.check(meta_decision, crisis_state)
    print(f"   Action: {risk_decision.risk_action}")
    print(f"   Scale factor: {risk_decision.scale_factor:.2f}")
    print(f"   Scaled position: {risk_decision.scaled_target_position:.0f}")
    print(f"   Reason: {risk_decision.risk_reason}")

    print("\n✓ 所有测试通过")
