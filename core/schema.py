"""
MKCS 系统 Schema 定义

这个模块定义了系统中所有层之间通信的数据结构。
所有 schema 都是可序列化、可版本化的。

设计原则：
1. 每个层只能通过 schema 与其他层通信
2. 不允许越权访问
3. 所有决策必须可追溯
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List, Optional, Literal
from enum import Enum
import json


class SchemaVersion:
    """Schema 版本控制"""
    CURRENT = "v1"


# =============================================================================
# L3: Market Analysis 输出
# =============================================================================

@dataclass
class MarketState:
    """
    市场状态（由 Market Analysis 层产出）

    职责边界：
    ✓ 描述市场当前状态
    ✓ 描述市场历史特征
    ✗ 不产生任何买卖建议
    ✗ 不包含任何预测
    """

    # 基本信息
    timestamp: datetime
    symbol: str

    # 市场状态
    regime: Literal["TREND", "RANGE", "HIGH_VOL", "CRISIS", "UNKNOWN"]
    regime_confidence: float  # 0-1

    # 波动率
    volatility_state: Literal["LOW", "NORMAL", "HIGH", "EXTREME"]
    volatility_trend: Literal["RISING", "FALLING", "STABLE"]
    volatility_percentile: float  # 0-1，相对历史分位数

    # 流动性
    liquidity_state: Literal["NORMAL", "THIN", "FROZEN"]
    volume_ratio: float  # 相对历史均值

    # 情绪
    sentiment_state: Literal["FEAR", "NEUTRAL", "GREED"]
    sentiment_score: float  # -1 到 1

    # 系统性风险
    systemic_risk_flags: Dict[str, bool] = field(default_factory=dict)

    # 趋势信息
    trend_strength: float = 0.0  # 0-1
    trend_direction: Literal["UP", "DOWN", "NEUTRAL"] = "NEUTRAL"

    # 元数据
    schema_version: str = SchemaVersion.CURRENT

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "volatility_state": self.volatility_state,
            "volatility_trend": self.volatility_trend,
            "volatility_percentile": self.volatility_percentile,
            "liquidity_state": self.liquidity_state,
            "volume_ratio": self.volume_ratio,
            "sentiment_state": self.sentiment_state,
            "sentiment_score": self.sentiment_score,
            "systemic_risk_flags": self.systemic_risk_flags,
            "trend_strength": self.trend_strength,
            "trend_direction": self.trend_direction,
            "schema_version": self.schema_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketState":
        """反序列化"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


# =============================================================================
# L2: Alpha 策略输出
# =============================================================================

@dataclass
class AlphaOpinion:
    """
    Alpha 策略观点（由 Alpha 策略层产出）

    职责边界：
    ✓ 表达对方向的看法
    ✓ 表达信心强度
    ✓ 给出理由
    ✗ 不决定仓位
    ✗ 不产生订单
    """

    # 策略标识
    strategy_name: str
    timestamp: datetime
    symbol: str

    # 观点内容
    direction: Literal[-1, 0, 1]  # -1: 做空, 0: 中性, 1: 做多
    strength: float  # 0-1，信号强度
    confidence: float  # 0-1，置信度
    horizon: Literal["intraday", "swing", "daily"]  # 持仓周期

    # 禁用状态（一等状态，不是 silent ignore）
    is_disabled: bool = False
    disabled_reason: str = ""  # "market_regime: RANGE", "volatility: EXTREME", etc.

    # 理由
    reason: str = ""

    # 元数据
    schema_version: str = SchemaVersion.CURRENT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "strategy_name": self.strategy_name,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "horizon": self.horizon,
            "is_disabled": self.is_disabled,
            "disabled_reason": self.disabled_reason,
            "reason": self.reason,
            "schema_version": self.schema_version,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlphaOpinion":
        """反序列化"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def has_opinion(self) -> bool:
        """是否有有效观点（未被禁用且非中性）"""
        return not self.is_disabled and self.direction != 0

    def get_position_signal(self) -> float:
        """获取仓位信号（-1 到 1）"""
        if self.is_disabled:
            return 0.0
        return self.direction * self.strength * self.confidence


# =============================================================================
# L4: Meta / Brain 输出
# =============================================================================

@dataclass
class MetaDecision:
    """
    元决策（由 MetaStrategy / Brain 层产出）

    职责边界：
    ✓ 融合多个 Alpha 观点
    ✓ 决定最终目标仓位
    ✓ 说明哪些策略被启用/禁用
    ✗ 不直接下单
    """

    # 决策标识
    timestamp: datetime
    symbol: str

    # 目标仓位（按 symbol）
    target_position: float  # 股数或合约数，可负表示做空
    target_weight: float  # 权重 0-1

    # 决策依据
    active_strategies: List[str] = field(default_factory=list)  # 参与的策略
    disabled_strategies: Dict[str, str] = field(default_factory=dict)  # {"ma": "regime: RANGE"}

    # 决策质量
    decision_confidence: float = 0.5  # 0-1
    consensus_level: Literal["STRONG", "WEAK", "NONE"] = "NONE"  # 策略共识程度

    # 决策理由
    reasoning: str = ""

    # 元数据
    schema_version: str = SchemaVersion.CURRENT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "target_position": self.target_position,
            "target_weight": self.target_weight,
            "active_strategies": self.active_strategies,
            "disabled_strategies": self.disabled_strategies,
            "decision_confidence": self.decision_confidence,
            "consensus_level": self.consensus_level,
            "reasoning": self.reasoning,
            "schema_version": self.schema_version,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaDecision":
        """反序列化"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


# =============================================================================
# L4: Risk 风控输出
# =============================================================================

@dataclass
class RiskDecision:
    """
    风控决策（由 Risk Gate 层产出）

    职责边界：
    ✓ 基于 MetaDecision 调整仓位
    ✓ 添加风控约束
    ✓ 设置冷却期
    """

    # 决策标识
    timestamp: datetime
    symbol: str

    # 调整后的目标仓位
    scaled_target_position: float  # 经过风控调整后的仓位
    scale_factor: float  # 应用的缩放系数 0-1

    # 风控动作
    risk_action: Literal["APPROVE", "SCALE_DOWN", "PAUSE", "DISABLE"]
    risk_reason: str = ""

    # 约束条件
    max_position: float = 0.0  # 最大允许仓位
    stop_loss: Optional[float] = None  # 止损价
    take_profit: Optional[float] = None  # 止盈价

    # 冷却期
    cooldown_until: Optional[datetime] = None  # 冷却结束时间
    cooldown_reason: str = ""

    # 元数据
    schema_version: str = SchemaVersion.CURRENT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "scaled_target_position": self.scaled_target_position,
            "scale_factor": self.scale_factor,
            "risk_action": self.risk_action,
            "risk_reason": self.risk_reason,
            "max_position": self.max_position,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "cooldown_reason": self.cooldown_reason,
            "schema_version": self.schema_version,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskDecision":
        """反序列化"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("cooldown_until"):
            data["cooldown_until"] = datetime.fromisoformat(data["cooldown_until"])
        return cls(**data)

    def is_allowed(self) -> bool:
        """是否允许交易"""
        return self.risk_action in ["APPROVE", "SCALE_DOWN"]


# =============================================================================
# L1: Execution 输出
# =============================================================================

@dataclass
class ExecutionResult:
    """
    执行结果（由 Execution 层产出）

    职责边界：
    ✓ 记录实际执行情况
    ✓ 记录滑点和成交价
    ✓ 记录失败原因
    """

    # 执行标识
    timestamp: datetime
    symbol: str
    execution_mode: Literal["paper", "replay", "live", "dry_run"]

    # 目标 vs 实际
    target_position: float  # 目标仓位
    actual_position: float  # 实际仓位
    target_price: Optional[float] = None  # 期望价格
    fill_price: Optional[float] = None  # 实际成交价

    # 执行状态
    status: Literal["FILLED", "PARTIAL", "FAILED", "SKIPPED"] = "SKIPPED"
    fill_quantity: float = 0.0  # 成交数量
    slippage: float = 0.0  # 滑点（相对价格的百分比）

    # 错误信息
    error_message: str = ""

    # 当前持仓状态
    current_positions: Dict[str, float] = field(default_factory=dict)  # symbol -> quantity
    cash_balance: float = 0.0

    # 元数据
    schema_version: str = SchemaVersion.CURRENT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "execution_mode": self.execution_mode,
            "target_position": self.target_position,
            "actual_position": self.actual_position,
            "target_price": self.target_price,
            "fill_price": self.fill_price,
            "status": self.status,
            "fill_quantity": self.fill_quantity,
            "slippage": self.slippage,
            "error_message": self.error_message,
            "current_positions": self.current_positions,
            "cash_balance": self.cash_balance,
            "schema_version": self.schema_version,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        """反序列化"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


# =============================================================================
# L5: LLM Advisor 输出
# =============================================================================

@dataclass
class AdvisorReport:
    """
    顾问报告（由 LLM Advisor 层产出）

    职责边界：
    ✓ 盘后总结
    ✓ 规则建议
    ✓ 参数建议
    ✗ 不参与实时决策
    """

    # 报告标识
    report_type: Literal["daily_summary", "rule_proposal", "param_suggestion", "post_mortem"]
    timestamp: datetime
    for_date: datetime  # 报告针对的日期

    # 报告内容
    summary: str = ""  # 总结文本
    key_events: List[str] = field(default_factory=list)  # 关键事件
    suggestions: List[str] = field(default_factory=list)  # 建议

    # 规则/参数建议（如果是 proposal 类型）
    proposed_rules: List[Dict[str, Any]] = field(default_factory=list)
    proposed_params: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    schema_version: str = SchemaVersion.CURRENT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "report_type": self.report_type,
            "timestamp": self.timestamp.isoformat(),
            "for_date": self.for_date.isoformat(),
            "summary": self.summary,
            "key_events": self.key_events,
            "suggestions": self.suggestions,
            "proposed_rules": self.proposed_rules,
            "proposed_params": self.proposed_params,
            "schema_version": self.schema_version,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvisorReport":
        """反序列化"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["for_date"] = datetime.fromisoformat(data["for_date"])
        return cls(**data)


# =============================================================================
# Alpha Gating 配置
# =============================================================================

@dataclass
class AlphaGatingConfig:
    """
    Alpha 策略禁用配置

    定义哪些市场状态下哪些策略被禁用或降权
    """

    strategy_name: str

    # 完全禁用的市场状态
    disabled_regimes: List[str] = field(default_factory=list)

    # 降权的市场状态
    scaled_regimes: Dict[str, float] = field(default_factory=dict)  # {"RANGE": 0.3}

    # 禁用的波动率状态
    disabled_volatility: List[str] = field(default_factory=list)

    # 降权的波动率状态
    scaled_volatility: Dict[str, float] = field(default_factory=dict)

    # 禁用的流动性状态
    disabled_liquidity: List[str] = field(default_factory=list)

    def should_disable(self, market_state: MarketState) -> tuple[bool, str]:
        """
        判断是否应该禁用该策略

        Returns:
            (should_disable, reason)
        """
        # 检查 regime
        if market_state.regime in self.disabled_regimes:
            return True, f"regime_{market_state.regime}"

        # 检查波动率
        if market_state.volatility_state in self.disabled_volatility:
            return True, f"volatility_{market_state.volatility_state}"

        # 检查流动性
        if market_state.liquidity_state in self.disabled_liquidity:
            return True, f"liquidity_{market_state.liquidity_state}"

        return False, ""

    def get_scale_factor(self, market_state: MarketState) -> float:
        """获取仓位缩放系数"""
        # 优先使用 regime 缩放
        if market_state.regime in self.scaled_regimes:
            return self.scaled_regimes[market_state.regime]

        # 其次使用波动率缩放
        if market_state.volatility_state in self.scaled_volatility:
            return self.scaled_volatility[market_state.volatility_state]

        return 1.0


# =============================================================================
# 默认 Gating 配置
# =============================================================================

DEFAULT_ALPHA_GATING = {
    "ma": AlphaGatingConfig(
        strategy_name="ma",
        disabled_regimes=["CRISIS"],
        scaled_regimes={"RANGE": 0.3},
        disabled_volatility=[],
        scaled_volatility={"EXTREME": 0.2, "HIGH": 0.5},
        disabled_liquidity=["FROZEN"]
    ),
    "breakout": AlphaGatingConfig(
        strategy_name="breakout",
        disabled_regimes=["CRISIS", "RANGE"],  # 震荡市禁用突破
        scaled_regimes={},
        disabled_volatility=[],
        scaled_volatility={"EXTREME": 0.3},
        disabled_liquidity=["FROZEN"]
    ),
    "ml": AlphaGatingConfig(
        strategy_name="ml",
        disabled_regimes=["CRISIS"],
        scaled_regimes={"RANGE": 0.5},
        disabled_volatility=[],
        scaled_volatility={"EXTREME": 0.1},
        disabled_liquidity=["FROZEN"]
    ),
}


if __name__ == "__main__":
    """测试代码"""
    print("=== MKCS Schema 测试 ===\n")

    # 测试 MarketState
    print("1. MarketState:")
    ms = MarketState(
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
    print(f"   Regime: {ms.regime}")
    print(f"   Serializable: {ms.to_dict()}")

    # 测试 AlphaOpinion
    print("\n2. AlphaOpinion:")
    ao = AlphaOpinion(
        strategy_name="ma",
        timestamp=datetime.now(),
        symbol="AAPL",
        direction=1,
        strength=0.8,
        confidence=0.7,
        horizon="daily",
        reason="MA 金叉"
    )
    print(f"   Has opinion: {ao.has_opinion()}")
    print(f"   Position signal: {ao.get_position_signal():.2f}")

    # 测试被禁用的观点
    ao_disabled = AlphaOpinion(
        strategy_name="ma",
        timestamp=datetime.now(),
        symbol="AAPL",
        direction=1,
        strength=0.8,
        confidence=0.7,
        horizon="daily",
        is_disabled=True,
        disabled_reason="regime: RANGE"
    )
    print(f"   Disabled has opinion: {ao_disabled.has_opinion()}")

    # 测试 MetaDecision
    print("\n3. MetaDecision:")
    md = MetaDecision(
        timestamp=datetime.now(),
        symbol="AAPL",
        target_position=100,
        target_weight=0.2,
        active_strategies=["ma"],
        disabled_strategies={"breakout": "regime: RANGE"},
        decision_confidence=0.6,
        consensus_level="WEAK"
    )
    print(f"   Target position: {md.target_position}")
    print(f"   Active strategies: {md.active_strategies}")

    # 测试 RiskDecision
    print("\n4. RiskDecision:")
    rd = RiskDecision(
        timestamp=datetime.now(),
        symbol="AAPL",
        scaled_target_position=80,
        scale_factor=0.8,
        risk_action="SCALE_DOWN",
        risk_reason="高波动降低仓位"
    )
    print(f"   Scaled position: {rd.scaled_target_position}")
    print(f"   Is allowed: {rd.is_allowed()}")

    # 测试 ExecutionResult
    print("\n5. ExecutionResult:")
    er = ExecutionResult(
        timestamp=datetime.now(),
        symbol="AAPL",
        execution_mode="paper",
        target_position=80,
        actual_position=80,
        status="FILLED",
        fill_quantity=80,
        slippage=0.001
    )
    print(f"   Status: {er.status}")

    # 测试 AlphaGatingConfig
    print("\n6. AlphaGatingConfig:")
    config = DEFAULT_ALPHA_GATING["ma"]
    should_disable, reason = config.should_disable(ms)
    print(f"   Should disable MA in TREND: {should_disable}")
    scale = config.get_scale_factor(ms)
    print(f"   Scale factor in TREND: {scale}")

    print("\n✓ 所有 Schema 测试通过")
