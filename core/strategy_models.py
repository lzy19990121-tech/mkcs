"""
策略增强模型

定义策略输出的统一结构和状态管理接口
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal, Optional, Dict, Any, List
from enum import Enum
import json
from abc import ABC, abstractmethod


class RegimeType(Enum):
    """市场状态类型"""
    TREND = "TREND"           # 趋势市
    RANGE = "RANGE"           # 震荡市
    HIGH_VOL = "HIGH_VOL"     # 高波动
    LOW_VOL = "LOW_VOL"       # 低波动
    UNKNOWN = "UNKNOWN"       # 未知


@dataclass
class RiskHints:
    """风险提示信息

    策略向风控层传递的建议性风控参数
    """
    stop_loss: Optional[Decimal] = None      # 止损价
    take_profit: Optional[Decimal] = None    # 止盈价
    trailing_stop: Optional[Decimal] = None  # 移动止损（距离当前价的百分比）
    max_holding_days: Optional[int] = None   # 最大持有天数
    position_limit: Optional[float] = None   # 仓位限制（占资本的百分比）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stop_loss": float(self.stop_loss) if self.stop_loss else None,
            "take_profit": float(self.take_profit) if self.take_profit else None,
            "trailing_stop": float(self.trailing_stop) if self.trailing_stop else None,
            "max_holding_days": self.max_holding_days,
            "position_limit": self.position_limit
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskHints":
        """从字典创建"""
        return cls(
            stop_loss=Decimal(str(data["stop_loss"])) if data.get("stop_loss") else None,
            take_profit=Decimal(str(data["take_profit"])) if data.get("take_profit") else None,
            trailing_stop=Decimal(str(data["trailing_stop"])) if data.get("trailing_stop") else None,
            max_holding_days=data.get("max_holding_days"),
            position_limit=data.get("position_limit")
        )


@dataclass
class StrategySignal:
    """统一策略信号输出

    所有策略必须返回此结构，包含完整的交易决策信息

    Attributes:
        symbol: 标的代码
        timestamp: 信号时间
        target_position: 目标持仓数量（正数做多，负数做空，0平仓）
        target_weight: 目标权重（-1到1，表示从满空到满多的连续仓位）
        confidence: 信号置信度 (0-1)
        reason: 信号原因描述
        regime: 当前市场状态
        risk_hints: 风险提示
        raw_action: 原始动作（BUY/SELL/HOLD），用于日志记录
        metadata: 策略特定的额外信息
    """
    symbol: str
    timestamp: datetime
    target_position: float          # 连续仓位，可以是小数表示部分仓位
    target_weight: float = 0.0      # -1.0 到 1.0
    confidence: float = 0.5
    reason: str = ""
    regime: RegimeType = RegimeType.UNKNOWN
    risk_hints: Optional[RiskHints] = None
    raw_action: Optional[Literal['BUY', 'SELL', 'HOLD']] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证数据有效性"""
        if not -1.0 <= self.target_weight <= 1.0:
            raise ValueError(f"target_weight 必须在 [-1, 1] 范围内: {self.target_weight}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence 必须在 [0, 1] 范围内: {self.confidence}")

        # 根据 target_position 推断 raw_action
        if self.raw_action is None:
            if self.target_position > 0:
                self.raw_action = 'BUY'
            elif self.target_position < 0:
                self.raw_action = 'SELL'
            else:
                self.raw_action = 'HOLD'

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于序列化和日志记录"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "target_position": self.target_position,
            "target_weight": self.target_weight,
            "confidence": self.confidence,
            "reason": self.reason,
            "regime": self.regime.value if isinstance(self.regime, RegimeType) else self.regime,
            "risk_hints": self.risk_hints.to_dict() if self.risk_hints else None,
            "raw_action": self.raw_action,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategySignal":
        """从字典恢复信号"""
        # 处理 regime
        regime = data.get("regime", "UNKNOWN")
        if isinstance(regime, str):
            try:
                regime = RegimeType(regime)
            except ValueError:
                regime = RegimeType.UNKNOWN

        # 处理 risk_hints
        risk_hints = None
        if data.get("risk_hints"):
            risk_hints = RiskHints.from_dict(data["risk_hints"])

        return cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            target_position=data["target_position"],
            target_weight=data.get("target_weight", 0.0),
            confidence=data.get("confidence", 0.5),
            reason=data.get("reason", ""),
            regime=regime,
            risk_hints=risk_hints,
            raw_action=data.get("raw_action"),
            metadata=data.get("metadata", {})
        )

    @property
    def is_long(self) -> bool:
        """是否做多信号"""
        return self.target_position > 0

    @property
    def is_short(self) -> bool:
        """是否做空信号"""
        return self.target_position < 0

    @property
    def is_close(self) -> bool:
        """是否平仓信号"""
        return self.target_position == 0

    @property
    def signal_strength(self) -> float:
        """信号强度（考虑置信度和仓位权重）"""
        return abs(self.target_weight) * self.confidence


@dataclass
class StrategyState:
    """策略状态基类

    用于策略状态的序列化和恢复，确保回放一致性
    """
    state_version: str = "1.0"          # 状态版本，用于兼容性检查
    last_update: datetime = None        # 最后更新时间
    internal_state: Dict[str, Any] = field(default_factory=dict)  # 策略内部状态

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "state_version": self.state_version,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "internal_state": self.internal_state
        }

    def to_json(self) -> str:
        """转换为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyState":
        """从字典恢复"""
        return cls(
            state_version=data.get("state_version", "1.0"),
            last_update=datetime.fromisoformat(data["last_update"]) if data.get("last_update") else None,
            internal_state=data.get("internal_state", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> "StrategyState":
        """从 JSON 恢复"""
        return cls.from_dict(json.loads(json_str))

    def update(self, key: str, value: Any):
        """更新状态"""
        self.internal_state[key] = value
        self.last_update = datetime.now()

    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        return self.internal_state.get(key, default)


class StatefulStrategy(ABC):
    """有状态策略接口

    扩展自 Strategy，增加状态管理能力
    """

    @abstractmethod
    def get_state(self) -> StrategyState:
        """获取当前策略状态

        Returns:
            当前策略状态的可序列化表示
        """
        pass

    @abstractmethod
    def set_state(self, state: StrategyState) -> None:
        """恢复策略状态

        Args:
            state: 之前保存的策略状态
        """
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """重置策略状态到初始值"""
        pass


@dataclass
class RegimeInfo:
    """市场状态信息"""
    regime: RegimeType
    confidence: float = 0.0            # 状态判定的置信度
    trend_strength: float = 0.0        # 趋势强度 (0-1)
    volatility_level: float = 0.0      # 波动率水平 (相对值)
    adx: Optional[float] = None        # 平均趋向指数
    rsi: Optional[float] = None        # 相对强弱指标
    atr: Optional[float] = None        # 平均真实波幅

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "trend_strength": self.trend_strength,
            "volatility_level": self.volatility_level,
            "adx": self.adx,
            "rsi": self.rsi,
            "atr": float(self.atr) if self.atr else None
        }


if __name__ == "__main__":
    """测试代码"""
    print("=== Strategy Models 测试 ===\n")

    # 测试 RiskHints
    print("1. RiskHints:")
    hints = RiskHints(
        stop_loss=Decimal("145.0"),
        take_profit=Decimal("160.0"),
        trailing_stop=Decimal("0.02"),
        max_holding_days=10,
        position_limit=0.2
    )
    print(f"   {hints.to_dict()}")

    # 测试 StrategySignal
    print("\n2. StrategySignal:")
    signal = StrategySignal(
        symbol="AAPL",
        timestamp=datetime.now(),
        target_position=100.0,
        target_weight=0.5,
        confidence=0.85,
        reason="MA金叉",
        regime=RegimeType.TREND,
        risk_hints=hints
    )
    print(f"   信号: {signal.raw_action}, 目标仓位: {signal.target_position}")
    print(f"   信号强度: {signal.signal_strength:.2f}")
    print(f"   JSON: {signal.to_json()[:100]}...")

    # 测试序列化和反序列化
    print("\n3. 序列化/反序列化:")
    signal_dict = signal.to_dict()
    restored = StrategySignal.from_dict(signal_dict)
    print(f"   恢复后信号: {restored.raw_action}, 置信度: {restored.confidence}")

    # 测试 StrategyState
    print("\n4. StrategyState:")
    state = StrategyState(
        state_version="1.0",
        last_update=datetime.now(),
        internal_state={
            "fast_ma": 150.5,
            "slow_ma": 148.2,
            "prev_fast_ma": 149.0,
            "prev_slow_ma": 149.5,
            "position_opened": datetime.now().isoformat()
        }
    )
    print(f"   状态: {state.internal_state}")
    state_json = state.to_json()
    restored_state = StrategyState.from_json(state_json)
    print(f"   恢复后 fast_ma: {restored_state.get('fast_ma')}")

    # 测试 RegimeInfo
    print("\n5. RegimeInfo:")
    regime_info = RegimeInfo(
        regime=RegimeType.RANGE,
        confidence=0.75,
        trend_strength=0.2,
        volatility_level=0.4,
        adx=18.5,
        rsi=52.0,
        atr=2.5
    )
    print(f"   市场状态: {regime_info.regime.value}, 置信度: {regime_info.confidence}")

    print("\n✓ 所有测试通过")
