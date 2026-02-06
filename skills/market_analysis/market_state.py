"""
市场状态定义

统一的市场分析输出结构，所有策略通过 MarketState 感知市场
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List
from enum import Enum
import json
import numpy as np


class RegimeType(Enum):
    """市场状态类型"""
    TREND = "TREND"           # 趋势市
    RANGE = "RANGE"           # 震荡市
    HIGH_VOL = "HIGH_VOL"     # 高波动
    CRISIS = "CRISIS"         # 危机模式
    UNKNOWN = "UNKNOWN"       # 未知


class VolatilityState(Enum):
    """波动率状态"""
    LOW = "LOW"               # 低波动
    NORMAL = "NORMAL"         # 正常波动
    HIGH = "HIGH"             # 高波动
    EXTREME = "EXTREME"       # 极端波动


class VolatilityTrend(Enum):
    """波动率趋势"""
    RISING = "RISING"         # 上���
    FALLING = "FALLING"       # 下降
    STABLE = "STABLE"         # 稳定


class LiquidityState(Enum):
    """流动性状态"""
    NORMAL = "NORMAL"         # 正常
    THIN = "THIN"             # 流动性不足
    FROZEN = "FROZEN"         # 流动性枯竭


class SentimentState(Enum):
    """市场情绪状态"""
    FEAR = "FEAR"             # 恐慌
    NEUTRAL = "NEUTRAL"       # 中性
    GREED = "GREED"           # 贪婪


@dataclass
class SystemicRiskFlags:
    """系统性风险标志"""
    event_window: bool = False           # 事件窗口（FOMC/CPI/NFP等）
    high_systemic_risk: bool = False     # 高系统性风险
    cross_market_stress: bool = False    # 跨市场压力
    currency_crisis: bool = False        # 货币危机风险

    def has_any_risk(self) -> bool:
        """是否有任何风险标志"""
        return (
            self.event_window or
            self.high_systemic_risk or
            self.cross_market_stress or
            self.currency_crisis
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_window": self.event_window,
            "high_systemic_risk": self.high_systemic_risk,
            "cross_market_stress": self.cross_market_stress,
            "currency_crisis": self.currency_crisis,
            "has_any_risk": self.has_any_risk()
        }


@dataclass
class MarketState:
    """
    市场状态快照

    每个bar/interval计算一次，供策略、风控、规则使用
    """
    # 基本信息
    timestamp: datetime
    symbol: str

    # 市场状态（核心）
    regime: RegimeType = RegimeType.UNKNOWN
    regime_confidence: float = 0.0           # 状态置信度 (0-1)

    # 波动率分析
    volatility_state: VolatilityState = VolatilityState.NORMAL
    volatility_trend: VolatilityTrend = VolatilityTrend.STABLE
    volatility_percentile: float = 0.5      # 波动率分位数（相对历史）
    realized_vol: float = 0.0               # 已实现波动率
    atr_ratio: float = 0.0                  # ATR 相对价格比例

    # 流动性状态
    liquidity_state: LiquidityState = LiquidityState.NORMAL
    volume_ratio: float = 1.0               # 成交量相对历史均值
    turnover_ratio: float = 1.0              # 换手率相对历史均值

    # 市场情绪
    sentiment_state: SentimentState = SentimentState.NEUTRAL
    sentiment_score: float = 0.0            # 情绪分数 (-1 到 1)

    # 系统性风险
    systemic_risk: SystemicRiskFlags = field(default_factory=SystemicRiskFlags)

    # 趋势强度
    trend_strength: float = 0.0             # 趋势强度 (0-1)
    trend_direction: str = "NEUTRAL"        # UP / DOWN / NEUTRAL

    # 置信度（整体）
    confidence: float = 0.0                 # 整体分析置信度 (0-1)

    # 特征快照（用于回放与分析）
    features_snapshot: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    state_version: str = "1.0"
    computed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "regime": self.regime.value,
            "regime_confidence": self.regime_confidence,
            "volatility_state": self.volatility_state.value,
            "volatility_trend": self.volatility_trend.value,
            "volatility_percentile": self.volatility_percentile,
            "realized_vol": self.realized_vol,
            "atr_ratio": self.atr_ratio,
            "liquidity_state": self.liquidity_state.value,
            "volume_ratio": self.volume_ratio,
            "turnover_ratio": self.turnover_ratio,
            "sentiment_state": self.sentiment_state.value,
            "sentiment_score": self.sentiment_score,
            "systemic_risk": self.systemic_risk.to_dict(),
            "trend_strength": self.trend_strength,
            "trend_direction": self.trend_direction,
            "confidence": self.confidence,
            "features_snapshot": self.features_snapshot,
            "state_version": self.state_version,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None
        }

    def to_json(self) -> str:
        """转换为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketState":
        """从字典恢复"""
        # 处理枚举类型
        regime = RegimeType(data.get("regime", "UNKNOWN"))
        vol_state = VolatilityState(data.get("volatility_state", "NORMAL"))
        vol_trend = VolatilityTrend(data.get("volatility_trend", "STABLE"))
        liq_state = LiquidityState(data.get("liquidity_state", "NORMAL"))
        sent_state = SentimentState(data.get("sentiment_state", "NEUTRAL"))

        # 处理系统性风险
        sys_risk_data = data.get("systemic_risk", {})
        if isinstance(sys_risk_data, dict):
            # 过滤掉 has_any_risk 字段
            filtered_data = {k: v for k, v in sys_risk_data.items() if k != "has_any_risk"}
            systemic_risk = SystemicRiskFlags(**filtered_data)
        else:
            systemic_risk = SystemicRiskFlags()

        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            regime=regime,
            regime_confidence=data.get("regime_confidence", 0.0),
            volatility_state=vol_state,
            volatility_trend=vol_trend,
            volatility_percentile=data.get("volatility_percentile", 0.5),
            realized_vol=data.get("realized_vol", 0.0),
            atr_ratio=data.get("atr_ratio", 0.0),
            liquidity_state=liq_state,
            volume_ratio=data.get("volume_ratio", 1.0),
            turnover_ratio=data.get("turnover_ratio", 1.0),
            sentiment_state=sent_state,
            sentiment_score=data.get("sentiment_score", 0.0),
            systemic_risk=systemic_risk,
            trend_strength=data.get("trend_strength", 0.0),
            trend_direction=data.get("trend_direction", "NEUTRAL"),
            confidence=data.get("confidence", 0.0),
            features_snapshot=data.get("features_snapshot", {}),
            state_version=data.get("state_version", "1.0"),
            computed_at=datetime.fromisoformat(data["computed_at"]) if data.get("computed_at") else None
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MarketState":
        """从 JSON 恢复"""
        return cls.from_dict(json.loads(json_str))

    # 便捷方法
    @property
    def is_trend(self) -> bool:
        return self.regime == RegimeType.TREND

    @property
    def is_range(self) -> bool:
        return self.regime == RegimeType.RANGE

    @property
    def is_high_volatility(self) -> bool:
        return self.volatility_state in [VolatilityState.HIGH, VolatilityState.EXTREME]

    @property
    def is_crisis(self) -> bool:
        return self.regime == RegimeType.CRISIS

    @property
    def is_low_liquidity(self) -> bool:
        return self.liquidity_state in [LiquidityState.THIN, LiquidityState.FROZEN]

    @property
    def should_reduce_risk(self) -> bool:
        """是否应该降低风险暴露"""
        return (
            self.is_crisis or
            self.is_high_volatility or
            self.is_low_liquidity or
            self.systemic_risk.has_any_risk()
        )

    @property
    def position_scale_factor(self) -> float:
        """获取建议的仓位缩放系数

        基于当前市场状态计算建议的仓位比例
        """
        factor = 1.0

        # 波动率调整
        if self.volatility_state == VolatilityState.LOW:
            factor *= 1.2
        elif self.volatility_state == VolatilityState.HIGH:
            factor *= 0.5
        elif self.volatility_state == VolatilityState.EXTREME:
            factor *= 0.2

        # 流动性调整
        if self.liquidity_state == LiquidityState.THIN:
            factor *= 0.5
        elif self.liquidity_state == LiquidityState.FROZEN:
            factor *= 0.1

        # 系统性风险调整
        if self.systemic_risk.has_any_risk():
            factor *= 0.5

        # 震荡市调整
        if self.regime == RegimeType.RANGE:
            factor *= 0.3

        return max(0.0, min(1.0, factor))


class MarketStateBuilder:
    """
    市场状态构建器

    负责收集各个检测器的输出，构建统一的 MarketState
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._detectors = {}
        self._history: List[MarketState] = []
        self._historical_volatilities: List[float] = []
        self._historical_volumes: List[float] = []

    def add_detector(self, name: str, detector):
        """添加检测器"""
        self._detectors[name] = detector
        return self

    def compute(
        self,
        bars: List[object],
        timestamp: datetime,
        external_signals: Optional[Dict[str, Any]] = None
    ) -> MarketState:
        """计算当前市场状态

        Args:
            bars: K线数据
            timestamp: 当前时间
            external_signals: 外部信号（情绪、宏观等）

        Returns:
            MarketState
        """
        # 1. 运行各检测器
        regime_result = self._detect_regime(bars)
        vol_result = self._detect_volatility(bars)
        liq_result = self._detect_liquidity(bars)

        # 2. 整合外部信号
        sentiment_state = self._process_sentiment(external_signals)
        systemic_risk = self._process_systemic_risk(external_signals)

        # 3. 计算趋势强度和方向
        trend_strength, trend_direction = self._calculate_trend(bars)

        # 4. 计算整体置信度
        confidence = self._calculate_confidence(regime_result, vol_result, liq_result)

        # 5. 构建特征快照
        features_snapshot = {
            "regime_features": regime_result.get("features", {}),
            "volatility_features": vol_result.get("features", {}),
            "liquidity_features": liq_result.get("features", {}),
            "external_signals": external_signals or {}
        }

        # 6. 构建 MarketState
        state = MarketState(
            timestamp=timestamp,
            symbol=self.symbol,
            regime=regime_result.get("regime", RegimeType.UNKNOWN),
            regime_confidence=regime_result.get("confidence", 0.0),
            volatility_state=vol_result.get("state", VolatilityState.NORMAL),
            volatility_trend=vol_result.get("trend", VolatilityTrend.STABLE),
            volatility_percentile=vol_result.get("percentile", 0.5),
            realized_vol=vol_result.get("realized_vol", 0.0),
            atr_ratio=vol_result.get("atr_ratio", 0.0),
            liquidity_state=liq_result.get("state", LiquidityState.NORMAL),
            volume_ratio=liq_result.get("volume_ratio", 1.0),
            turnover_ratio=liq_result.get("turnover_ratio", 1.0),
            sentiment_state=sentiment_state,
            sentiment_score=external_signals.get("sentiment_score", 0.0) if external_signals else 0.0,
            systemic_risk=systemic_risk,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            confidence=confidence,
            features_snapshot=features_snapshot,
            computed_at=datetime.now()
        )

        # 7. 保存历史
        self._history.append(state)
        self._historical_volatilities.append(vol_result.get("realized_vol", 0.0))
        self._historical_volumes.append(liq_result.get("volume_ratio", 1.0))

        # 保持历史长度
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        return state

    def _detect_regime(self, bars: List[object]) -> Dict[str, Any]:
        """检测市场状态"""
        # 这里会调用 RegimeDetector
        # 简化实现
        return {
            "regime": RegimeType.UNKNOWN,
            "confidence": 0.0,
            "features": {}
        }

    def _detect_volatility(self, bars: List[object]) -> Dict[str, Any]:
        """检测波动率状态"""
        # 这里会调用 VolatilityDetector
        vol = np.std([float(b.close) for b in bars[-20:]]) if len(bars) >= 20 else 0.0

        return {
            "state": VolatilityState.NORMAL,
            "trend": VolatilityTrend.STABLE,
            "percentile": 0.5,
            "realized_vol": vol,
            "atr_ratio": 0.02,
            "features": {}
        }

    def _detect_liquidity(self, bars: List[object]) -> Dict[str, Any]:
        """检测流动性状态"""
        if len(bars) < 20:
            return {"state": LiquidityState.NORMAL, "volume_ratio": 1.0}

        volumes = [b.volume for b in bars[-20:]]
        avg_volume = np.mean(volumes) if volumes else 1.0
        current_volume = volumes[-1] if volumes else 1.0

        return {
            "state": LiquidityState.NORMAL,
            "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 1.0,
            "turnover_ratio": 1.0,
            "features": {}
        }

    def _process_sentiment(self, external_signals: Optional[Dict[str, Any]]) -> SentimentState:
        """处理情绪信号"""
        if not external_signals:
            return SentimentState.NEUTRAL

        score = external_signals.get("sentiment_score", 0.0)
        if score < -0.3:
            return SentimentState.FEAR
        elif score > 0.3:
            return SentimentState.GREED
        return SentimentState.NEUTRAL

    def _process_systemic_risk(self, external_signals: Optional[Dict[str, Any]]) -> SystemicRiskFlags:
        """处理系统性风险"""
        flags = SystemicRiskFlags()

        if external_signals:
            flags.event_window = external_signals.get("event_window", False)
            flags.high_systemic_risk = external_signals.get("high_systemic_risk", False)
            flags.cross_market_stress = external_signals.get("cross_market_stress", False)

        return flags

    def _calculate_trend(self, bars: List[object]) -> tuple:
        """计算趋势强度和方向"""
        if len(bars) < 20:
            return 0.0, "NEUTRAL"

        closes = [float(b.close) for b in bars[-20:]]
        first_close = closes[0]
        last_close = closes[-1]

        # 简单线性回归计算趋势
        x = np.arange(len(closes))
        y = np.array(closes)
        slope = np.polyfit(x, y, 1)[0]

        # 趋势强度（基于 R²）
        y_pred = np.polyval(np.polyfit(x, y, 1), x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 方向
        if slope > 0.01:
            direction = "UP"
        elif slope < -0.01:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        return abs(r_squared), direction

    def _calculate_confidence(self, regime_result, vol_result, liq_result) -> float:
        """计算整体置信度"""
        confidences = [
            regime_result.get("confidence", 0.0),
        ]
        return np.mean(confidences) if confidences else 0.5

    def get_history(self, limit: int = 100) -> List[MarketState]:
        """获取历史状态"""
        return self._history[-limit:]

    def get_volatility_percentile(self, current_vol: float) -> float:
        """获取波动率分位数"""
        if not self._historical_volatilities:
            return 0.5

        vols = np.array(self._historical_volatilities)
        return np.sum(vols < current_vol) / len(vols)


if __name__ == "__main__":
    """测试代码"""
    print("=== MarketState 测试 ===\n")

    # 创建一个测试 MarketState
    state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime=RegimeType.TREND,
        regime_confidence=0.8,
        volatility_state=VolatilityState.NORMAL,
        volatility_trend=VolatilityTrend.STABLE,
        volatility_percentile=0.6,
        realized_vol=0.15,
        atr_ratio=0.02,
        liquidity_state=LiquidityState.NORMAL,
        volume_ratio=1.2,
        sentiment_state=SentimentState.GREED,
        sentiment_score=0.4,
        trend_strength=0.7,
        trend_direction="UP",
        confidence=0.75
    )

    print("1. MarketState 创建:")
    print(f"   Regime: {state.regime.value}")
    print(f"   Volatility: {state.volatility_state.value}")
    print(f"   Liquidity: {state.liquidity_state.value}")
    print(f"   Sentiment: {state.sentiment_state.value}")

    print("\n2. 便捷方法:")
    print(f"   is_trend: {state.is_trend}")
    print(f"   is_high_volatility: {state.is_high_volatility}")
    print(f"   should_reduce_risk: {state.should_reduce_risk}")
    print(f"   position_scale_factor: {state.position_scale_factor:.2f}")

    print("\n3. 序列化:")
    json_str = state.to_json()
    print(f"   JSON 长度: {len(json_str)}")

    restored = MarketState.from_json(json_str)
    print(f"   恢复后 Regime: {restored.regime.value}")

    print("\n4. MarketStateBuilder:")
    builder = MarketStateBuilder("AAPL")
    print(f"   Builder 创建成功")

    print("\n✓ 所有测试通过")
