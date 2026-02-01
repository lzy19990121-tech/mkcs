"""
SPL-7a-A: 运行态风险信号 Schema 定义

定义在线监控需要采集的风险指标，与 SPL-4/5/6 口径完全一致。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np


class MetricType(Enum):
    """指标类型"""
    ROLLING_RETURN = "rolling_return"           # 滚动收益
    DRAWDOWN = "drawdown"                       # 回撤
    DURATION = "duration"                       # 持续时长
    STABILITY_SCORE = "stability_score"         # 稳定性评分
    SPIKE_METRIC = "spike_metric"               # 尖刺指标
    REGIME_FEATURE = "regime_feature"           # 市场状态特征
    GATING_EVENT = "gating_event"               # Gating 事件
    ALLOCATOR_EVENT = "allocator_event"         # Allocator 事件


class RiskLevel(Enum):
    """风险等级"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RollingReturnMetrics:
    """滚动收益指标"""
    window_1d_return: float = 0.0
    window_5d_return: float = 0.0
    window_20d_return: float = 0.0
    window_60d_return: float = 0.0

    # 分位数收益
    percentile_5: float = 0.0      # P5 收益
    percentile_25: float = 0.0     # P25 收益
    percentile_75: float = 0.0     # P75 收益
    percentile_95: float = 0.0     # P95 收益

    def to_dict(self) -> Dict[str, float]:
        return {
            "window_1d_return": self.window_1d_return,
            "window_5d_return": self.window_5d_return,
            "window_20d_return": self.window_20d_return,
            "window_60d_return": self.window_60d_return,
            "percentile_5": self.percentile_5,
            "percentile_25": self.percentile_25,
            "percentile_75": self.percentile_75,
            "percentile_95": self.percentile_95
        }


@dataclass
class DrawdownMetrics:
    """回撤指标"""
    current_drawdown: float = 0.0          # 当前回撤
    max_drawdown: float = 0.0              # 最大回撤
    drawdown_duration: int = 0             # 回撤持续天数
    avg_drawdown: float = 0.0              # 平均回撤

    # 回撤统计
    drawdown_count: int = 0                # 回撤次数
    severe_drawdown_count: int = 0         # 严重回撤次数（>5%）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration": self.drawdown_duration,
            "avg_drawdown": self.avg_drawdown,
            "drawdown_count": self.drawdown_count,
            "severe_drawdown_count": self.severe_drawdown_count
        }


@dataclass
class SpikeMetrics:
    """尖刺指标"""
    max_single_loss: float = 0.0           # 最大单步亏损
    max_single_gain: float = 0.0           # 最大单步收益

    # 连续亏损统计
    max_consecutive_losses: int = 0        # 最大连续亏损次数
    current_consecutive_losses: int = 0    # 当前连续亏损次数

    # 损失聚类
    loss_clustering_score: float = 0.0     # 损失聚类评分
    recent_spike_count: int = 0            # 近期尖刺次数（20天内）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_single_loss": self.max_single_loss,
            "max_single_gain": self.max_single_gain,
            "max_consecutive_losses": self.max_consecutive_losses,
            "current_consecutive_losses": self.current_consecutive_losses,
            "loss_clustering_score": self.loss_clustering_score,
            "recent_spike_count": self.recent_spike_count
        }


@dataclass
class StabilityMetrics:
    """稳定性指标"""
    stability_score: float = 100.0         # 稳定性评分（0-100）

    # 波动性
    volatility_1d: float = 0.0             # 1日波动率
    volatility_5d: float = 0.0             # 5日波动率
    volatility_20d: float = 0.0            # 20日波动率

    # 收益标准差
    return_std_1d: float = 0.0
    return_std_5d: float = 0.0
    return_std_20d: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "stability_score": self.stability_score,
            "volatility_1d": self.volatility_1d,
            "volatility_5d": self.volatility_5d,
            "volatility_20d": self.volatility_20d,
            "return_std_1d": self.return_std_1d,
            "return_std_5d": self.return_std_5d,
            "return_std_20d": self.return_std_20d
        }


@dataclass
class RegimeFeatures:
    """市场状态特征（与 SPL-5a 一致）"""
    volatility_bucket: str = "med"        # low/med/high
    trend_strength: str = "weak"           # weak/strong
    liquidity_level: str = "high"          # low/high

    # 原始值
    realized_volatility: float = 0.0       # 实现波动率
    adx: float = 0.0                       # 平均趋向指数
    spread_cost: float = 0.0               # 价差成本代理

    def to_dict(self) -> Dict[str, Any]:
        return {
            "volatility_bucket": self.volatility_bucket,
            "trend_strength": self.trend_strength,
            "liquidity_level": self.liquidity_level,
            "realized_volatility": self.realized_volatility,
            "adx": self.adx,
            "spread_cost": self.spread_cost
        }


@dataclass
class GatingEvent:
    """Gating 事件"""
    event_id: str
    timestamp: datetime
    strategy_id: str

    action: str  # "ALLOW", "GATE", "REDUCE", "DISABLE"
    rule_id: Optional[str] = None
    threshold: Optional[float] = None
    current_value: Optional[float] = None

    # 上下文
    regime_features: Optional[RegimeFeatures] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "action": self.action,
            "rule_id": self.rule_id,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "regime_features": self.regime_features.to_dict() if self.regime_features else None,
            "reason": self.reason
        }


@dataclass
class AllocatorEvent:
    """Allocator 事件"""
    event_id: str
    timestamp: datetime

    allocator_type: str  # "rules", "optimizer_v2"
    action: str  # "rebalance", "cap_hit", "fallback"

    # 权重变化
    old_weights: Dict[str, float] = field(default_factory=dict)
    new_weights: Dict[str, float] = field(default_factory=dict)
    weight_changes: Dict[str, float] = field(default_factory=dict)

    # 触发原因
    reason: str = ""
    constraints_hit: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "allocator_type": self.allocator_type,
            "action": self.action,
            "old_weights": self.old_weights,
            "new_weights": self.new_weights,
            "weight_changes": self.weight_changes,
            "reason": self.reason,
            "constraints_hit": self.constraints_hit
        }


@dataclass
class RiskSignal:
    """风险信号（完整快照）

    在时间 t 的完整风险状态快照
    """
    timestamp: datetime
    strategy_id: str

    # 基础指标
    rolling_returns: RollingReturnMetrics
    drawdown: DrawdownMetrics
    spike: SpikeMetrics
    stability: StabilityMetrics
    regime: RegimeFeatures

    # 事件（当前时刻的事件列表）
    gating_events: List[GatingEvent] = field(default_factory=list)
    allocator_events: List[AllocatorEvent] = field(default_factory=list)

    # 元信息
    data_version: str = "1.0"
    source: str = "online_monitoring"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "rolling_returns": self.rolling_returns.to_dict(),
            "drawdown": self.drawdown.to_dict(),
            "spike": self.spike.to_dict(),
            "stability": self.stability.to_dict(),
            "regime": self.regime.to_dict(),
            "gating_events": [e.to_dict() for e in self.gating_events],
            "allocator_events": [e.to_dict() for e in self.allocator_events],
            "data_version": self.data_version,
            "source": self.source
        }


@dataclass
class PortfolioRiskSignal:
    """组合级风险信号"""
    timestamp: datetime

    # 组合指标（聚合）
    portfolio_return_1d: float = 0.0
    portfolio_return_5d: float = 0.0
    portfolio_return_20d: float = 0.0

    # 组合风险
    portfolio_var: float = 0.0              # 组合方差
    portfolio_cvar_95: float = 0.0           # 条件风险价值
    portfolio_max_dd: float = 0.0            # 最大回撤

    # 协同指标
    avg_correlation: float = 0.0             # 平均相关性
    tail_correlation: float = 0.0            # 尾部相关性
    co_crash_count: int = 0                  # Co-crash 次数

    # 策略级信号
    strategy_signals: Dict[str, RiskSignal] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio_return_1d": self.portfolio_return_1d,
            "portfolio_return_5d": self.portfolio_return_5d,
            "portfolio_return_20d": self.portfolio_return_20d,
            "portfolio_var": self.portfolio_var,
            "portfolio_cvar_95": self.portfolio_cvar_95,
            "portfolio_max_dd": self.portfolio_max_dd,
            "avg_correlation": self.avg_correlation,
            "tail_correlation": self.tail_correlation,
            "co_crash_count": self.co_crash_count,
            "strategy_signals": {
                k: v.to_dict() for k, v in self.strategy_signals.items()
            }
        }


# Schema 版本控制
CURRENT_SCHEMA_VERSION = "1.0"

# 向后兼容的 schema 映射
SCHEMA_COMPATIBILITY = {
    "1.0": ["1.0"],  # 当前版本兼容自身
    # 未来版本可以在这里添加向后兼容声明
}
