"""
SPL-7a-B: 趋势检测器

检测风险指标的趋势变化，实现早期预警。
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from collections import deque
import numpy as np

from analysis.online.risk_signal_schema import RiskSignal


class TrendDirection(Enum):
    """趋势方向"""
    INCREASING = "increasing"  # 上升趋势（风险增大）
    DECREASING = "decreasing"  # 下降趋势（风险降低）
    STABLE = "stable"          # 稳定
    UNKNOWN = "unknown"        # 未知


class TrendMagnitude(Enum):
    """趋势强度"""
    NONE = "none"        # 无显著趋势
    MILD = "mild"        # 轻微趋势
    MODERATE = "moderate"  # 中等趋势
    SEVERE = "severe"    # 严重趋势


@dataclass
class TrendIndicator:
    """趋势指标"""
    metric_name: str
    current_value: float
    direction: TrendDirection
    magnitude: TrendMagnitude

    # 统计信息
    mean: float
    std: float
    min_value: float
    max_value: float

    # 趋势统计
    slope: float  # 线性回归斜率
    r_squared: float  # 拟合优度
    growth_rate: float  # 增长率（相对于均值）

    # 时间信息
    window_start: datetime
    window_end: datetime
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "direction": self.direction.value,
            "magnitude": self.magnitude.value,
            "mean": self.mean,
            "std": self.std,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "growth_rate": self.growth_rate,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "sample_count": self.sample_count
        }


@dataclass
class TrendAlert:
    """趋势告警"""
    timestamp: datetime
    strategy_id: str
    metric_name: str
    alert_type: str  # "TREND_UP", "TREND_DOWN", "SPIKE", "DROPOUT"

    # 趋势信息
    direction: TrendDirection
    magnitude: TrendMagnitude
    current_value: float
    threshold: float

    # 描述
    message: str
    severity: str  # "info", "warning", "critical"

    # 建议
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "metric_name": self.metric_name,
            "alert_type": self.alert_type,
            "direction": self.direction.value,
            "magnitude": self.magnitude.value,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "message": self.message,
            "severity": self.severity,
            "recommendation": self.recommendation
        }


@dataclass
class TrendDetectorConfig:
    """趋势检测器配置"""
    # 检测窗口
    short_window_seconds: int = 3600     # 1 小时
    medium_window_seconds: int = 86400   # 1 天
    long_window_seconds: int = 604800    # 1 周

    # 趋势阈值
    mild_growth_threshold: float = 0.10   # 10% 增长
    moderate_growth_threshold: float = 0.25  # 25% 增长
    severe_growth_threshold: float = 0.50    # 50% 增长

    # 显著性检验
    min_sample_size: int = 10             # 最小样本数
    significance_level: float = 0.05      # 显著性水平

    # 告警配置
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 1800   # 30 分钟告警冷却


class TrendDetector:
    """趋势检测器

    检测风险指标的趋势变化。
    """

    def __init__(
        self,
        strategy_id: str,
        config: Optional[TrendDetectorConfig] = None
    ):
        """初始化检测器

        Args:
            strategy_id: 策略 ID
            config: 检测器配置
        """
        self.strategy_id = strategy_id
        self.config = config or TrendDetectorConfig()

        # 趋势数据缓存（多个窗口）
        self.trend_buffers: Dict[str, Dict[str, deque]] = {
            "envelope_usage": {
                "short": deque(maxlen=60),
                "medium": deque(maxlen=240),
                "long": deque(maxlen=1000)
            },
            "spike_metrics": {
                "short": deque(maxlen=60),
                "medium": deque(maxlen=240),
                "long": deque(maxlen=1000)
            },
            "volatility": {
                "short": deque(maxlen=60),
                "medium": deque(maxlen=240),
                "long": deque(maxlen=1000)
            },
            "gating_frequency": {
                "short": deque(maxlen=60),
                "medium": deque(maxlen=240),
                "long": deque(maxlen=1000)
            }
        }

        # 最近告警（防重复）
        self.recent_alerts: deque = deque(maxlen=100)
        self.last_alert_time: Dict[str, datetime] = {}

    def update(self, signal: RiskSignal) -> List[TrendAlert]:
        """更新检测器并生成告警

        Args:
            signal: 风险信号

        Returns:
            告警列表
        """
        # 更新趋势数据
        self._update_trend_buffers(signal)

        # 检测趋势
        alerts = []

        if self.config.enable_alerts:
            # 检测 envelope 使用率趋势
            envelope_alerts = self._detect_envelope_trend()
            alerts.extend(envelope_alerts)

            # 检测 spike 趋势
            spike_alerts = self._detect_spike_trend()
            alerts.extend(spike_alerts)

            # 检测波动率趋势
            volatility_alerts = self._detect_volatility_trend()
            alerts.extend(volatility_alerts)

            # 检测 gating 频率趋势
            gating_alerts = self._detect_gating_frequency_trend()
            alerts.extend(gating_alerts)

        # 记录告警
        for alert in alerts:
            self.recent_alerts.append(alert)

        return alerts

    def _update_trend_buffers(self, signal: RiskSignal):
        """更新趋势缓冲区"""
        timestamp = signal.timestamp

        # Envelope 使用率
        envelope_limit = 0.10
        envelope_usage = min(1.0, signal.drawdown.current_drawdown / envelope_limit)
        for window in ["short", "medium", "long"]:
            self.trend_buffers["envelope_usage"][window].append((timestamp, envelope_usage))

        # Spike 指标
        spike_value = signal.spike.recent_spike_count
        for window in ["short", "medium", "long"]:
            self.trend_buffers["spike_metrics"][window].append((timestamp, spike_value))

        # 波动率
        volatility = signal.stability.volatility_20d
        for window in ["short", "medium", "long"]:
            self.trend_buffers["volatility"][window].append((timestamp, volatility))

        # Gating 频率
        gating_count = len(signal.gating_events)
        for window in ["short", "medium", "long"]:
            self.trend_buffers["gating_frequency"][window].append((timestamp, gating_count))

    def _detect_envelope_trend(self) -> List[TrendAlert]:
        """检测 envelope 使用率趋势"""
        alerts = []
        metric_name = "envelope_usage"

        # 使用中等窗口
        buffer = self.trend_buffers[metric_name]["medium"]
        if len(buffer) < self.config.min_sample_size:
            return alerts

        # 提取数据
        timestamps, values = zip(*buffer)
        values = np.array(values)

        # 计算趋势
        indicator = self._calculate_trend(metric_name, timestamps, values)

        # 生成告警
        if (indicator.direction == TrendDirection.INCREASING and
            indicator.magnitude in [TrendMagnitude.MODERATE, TrendMagnitude.SEVERE]):

            # 检查冷却时间
            if self._check_alert_cooldown(metric_name):
                alert = TrendAlert(
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metric_name=metric_name,
                    alert_type="TREND_UP",
                    direction=indicator.direction,
                    magnitude=indicator.magnitude,
                    current_value=indicator.current_value,
                    threshold=indicator.mean + indicator.std,
                    message=f"Envelope 使用率呈上升趋势（当前 {indicator.current_value:.1%}）",
                    severity="warning" if indicator.magnitude == TrendMagnitude.MODERATE else "critical",
                    recommendation="建议检查 gating 配置，准备应对可能的回撤"
                )
                alerts.append(alert)
                self.last_alert_time[metric_name] = datetime.now()

        return alerts

    def _detect_spike_trend(self) -> List[TrendAlert]:
        """检测 spike 趋势"""
        alerts = []
        metric_name = "spike_metrics"

        buffer = self.trend_buffers[metric_name]["medium"]
        if len(buffer) < self.config.min_sample_size:
            return alerts

        timestamps, values = zip(*buffer)
        values = np.array(values)

        indicator = self._calculate_trend(metric_name, timestamps, values)

        # Spike 激增告警
        if (indicator.direction == TrendDirection.INCREASING and
            indicator.magnitude == TrendMagnitude.SEVERE):

            if self._check_alert_cooldown(metric_name):
                alert = TrendAlert(
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metric_name=metric_name,
                    alert_type="SPIKE_SURGE",
                    direction=indicator.direction,
                    magnitude=indicator.magnitude,
                    current_value=int(indicator.current_value),
                    threshold=int(indicator.mean + indicator.std),
                    message=f"Spike 指标激增（{int(indicator.current_value)} 次，增长 {indicator.growth_rate:.1%}）",
                    severity="critical",
                    recommendation="考虑启用 gating 或降低仓位"
                )
                alerts.append(alert)
                self.last_alert_time[metric_name] = datetime.now()

        return alerts

    def _detect_volatility_trend(self) -> List[TrendAlert]:
        """检测波动率趋势"""
        alerts = []
        metric_name = "volatility"

        buffer = self.trend_buffers[metric_name]["medium"]
        if len(buffer) < self.config.min_sample_size:
            return alerts

        timestamps, values = zip(*buffer)
        values = np.array(values)

        indicator = self._calculate_trend(metric_name, timestamps, values)

        # 波动率上升告警
        if (indicator.direction == TrendDirection.INCREASING and
            indicator.magnitude in [TrendMagnitude.MODERATE, TrendMagnitude.SEVERE]):

            if self._check_alert_cooldown(metric_name):
                alert = TrendAlert(
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metric_name=metric_name,
                    alert_type="VOLATILITY_UP",
                    direction=indicator.direction,
                    magnitude=indicator.magnitude,
                    current_value=indicator.current_value,
                    threshold=0.03,  # 3%
                    message=f"波动率呈上升趋势（当前 {indicator.current_value:.2%}）",
                    severity="warning",
                    recommendation="监控市场状态，考虑收紧风控"
                )
                alerts.append(alert)
                self.last_alert_time[metric_name] = datetime.now()

        return alerts

    def _detect_gating_frequency_trend(self) -> List[TrendAlert]:
        """检测 gating 频率趋势"""
        alerts = []
        metric_name = "gating_frequency"

        buffer = self.trend_buffers[metric_name]["medium"]
        if len(buffer) < self.config.min_sample_size:
            return alerts

        timestamps, values = zip(*buffer)
        values = np.array(values)

        indicator = self._calculate_trend(metric_name, timestamps, values)

        # Gating 频率异常告警
        if (indicator.direction == TrendDirection.INCREASING and
            indicator.magnitude in [TrendMagnitude.MODERATE, TrendMagnitude.SEVERE]):

            if self._check_alert_cooldown(metric_name):
                alert = TrendAlert(
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metric_name=metric_name,
                    alert_type="GATING_FREQUENCY_UP",
                    direction=indicator.direction,
                    magnitude=indicator.magnitude,
                    current_value=indicator.current_value,
                    threshold=5,  # 5 次
                    message=f"Gating 触发频率上升（近期 {int(indicator.current_value)} 次）",
                    severity="warning",
                    recommendation="检查是否需要调整 gating 阈值或市场状态变化"
                )
                alerts.append(alert)
                self.last_alert_time[metric_name] = datetime.now()

        return alerts

    def _calculate_trend(
        self,
        metric_name: str,
        timestamps: Tuple,
        values: np.ndarray
    ) -> TrendIndicator:
        """计算趋势指标

        Args:
            metric_name: 指标名称
            timestamps: 时间戳序列
            values: 值序列

        Returns:
            TrendIndicator
        """
        # 基本统计
        mean = np.mean(values)
        std = np.std(values)
        min_value = np.min(values)
        max_value = np.max(values)

        # 线性回归计算趋势
        if len(values) >= 2:
            x = np.arange(len(values))
            coefficients = np.polyfit(x, values, 1)
            slope = coefficients[0]

            # 计算 R²
            y_pred = np.polyval(coefficients, x)
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - mean) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            slope = 0.0
            r_squared = 0.0

        # 增长率
        if mean > 0:
            growth_rate = (values[-1] - mean) / mean
        else:
            growth_rate = 0.0

        # 判定方向
        if slope > 0.001:
            direction = TrendDirection.INCREASING
        elif slope < -0.001:
            direction = TrendDirection.DECREASING
        else:
            direction = TrendDirection.STABLE

        # 判定强度
        abs_growth = abs(growth_rate)
        if abs_growth < self.config.mild_growth_threshold:
            magnitude = TrendMagnitude.NONE
        elif abs_growth < self.config.moderate_growth_threshold:
            magnitude = TrendMagnitude.MILD
        elif abs_growth < self.config.severe_growth_threshold:
            magnitude = TrendMagnitude.MODERATE
        else:
            magnitude = TrendMagnitude.SEVERE

        return TrendIndicator(
            metric_name=metric_name,
            current_value=values[-1],
            direction=direction,
            magnitude=magnitude,
            mean=mean,
            std=std,
            min_value=min_value,
            max_value=max_value,
            slope=slope,
            r_squared=r_squared,
            growth_rate=growth_rate,
            window_start=timestamps[0],
            window_end=timestamps[-1],
            sample_count=len(values)
        )

    def _check_alert_cooldown(self, metric_name: str) -> bool:
        """检查告警冷却时间

        Args:
            metric_name: 指标名称

        Returns:
            True（可以告警）或 False（冷却中）
        """
        if metric_name not in self.last_alert_time:
            return True

        elapsed = (datetime.now() - self.last_alert_time[metric_name]).total_seconds()
        return elapsed >= self.config.alert_cooldown_seconds

    def get_all_trends(self) -> Dict[str, TrendIndicator]:
        """获取所有指标的趋势

        Returns:
            指标名称 -> TrendIndicator
        """
        trends = {}

        for metric_name in ["envelope_usage", "spike_metrics", "volatility", "gating_frequency"]:
            buffer = self.trend_buffers[metric_name]["medium"]
            if len(buffer) >= self.config.min_sample_size:
                timestamps, values = zip(*buffer)
                indicator = self._calculate_trend(metric_name, timestamps, np.array(values))
                trends[metric_name] = indicator

        return trends
