"""
SPL-7a-C: 在线告警系统

实现运行态风险告警，不等同于 gating（早期预警）。
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.online.risk_signal_schema import RiskSignal, PortfolioRiskSignal
from analysis.online.risk_state_machine import RiskState, StateTransitionEvent
from analysis.online.trend_detector import TrendAlert


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"           # 信息：风险趋势值得关注
    WARNING = "warning"     # 警告：风险上升，需要准备
    CRITICAL = "critical"   # 严重：风险事件已发生或即将发生


class AlertChannel(Enum):
    """告警渠道"""
    LOG = "log"             # 日志
    WEBHOOK = "webhook"     # Webhook
    SLACK = "slack"         # Slack
    EMAIL = "email"         # 邮件


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    description: str

    # 触发条件
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "trend_up", "trend_down"
    threshold: float
    severity: AlertSeverity

    # 告警配置
    enabled: bool = True
    cooldown_seconds: int = 3600  # 告警冷却时间（1 小时）

    # 目标渠道
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "cooldown_seconds": self.cooldown_seconds,
            "channels": [c.value for c in self.channels]
        }


@dataclass
class Alert:
    """告警"""
    alert_id: str
    timestamp: datetime
    rule_id: str
    strategy_id: str

    # 告警内容
    severity: AlertSeverity
    title: str
    message: str

    # 触发信息
    metric_name: str
    current_value: float
    threshold: float

    # 上下文
    context: Dict[str, Any] = field(default_factory=dict)

    # 发送状态
    sent_channels: List[str] = field(default_factory=list)
    failed_channels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "rule_id": self.rule_id,
            "strategy_id": self.strategy_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "context": self.context,
            "sent_channels": self.sent_channels,
            "failed_channels": self.failed_channels
        }


class AlertRuleEngine:
    """告警规则引擎

    根据风险信号和状态转换生成告警。
    """

    def __init__(self, config_path: Optional[str] = None):
        """初始化规则引擎

        Args:
            config_path: 告警规则配置文件路径
        """
        self.rules: Dict[str, AlertRule] = {}
        self.last_alert_time: Dict[str, datetime] = {}

        # 加载默认规则
        self._load_default_rules()

    def _load_default_rules(self):
        """加载默认告警规则"""
        # Rule 1: 接近 envelope（WARNING）
        self.rules["envelope_approach"] = AlertRule(
            rule_id="envelope_approach",
            name="接近 Envelope",
            description="回撤接近 envelope 上限（70%）",
            metric_name="envelope_usage",
            condition="greater_than",
            threshold=0.7,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK]
        )

        # Rule 2: Envelope 严重（CRITICAL）
        self.rules["envelope_critical"] = AlertRule(
            rule_id="envelope_critical",
            name="Envelope 严重",
            description="回撤接近 envelope 硬上限（90%）",
            metric_name="envelope_usage",
            condition="greater_than",
            threshold=0.9,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.WEBHOOK]
        )

        # Rule 3: Spike 激增（WARNING）
        self.rules["spike_surge"] = AlertRule(
            rule_id="spike_surge",
            name="Spike 激增",
            description="近期 spike 次数异常增加",
            metric_name="spike_count",
            condition="greater_than",
            threshold=5,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK]
        )

        # Rule 4: Gating 频率异常（WARNING）
        self.rules["gating_frequency_high"] = AlertRule(
            rule_id="gating_frequency_high",
            name="Gating 频率异常",
            description="Gating 触发频率过高（>20%）",
            metric_name="gating_frequency",
            condition="greater_than",
            threshold=0.2,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK]
        )

        # Rule 5: 波动率上升（INFO）
        self.rules["volatility_rising"] = AlertRule(
            rule_id="volatility_rising",
            name="波动率上升",
            description="波动率呈上升趋势",
            metric_name="volatility_trend",
            condition="trend_up",
            threshold=0.0,
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.LOG]
        )

        # Rule 6: 状态转换到 CRITICAL（CRITICAL）
        self.rules["state_critical"] = AlertRule(
            rule_id="state_critical",
            name="状态转为严重",
            description="风险状态转换为 CRITICAL",
            metric_name="risk_state",
            condition="equals",
            threshold=float(RiskState.CRITICAL.value),
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.EMAIL]
        )

        # Rule 7: Allocator cap 命中率异常（WARNING）
        self.rules["allocator_cap_hit"] = AlertRule(
            rule_id="allocator_cap_hit",
            name="Allocator Cap 命中率异常",
            description="Allocator cap 命中频率过高",
            metric_name="cap_hit_rate",
            condition="greater_than",
            threshold=0.3,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.SLACK]
        )

    def evaluate(
        self,
        signal: RiskSignal,
        state: RiskState,
        trends: Dict[str, Any],
        state_transition: Optional[StateTransitionEvent] = None
    ) -> List[Alert]:
        """评估告警规则

        Args:
            signal: 风险信号
            state: 当前风险状态
            trends: 趋势数据
            state_transition: 状态转换事件（可选）

        Returns:
            告警列表
        """
        alerts = []

        # 准备指标数据
        metrics = self._extract_metrics(signal, state, trends, state_transition)

        # 评估所有规则
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue

            # 检查冷却时间
            if not self._check_cooldown(rule_id):
                continue

            # 评估规则
            alert = self._evaluate_rule(rule, metrics, signal.strategy_id)
            if alert:
                alerts.append(alert)
                self.last_alert_time[rule_id] = datetime.now()

        return alerts

    def _extract_metrics(
        self,
        signal: RiskSignal,
        state: RiskState,
        trends: Dict[str, Any],
        state_transition: Optional[StateTransitionEvent]
    ) -> Dict[str, float]:
        """提取指标数据"""
        metrics = {}

        # Envelope 使用率
        envelope_limit = 0.10
        metrics["envelope_usage"] = min(1.0, signal.drawdown.current_drawdown / envelope_limit)

        # Spike 计数
        metrics["spike_count"] = float(signal.spike.recent_spike_count)

        # Gating 频率
        recent_gating_count = len(signal.gating_events)
        metrics["gating_frequency"] = min(1.0, recent_gating_count / 10.0)

        # 波动率趋势（斜率）
        if "volatility" in trends:
            metrics["volatility_trend"] = trends["volatility"].slope
        else:
            metrics["volatility_trend"] = 0.0

        # 风险状态
        state_order = {RiskState.NORMAL: 0, RiskState.WARNING: 1, RiskState.CRITICAL: 2}
        metrics["risk_state"] = float(state_order[state])

        # Cap 命中率（从 allocator 事件中提取）
        cap_hit_count = sum(
            1 for e in signal.allocator_events
            if e.action == "cap_hit"
        )
        metrics["cap_hit_rate"] = cap_hit_count / max(1, len(signal.allocator_events))

        return metrics

    def _evaluate_rule(
        self,
        rule: AlertRule,
        metrics: Dict[str, float],
        strategy_id: str
    ) -> Optional[Alert]:
        """评估单个规则

        Args:
            rule: 告警规则
            metrics: 指标数据
            strategy_id: 策略 ID

        Returns:
            Alert（如果规则触发）或 None
        """
        # 获取指标值
        metric_value = metrics.get(rule.metric_name, 0.0)

        # 检查条件
        triggered = False
        if rule.condition == "greater_than":
            triggered = metric_value > rule.threshold
        elif rule.condition == "less_than":
            triggered = metric_value < rule.threshold
        elif rule.condition == "equals":
            triggered = abs(metric_value - rule.threshold) < 0.01
        elif rule.condition == "trend_up":
            triggered = metric_value > 0.001
        elif rule.condition == "trend_down":
            triggered = metric_value < -0.001

        if not triggered:
            return None

        # 生成告警
        alert = Alert(
            alert_id=f"{rule.rule_id}_{strategy_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            rule_id=rule.rule_id,
            strategy_id=strategy_id,
            severity=rule.severity,
            title=f"[{rule.severity.value.upper()}] {rule.name}",
            message=rule.description,
            metric_name=rule.metric_name,
            current_value=metric_value,
            threshold=rule.threshold,
            context={"metrics": metrics}
        )

        return alert

    def _check_cooldown(self, rule_id: str) -> bool:
        """检查告警冷却时间

        Args:
            rule_id: 规则 ID

        Returns:
            True（可以告警）或 False（冷却中）
        """
        if rule_id not in self.last_alert_time:
            return True

        rule = self.rules[rule_id]
        elapsed = (datetime.now() - self.last_alert_time[rule_id]).total_seconds()
        return elapsed >= rule.cooldown_seconds


class AlertSender:
    """告警发送器

    将告警发送到各种渠道。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化发送器

        Args:
            config: 渠道配置
        """
        self.config = config

    async def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """发送告警到所有配置的渠道

        Args:
            alert: 告警

        Returns:
            发送结果 {channel: success}
        """
        results = {}

        # 发送到各个渠道
        if AlertChannel.LOG in alert.context.get("channels", []):
            results["log"] = self._send_to_log(alert)

        if AlertChannel.SLACK in alert.context.get("channels", []):
            results["slack"] = await self._send_to_slack(alert)

        if AlertChannel.WEBHOOK in alert.context.get("channels", []):
            results["webhook"] = await self._send_to_webhook(alert)

        if AlertChannel.EMAIL in alert.context.get("channels", []):
            results["email"] = await self._send_to_email(alert)

        # 更新告警状态
        alert.sent_channels = [ch for ch, success in results.items() if success]
        alert.failed_channels = [ch for ch, success in results.items() if not success]

        return results

    def _send_to_log(self, alert: Alert) -> bool:
        """发送到日志

        Args:
            alert: 告警

        Returns:
            是否成功
        """
        import logging

        logger = logging.getLogger("risk_alerting")

        # 根据严重程度选择日志级别
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.INFO)

        # 记录日志
        log_message = f"{alert.title} | {alert.message} | {alert.metric_name}: {alert.current_value:.2%}"
        logger.log(log_level, log_message, extra={"alert": alert.to_dict()})

        return True

    async def _send_to_slack(self, alert: Alert) -> bool:
        """发送到 Slack

        Args:
            alert: 告警

        Returns:
            是否成功
        """
        # TODO: 实现 Slack webhook 集成
        # 这里简化为日志输出
        print(f"[SLACK] {alert.title}: {alert.message}")
        return True

    async def _send_to_webhook(self, alert: Alert) -> bool:
        """发送到 Webhook

        Args:
            alert: 告警

        Returns:
            是否成功
        """
        # TODO: 实现 Webhook 集成
        import aiohttp

        webhook_url = self.config.get("webhook_url")
        if not webhook_url:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                payload = alert.to_dict()
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
        except Exception as e:
            print(f"Webhook 发送失败: {e}")
            return False

    async def _send_to_email(self, alert: Alert) -> bool:
        """发送到邮件

        Args:
            alert: 告警

        Returns:
            是否成功
        """
        # TODO: 实现邮件集成
        print(f"[EMAIL] {alert.title}: {alert.message}")
        return True


class AlertingManager:
    """告警管理器

    集成规则引擎和发送器。
    """

    def __init__(self, config_path: Optional[str] = None):
        """初始化管理器

        Args:
            config_path: 配置文件路径
        """
        self.rule_engine = AlertRuleEngine(config_path)

        # TODO: 从配置加载发送器配置
        sender_config = {}
        self.sender = AlertSender(sender_config)

        # 告警历史
        self.alert_history: List[Alert] = []

    def process_risk_update(
        self,
        signal: RiskSignal,
        state: RiskState,
        trends: Dict[str, Any],
        state_transition: Optional[StateTransitionEvent] = None
    ) -> List[Alert]:
        """处理风险更新并生成告警

        Args:
            signal: 风险信号
            state: 当前状态
            trends: 趋势数据
            state_transition: 状态转换（可选）

        Returns:
            告警列表
        """
        # 评估规则
        alerts = self.rule_engine.evaluate(signal, state, trends, state_transition)

        # 发送告警
        for alert in alerts:
            # 异步发送（这里简化为同步）
            results = asyncio.run(self.sender.send_alert(alert))

        # 记录历史
        self.alert_history.extend(alerts)

        return alerts

    def get_recent_alerts(
        self,
        limit: int = 50,
        severity: Optional[AlertSeverity] = None,
        strategy_id: Optional[str] = None
    ) -> List[Alert]:
        """获取最近的告警

        Args:
            limit: 返回数量
            severity: 过滤严重程度
            strategy_id: 过滤策略 ID

        Returns:
            告警列表
        """
        alerts = self.alert_history[-limit:]

        # 应用过滤
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if strategy_id:
            alerts = [a for a in alerts if a.strategy_id == strategy_id]

        return alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计

        Returns:
            统计信息
        """
        total = len(self.alert_history)

        by_severity = {
            AlertSeverity.INFO.value: 0,
            AlertSeverity.WARNING.value: 0,
            AlertSeverity.CRITICAL.value: 0
        }

        for alert in self.alert_history:
            by_severity[alert.severity.value] += 1

        by_strategy: Dict[str, int] = {}
        for alert in self.alert_history:
            by_strategy[alert.strategy_id] = by_strategy.get(alert.strategy_id, 0) + 1

        return {
            "total_alerts": total,
            "by_severity": by_severity,
            "by_strategy": by_strategy,
            "last_24h": sum(
                1 for a in self.alert_history
                if (datetime.now() - a.timestamp).total_seconds() < 86400
            )
        }
