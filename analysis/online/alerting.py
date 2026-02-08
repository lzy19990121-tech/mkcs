"""
SPL-7a-C: 在线告警系统

实现运行态风险告警，不等同于 gating（早期预警）。
"""

import sys
import os
import logging
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

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

    # 上下文（新增字段）
    run_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    links: Dict[str, str] = field(default_factory=dict)  # {"web_ui": "url", "report": "path"}
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
            "run_id": self.run_id,
            "tags": self.tags,
            "links": self.links,
            "context": self.context,
            "sent_channels": self.sent_channels,
            "failed_channels": self.failed_channels
        }

    def to_slack_message(self) -> Dict[str, Any]:
        """转换为 Slack 消息格式"""
        color = {
            AlertSeverity.INFO: "36a64f",     # 绿色
            AlertSeverity.WARNING: "ff9900",  # 橙色
            AlertSeverity.CRITICAL: "ff0000"  # 红色
        }.get(self.severity, "808080")

        # 构建附件字段
        fields = [
            {"title": "策略", "value": self.strategy_id, "short": True},
            {"title": "指标", "value": self.metric_name, "short": True},
            {"title": "当前值", "value": f"{self.current_value:.2%}", "short": True},
            {"title": "阈值", "value": f"{self.threshold:.2%}", "short": True},
        ]

        if self.run_id:
            fields.append({"title": "Run ID", "value": self.run_id, "short": False})

        # 生成链接
        actions = []
        if self.links.get("web_ui"):
            actions.append({
                "type": "button",
                "text": "查看详情",
                "url": self.links["web_ui"]
            })

        attachment = {
            "color": color,
            "title": self.title,
            "text": self.message,
            "fields": fields,
            "footer": f"Alert ID: {self.alert_id}",
            "ts": int(self.timestamp.timestamp())
        }

        if actions:
            attachment["actions"] = actions

        return {"attachments": [attachment]}

    def to_email_subject(self) -> str:
        """生成邮件主题"""
        prefix = {
            AlertSeverity.INFO: "[INFO]",
            AlertSeverity.WARNING: "[WARNING]",
            AlertSeverity.CRITICAL: "[CRITICAL]"
        }.get(self.severity, "[ALERT]")

        run_info = f" [{self.run_id}]" if self.run_id else ""
        return f"{prefix} {self.title}{run_info} - {self.strategy_id}"

    def to_email_body(self) -> str:
        """生成邮件正文"""
        body = f"""
<h2>{self.title}</h2>

<p><strong>策略:</strong> {self.strategy_id}</p>
<p><strong>时间:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        if self.run_id:
            body += f"<p><strong>Run ID:</strong> {self.run_id}</p>"

        body += f"""
<table border="1" cellpadding="5">
<tr><td><strong>指标</strong></td><td>{self.metric_name}</td></tr>
<tr><td><strong>当前值</strong></td><td>{self.current_value:.4f}</td></tr>
<tr><td><strong>阈值</strong></td><td>{self.threshold:.4f}</td></tr>
</table>

<h3>详细信息</h3>
<p>{self.message}</p>
"""

        if self.links:
            body += "<h3>相关链接</h3><ul>"
            for name, url in self.links.items():
                body += f'<li><a href="{url}">{name}</a></li>'
            body += "</ul>"

        return body


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
            threshold=2.0,  # CRITICAL 对应的数值（NORMAL=0, WARNING=1, CRITICAL=2）
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
        state_transition: Optional[StateTransitionEvent] = None,
        run_id: Optional[str] = None,
        links: Optional[Dict[str, str]] = None
    ) -> List[Alert]:
        """评估告警规则

        Args:
            signal: 风险信号
            state: 当前风险状态
            trends: 趋势数据
            state_transition: 状态转换事件（可选）
            run_id: 运行 ID（可选）
            links: 相关链接（可选）

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
            alert = self._evaluate_rule(rule, metrics, signal.strategy_id, run_id, links)
            if alert:
                # 设置渠道
                alert.context["channels"] = rule.channels
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
        state_order = {
            RiskState.NORMAL: 0,
            RiskState.WARNING: 1,
            RiskState.CRITICAL: 2,
            "NORMAL": 0,
            "WARNING": 1,
            "CRITICAL": 2
        }
        # 处理 state 可能是枚举或字符串的情况
        state_key = state if isinstance(state, str) else state.value if hasattr(state, 'value') else state
        metrics["risk_state"] = float(state_order.get(state_key, 0))

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
        strategy_id: str,
        run_id: Optional[str] = None,
        links: Optional[Dict[str, str]] = None
    ) -> Optional[Alert]:
        """评估单个规则

        Args:
            rule: 告警规则
            metrics: 指标数据
            strategy_id: 策略 ID
            run_id: 运行 ID（可选）
            links: 相关链接（可选）

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
            run_id=run_id,
            tags=[rule.rule_id, strategy_id],
            links=links or {},
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

    将告警发送到各种渠道，支持异步、降级和重试。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化发送器

        Args:
            config: 渠道配置
                - dry_run: 只打印不发送
                - webhook_url: Webhook URL
                - slack_webhook_url: Slack Webhook URL
                - smtp_config: SMTP 配置
                - max_retries: 最大重试次数
                - retry_delay: 重试延迟（秒）
                - timeout: 请求超时（秒）
        """
        self.config = config
        self.dry_run = config.get("dry_run", False)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.timeout = config.get("timeout", 10.0)

        # 线程池用于异步发送（不阻塞主循环）
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="alert_sender")

        # 熔断器状态
        self._circuit_breaker: Dict[str, Dict] = {
            "slack": {"failures": 0, "last_failure": None, "open_until": None},
            "webhook": {"failures": 0, "last_failure": None, "open_until": None},
            "email": {"failures": 0, "last_failure": None, "open_until": None}
        }
        self._circuit_breaker_threshold = 5  # 连续失败5次触发熔断
        self._circuit_breaker_timeout = 300  # 熔断5分钟后恢复

        # 通知日志
        self._logger = logging.getLogger("alert_sender")

    def send_alert_async(self, alert: Alert, channels: List[AlertChannel]) -> None:
        """异步发送告警（不阻塞主循环）

        Args:
            alert: 告警
            channels: 要发送的渠道列表
        """
        # 提交到线程池执行
        self.executor.submit(self._send_all_channels, alert, channels)

    def _send_all_channels(self, alert: Alert, channels: List[AlertChannel]) -> Dict[str, bool]:
        """发送到所有渠道（同步方法，在线程池中运行）

        Args:
            alert: 告警
            channels: 渠道列表

        Returns:
            发送结果
        """
        results = {}

        for channel in channels:
            # 检查熔断器
            if self._is_circuit_open(channel.value):
                self._logger.warning(f"熔断器开启，跳过 {channel.value} 发送")
                results[channel.value] = False
                continue

            # 尝试发送
            success = self._send_with_retry(channel, alert)

            # 更新熔断器状态
            if success:
                self._reset_circuit_breaker(channel.value)
            else:
                self._record_failure(channel.value)

            results[channel.value] = success

        # 更新告警状态
        alert.sent_channels = [ch for ch, success in results.items() if success]
        alert.failed_channels = [ch for ch, success in results.items() if not success]

        return results

    def _send_with_retry(self, channel: AlertChannel, alert: Alert) -> bool:
        """带重试的发送

        Args:
            channel: 渠道
            alert: 告警

        Returns:
            是否成功
        """
        for attempt in range(self.max_retries):
            try:
                if channel == AlertChannel.LOG:
                    return self._send_to_log(alert)
                elif channel == AlertChannel.SLACK:
                    return self._send_to_slack(alert)
                elif channel == AlertChannel.WEBHOOK:
                    return self._send_to_webhook(alert)
                elif channel == AlertChannel.EMAIL:
                    return self._send_to_email(alert)
                else:
                    self._logger.warning(f"未知渠道: {channel}")
                    return False

            except Exception as e:
                self._logger.error(f"发送到 {channel.value} 失败 (尝试 {attempt+1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避

        return False

    def _is_circuit_open(self, channel: str) -> bool:
        """检查熔断器是否开启

        Args:
            channel: 渠道名称

        Returns:
            是否熔断
        """
        breaker = self._circuit_breaker.get(channel, {})
        if breaker.get("open_until"):
            if datetime.now() < breaker["open_until"]:
                return True
            else:
                # 熔断超时，重置
                self._reset_circuit_breaker(channel)
        return False

    def _record_failure(self, channel: str) -> None:
        """记录失败，可能触发熔断

        Args:
            channel: 渠道名称
        """
        breaker = self._circuit_breaker.get(channel, {})
        breaker["failures"] = breaker.get("failures", 0) + 1
        breaker["last_failure"] = datetime.now()

        if breaker["failures"] >= self._circuit_breaker_threshold:
            breaker["open_until"] = datetime.now() + timedelta(seconds=self._circuit_breaker_timeout)
            self._logger.warning(f"{channel} 熔断器已开启")

        self._circuit_breaker[channel] = breaker

    def _reset_circuit_breaker(self, channel: str) -> None:
        """重置熔断器

        Args:
            channel: 渠道名称
        """
        self._circuit_breaker[channel] = {
            "failures": 0,
            "last_failure": None,
            "open_until": None
        }

    def _send_to_log(self, alert: Alert) -> bool:
        """发送到日志

        Args:
            alert: 告警

        Returns:
            是否成功
        """
        try:
            logger = logging.getLogger("risk_alerting")

            # 根据严重程度选择日志级别
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }.get(alert.severity, logging.INFO)

            # 记录日志
            run_info = f" [{alert.run_id}]" if alert.run_id else ""
            log_message = f"{alert.title}{run_info} | {alert.message} | {alert.metric_name}: {alert.current_value:.2%}"
            logger.log(log_level, log_message, extra={"alert": alert.to_dict()})

            return True
        except Exception as e:
            self._logger.error(f"日志发送失败: {e}")
            return False

    def _send_to_slack(self, alert: Alert) -> bool:
        """发送到 Slack

        Args:
            alert: 告警

        Returns:
            是否成功
        """
        if self.dry_run:
            print(f"[DRY-RUN SLACK] {alert.title}: {alert.message}")
            return True

        slack_webhook_url = self.config.get("slack_webhook_url")
        if not slack_webhook_url:
            self._logger.warning("未配置 Slack Webhook URL")
            return False

        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            # 配置重试
            retry_strategy = Retry(total=2, backoff_factor=1)
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session = requests.Session()
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            # 发送消息
            response = session.post(
                slack_webhook_url,
                json=alert.to_slack_message(),
                timeout=self.timeout
            )
            response.raise_for_status()

            return response.status_code == 200

        except Exception as e:
            self._logger.error(f"Slack 发送失败: {e}")
            return False

    def _send_to_webhook(self, alert: Alert) -> bool:
        """发送到 Webhook

        Args:
            alert: 告警

        Returns:
            是否成功
        """
        if self.dry_run:
            print(f"[DRY-RUN WEBHOOK] {alert.title}: {alert.message}")
            return True

        webhook_url = self.config.get("webhook_url")
        if not webhook_url:
            self._logger.warning("未配置 Webhook URL")
            return False

        try:
            import requests

            response = requests.post(
                webhook_url,
                json=alert.to_dict(),
                timeout=self.timeout
            )
            response.raise_for_status()

            return response.status_code == 200

        except Exception as e:
            self._logger.error(f"Webhook 发送失败: {e}")
            return False

    def _send_to_email(self, alert: Alert) -> bool:
        """发送到邮件

        Args:
            alert: 告警

        Returns:
            是否成功
        """
        if self.dry_run:
            print(f"[DRY-RUN EMAIL] {alert.to_email_subject()}")
            return True

        smtp_config = self.config.get("smtp_config")
        if not smtp_config:
            self._logger.warning("未配置 SMTP")
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # 创建邮件
            msg = MIMEMultipart('alternative')
            msg['Subject'] = alert.to_email_subject()
            msg['From'] = smtp_config.get("from")
            msg['To'] = ", ".join(smtp_config.get("to", []))

            # 添加 HTML 正文
            html_part = MIMEText(alert.to_email_body(), 'html')
            msg.attach(html_part)

            # 发送邮件
            with smtplib.SMTP(
                smtp_config.get("host", "localhost"),
                smtp_config.get("port", 25),
                timeout=self.timeout
            ) as server:
                if smtp_config.get("use_tls", False):
                    server.starttls()

                if smtp_config.get("username"):
                    server.login(
                        smtp_config["username"],
                        smtp_config["password"]
                    )

                server.send_message(msg)

            return True

        except Exception as e:
            self._logger.error(f"邮件发送失败: {e}")
            return False

    def shutdown(self) -> None:
        """关闭发送器，等待所有任务完成"""
        # Python 3.8 兼容性：timeout 参数在某些版本不可用
        try:
            self.executor.shutdown(wait=True, timeout=30)
        except TypeError:
            # 降级：不使用 timeout
            self.executor.shutdown(wait=True)


class AlertingManager:
    """告警管理器

    集成规则引擎和发送器。
    """

    def __init__(self, config_path: Optional[str] = None, sender_config: Optional[Dict[str, Any]] = None):
        """初始化管理器

        Args:
            config_path: 配置文件路径
            sender_config: 发送器配置
        """
        self.rule_engine = AlertRuleEngine(config_path)

        # 从配置加载发送器配置
        sender_config = sender_config or self._load_sender_config()
        self.sender = AlertSender(sender_config)

        # 告警历史
        self.alert_history: List[Alert] = []
        self._max_history = 1000

        # 当前运行上下文
        self._current_run_id: Optional[str] = None

    def _load_sender_config(self) -> Dict[str, Any]:
        """从配置文件或环境变量加载发送器配置"""
        import os

        config = {}

        # 从环境变量加载配置
        config["dry_run"] = os.getenv("ALERT_DRY_RUN", "false").lower() == "true"
        config["slack_webhook_url"] = os.getenv("SLACK_WEBHOOK_URL")
        config["webhook_url"] = os.getenv("ALERT_WEBHOOK_URL")
        config["smtp_config"] = {
            "host": os.getenv("SMTP_HOST"),
            "port": int(os.getenv("SMTP_PORT", "25")),
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "from": os.getenv("SMTP_FROM"),
            "to": os.getenv("SMTP_TO", "").split(","),
            "use_tls": os.getenv("SMTP_USE_TLS", "false").lower() == "true"
        }

        return config

    def set_run_context(self, run_id: str, links: Optional[Dict[str, str]] = None) -> None:
        """设置当前运行上下文

        Args:
            run_id: 运行 ID
            links: 相关链接
        """
        self._current_run_id = run_id
        self._current_links = links or {}

    def clear_run_context(self) -> None:
        """清除运行上下文"""
        self._current_run_id = None
        self._current_links = {}

    def process_risk_update(
        self,
        signal: RiskSignal,
        state: RiskState,
        trends: Dict[str, Any],
        state_transition: Optional[StateTransitionEvent] = None,
        run_id: Optional[str] = None,
        links: Optional[Dict[str, str]] = None
    ) -> List[Alert]:
        """处理风险更新并生成告警

        Args:
            signal: 风险信号
            state: 当前状态
            trends: 趋势数据
            state_transition: 状态转换（可选）
            run_id: 运行 ID（可选）
            links: 相关链接（可选）

        Returns:
            告警列表
        """
        # 使用传入的或上下文的 run_id/links
        run_id = run_id or self._current_run_id
        links = links or self._current_links

        # 评估规则
        alerts = self.rule_engine.evaluate(signal, state, trends, state_transition, run_id, links)

        if not alerts:
            return []

        # 异步发送告警（不阻塞主循环）
        for alert in alerts:
            # 收集要发送的渠道
            channels = alert.context.get("channels", [AlertChannel.LOG])
            self.sender.send_alert_async(alert, channels)

        # 记录历史
        self.alert_history.extend(alerts)

        # 限制历史大小
        if len(self.alert_history) > self._max_history:
            self.alert_history = self.alert_history[-self._max_history:]

        return alerts

    def send_manual_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        strategy_id: str = "system",
        tags: Optional[List[str]] = None,
        links: Optional[Dict[str, str]] = None
    ) -> Alert:
        """手动发送告警

        Args:
            title: 标题
            message: 消息
            severity: 严重程度
            strategy_id: 策略 ID
            tags: 标签
            links: 链接

        Returns:
            创建的告警
        """
        alert = Alert(
            alert_id=f"manual_{strategy_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            rule_id="manual",
            strategy_id=strategy_id,
            severity=severity,
            title=title,
            message=message,
            metric_name="manual",
            current_value=0.0,
            threshold=0.0,
            run_id=self._current_run_id,
            tags=tags or [],
            links=links or {},
            context={"channels": [AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.WEBHOOK]}
        )

        # 发送告警
        channels = alert.context.get("channels", [AlertChannel.LOG])
        self.sender.send_alert_async(alert, channels)

        # 记录历史
        self.alert_history.append(alert)

        return alert

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

        # 发送成功率统计
        total_sent = 0
        total_failed = 0
        for alert in self.alert_history:
            total_sent += len(alert.sent_channels)
            total_failed += len(alert.failed_channels)

        return {
            "total_alerts": total,
            "by_severity": by_severity,
            "by_strategy": by_strategy,
            "last_24h": sum(
                1 for a in self.alert_history
                if (datetime.now() - a.timestamp).total_seconds() < 86400
            ),
            "sent_success_rate": total_sent / max(1, total_sent + total_failed),
            "circuit_breaker_status": self.sender._circuit_breaker
        }

    def shutdown(self) -> None:
        """关闭管理器"""
        self.sender.shutdown()
