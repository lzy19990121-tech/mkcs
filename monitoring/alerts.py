"""
实时监控和告警系统

监控交易系统状态，检测异常并触发告警
"""

import logging
import smtplib
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """告警类型"""
    # 交易相关
    POSITION_LOSS = "position_loss"              # 仓位亏损超标
    DAILY_LOSS_LIMIT = "daily_loss_limit"      # 单日亏损限制
    DAILY_PROFIT_TARGET = "daily_profit_target"  # 单日盈利目标
    MAX_DRAWDOWN = "max_drawdown"              # 最大回撤超标

    # 模型相关
    MODEL_PERFORMANCE_DROP = "model_perf_drop"  # 模型性能下降
    MODEL_ERROR = "model_error"                  # 模型执行错误

    # 系统相关
    DATA_ANOMALY = "data_anomaly"              # 数据异常
    SYSTEM_ERROR = "system_error"                # 系统错误
    CONNECTION_LOST = "connection_lost"         # 连接丢失
    API_RATE_LIMIT = "api_rate_limit"           # API 限流

    # 策略相关
    STRATEGY_SIGNAL_SURGE = "signal_surge"      # 信号激增
    STRATEGY_INACTIVITY = "strategy_inactivity"  # 策略长时间无信号

    # 风控相关
    RISK_LIMIT_BREACH = "risk_limit_breach"      # 风控限制触发
    MARGIN_CALL = "margin_call"                  # 追加保证金


@dataclass
class Alert:
    """告警对象"""
    alert_type: AlertType
    severity: str  # low, medium, high, critical
    timestamp: datetime
    title: str
    message: str
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'title': self.title,
            'message': self.message,
            'symbol': self.symbol,
            'metadata': self.metadata
        }


class AlertChannel:
    """告警通道"""

    def send(self, alert: Alert) -> bool:
        """发送告警

        Args:
            alert: 告警对象

        Returns:
            是否发送成功
        """
        raise NotImplementedError


class ConsoleAlertChannel(AlertChannel):
    """控制台告警通道"""

    def send(self, alert: Alert) -> bool:
        """输出告警到控制台"""
        timestamp_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # 根据严重程度使用不同的颜色
        severity_colors = {
            'low': '\033[0;32m',      # 绿色
            'medium': '\033[0;33m',    # 黄色
            'high': '\033[0;31m',     # 红色
            'critical': '\033[0;35m'  # 品红
        }

        color = severity_colors.get(alert.severity, '')
        reset = '\033[0m'

        print(f"{color}[{alert.severity.upper()}]{reset} {timestamp_str} - {alert.title}")
        print(f"  {alert.message}")
        if alert.symbol:
            print(f"  标的: {alert.symbol}")
        print(f"  类型: {alert.alert_type.value}")

        return True


class FileAlertChannel(AlertChannel):
    """文件告警通道（持久化）"""

    def __init__(self, log_file: str = "logs/alerts.log"):
        """初始化文件通道

        Args:
            log_file: 日志文件路径
        """
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> bool:
        """写入告警到文件"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
            return True
        except Exception as e:
            logger.error(f"写入告警文件失败: {e}")
            return False


class EmailAlertChannel(AlertChannel):
    """邮件告警通道"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str]
    ):
        """初始化邮件通道

        Args:
            smtp_server: SMTP 服务器地址
            smtp_port: SMTP 端口
            username: 用户名
            password: 密码
            from_addr: 发件人地址
            to_addrs: 收件人地址列表
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def send(self, alert: Alert) -> bool:
        """发送邮件告警"""
        try:
            # 根据严重程度设置邮件主题前缀
            severity_prefix = {
                'low': '[INFO]',
                'medium': '[WARNING]',
                'high': '[ALERT]',
                'critical': '[CRITICAL]'
            }.get(alert.severity, '[INFO]')

            subject = f"{severity_prefix} {alert.title}"

            # 构建邮件内容
            body = f"""
时间: {alert.timestamp}
类型: {alert.alert_type.value}
严重程度: {alert.severity}

{alert.message}
"""

            if alert.symbol:
                body += f"\n标的: {alert.symbol}"

            if alert.metadata:
                body += f"\n详细信息:\n{json.dumps(alert.metadata, indent=2)}"

            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(
                    self.from_addr,
                    self.to_addrs,
                    subject.encode('utf-8'),
                    body.encode('utf-8')
                )

            logger.info(f"邮件告警已发送: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")
            return False


class WebhookAlertChannel(AlertChannel):
    """Webhook 告警通道"""

    def __init__(self, webhook_url: str):
        """初始化 Webhook 通道

        Args:
            webhook_url: Webhook URL
        """
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> bool:
        """发送 Webhook"""
        try:
            import urllib.request

            data = json.dumps(alert.to_dict()).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.info(f"Webhook 告警已发送: {alert.title}")
                    return True
                else:
                    logger.warning(f"Webhook 返回错误: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"发送 Webhook 失败: {e}")
            return False


class AlertManager:
    """告警管理器

    监控系统状态并触发告警
    """

    def __init__(
        self,
        channels: Optional[List[AlertChannel]] = None,
        min_alert_interval: int = 60  # 最小告警间隔（秒）
    ):
        """初始化告警管理器

        Args:
            channels: 告警通道列表
            min_alert_interval: 同一类型的两次告警最小间隔
        """
        self.channels = channels or [ConsoleAlertChannel()]
        self.min_alert_interval = min_alert_interval

        # 告警状态
        self._last_alert_time: Dict[AlertType, datetime] = {}
        self._alert_counts: Dict[AlertType, int] = {}

        # 监控规则配置
        self.config = {
            # 仓位告警
            'position_loss_threshold': Decimal("0.05"),  # 单个仓位亏损 5%
            'max_total_loss': Decimal("0.10"),       # 总亏损 10%

            # 单日限制
            'daily_loss_limit': Decimal("0.05"),     # 单日亏损 5%
            'daily_profit_target': Decimal("0.03"),  # 单日盈利目标 3%

            # 回撤告警
            'max_drawdown_threshold': Decimal("0.15"), # 最大回撤 15%

            # 模型告警
            'model_accuracy_threshold': 0.5,      # 模型准确率阈值
            'recent_signals_threshold': 100,        # 短时间内信号数阈值

            # 数据告警
            'data_age_limit': 300,                  # 数据时效限制（秒）
        }

    def check_and_alert(
        self,
        alert_type: AlertType,
        severity: str = "medium",
        **kwargs
    ) -> bool:
        """检查并发送告警

        Args:
            alert_type: 告警类型
            severity: 严重程度
            **kwargs: 告警元数据

        Returns:
            是否发送了告警
        """
        # 检查告警频率限制
        now = datetime.now()
        last_time = self._last_alert_time.get(alert_type)

        if last_time and (now - last_time).total_seconds() < self.min_alert_interval:
            logger.debug(f"告警频率限制: {alert_type.value}")
            return False

        # 创建告警
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            timestamp=now,
            title=self._generate_title(alert_type, kwargs),
            message=self._generate_message(alert_type, kwargs),
            symbol=kwargs.get('symbol'),
            metadata=kwargs
        )

        # 记录
        self._last_alert_time[alert_type] = now
        self._alert_counts[alert_type] = self._alert_counts.get(alert_type, 0) + 1

        # 发送到所有通道
        success_count = 0
        for channel in self.channels:
            if channel.send(alert):
                success_count += 1

        logger.info(f"告警已发送: {alert.title} ({success_count}/{len(self.channels)} 通道成功)")
        return success_count > 0

    def check_position_loss(self, positions: Dict, current_prices: Dict[str, Decimal]):
        """检查仓位亏损

        Args:
            positions: 持仓字典 {symbol: position}
            current_prices: 当前价格字典 {symbol: price}
        """
        threshold = self.config['position_loss_threshold']

        for symbol, pos in positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            cost_basis = pos.avg_price
            loss_ratio = (cost_basis - current_price) / cost_basis if pos.is_long else (current_price - cost_basis) / cost_basis

            if loss_ratio > threshold:
                self.check_and_alert(
                    AlertType.POSITION_LOSS,
                    severity="high",
                    symbol=symbol,
                    loss_ratio=float(loss_ratio),
                    unrealized_pnl=float(pos.unrealized_pnl),
                    position_value=float(pos.market_value)
                )

    def check_daily_loss_limit(self, daily_pnl: Decimal, initial_equity: Decimal):
        """检查单日亏损限制

        Args:
            daily_pnl: 当日盈亏
            initial_equity: 初始权益
        """
        daily_loss_limit = self.config['daily_loss_limit']

        loss_ratio = -daily_pnl / initial_equity if daily_pnl < 0 else 0

        if loss_ratio > daily_loss_limit:
            self.check_and_alert(
                AlertType.DAILY_LOSS_LIMIT,
                severity="critical",
                daily_pnl=float(daily_pnl),
                loss_ratio=float(loss_ratio),
                limit=float(daily_loss_limit)
            )

    def check_daily_profit_target(self, daily_pnl: Decimal, initial_equity: Decimal):
        """检查单日盈利目标

        Args:
            daily_pnl: 当日盈亏
            initial_equity: 初始权益
        """
        profit_target = self.config['daily_profit_target']
        profit_ratio = daily_pnl / initial_equity

        if profit_ratio > profit_target:
            self.check_and_alert(
                AlertType.DAILY_PROFIT_TARGET,
                severity="low",
                daily_pnl=float(daily_pnl),
                profit_ratio=float(profit_ratio),
                target=float(profit_target)
            )

    def check_model_performance(self, recent_accuracy: float):
        """检查模型性能

        Args:
            recent_accuracy: 最近准确率
        """
        threshold = self.config['model_accuracy_threshold']

        if recent_accuracy < threshold:
            self.check_and_alert(
                AlertType.MODEL_PERFORMANCE_DROP,
                severity="medium",
                recent_accuracy=recent_accuracy,
                threshold=threshold
            )

    def check_max_drawdown(self, current_drawdown: Decimal, peak_equity: Decimal):
        """检查最大回撤

        Args:
            current_drawdown: 当前回撤
            peak_equity: 峰值权益
        """
        threshold = self.config['max_drawdown_threshold']

        if current_drawdown > threshold:
            self.check_and_alert(
                AlertType.MAX_DRAWDOWN,
                severity="critical",
                current_drawdown=float(current_drawdown),
                peak_equity=float(peak_equity),
                threshold=float(threshold)
            )

    def check_data_anomaly(self, data_age_seconds: int, symbol: str):
        """检查数据异常

        Args:
            data_age_seconds: 数据延迟（秒）
            symbol: 标的代码
        """
        limit = self.config['data_age_limit']

        if data_age_seconds > limit:
            self.check_and_alert(
                AlertType.DATA_ANOMALY,
                severity="high",
                symbol=symbol,
                data_age=data_age_seconds,
                limit=limit
            )

    def check_signal_surge(self, recent_signal_count: int):
        """检查信号激增

        Args:
            recent_signal_count: 最近信号数量
        """
        threshold = self.config['recent_signals_threshold']

        if recent_signal_count > threshold:
            self.check_and_alert(
                AlertType.STRATEGY_SIGNAL_SURGE,
                severity="medium",
                signal_count=recent_signal_count,
                threshold=threshold
            )

    def check_strategy_inactivity(self, last_signal_time: datetime, symbol: str):
        """检查策略不活跃

        Args:
            last_signal_time: 上次信号时间
            symbol: 标的代码
        """
        inactive_hours = (datetime.now() - last_signal_time).total_seconds() / 3600

        if inactive_hours > 24:  # 超过 24 小时无信号
            self.check_and_alert(
                AlertType.STRATEGY_INACTIVITY,
                severity="low",
                symbol=symbol,
                inactive_hours=inactive_hours
            )

    def get_alert_stats(self) -> Dict:
        """获取告警统计"""
        return {
            'total_alerts': sum(self._alert_counts.values()),
            'alerts_by_type': dict(self._alert_counts),
            'active_channels': len(self.channels),
            'config': self.config
        }

    def _generate_title(self, alert_type: AlertType, metadata: Dict) -> str:
        """生成告警标题"""
        templates = {
            AlertType.POSITION_LOSS: "仓位亏损告警",
            AlertType.DAILY_LOSS_LIMIT: "单日亏损限制触发",
            AlertType.DAILY_PROFIT_TARGET: "单日盈利目标达成",
            AlertType.MAX_DRAWDOWN: "最大回撤超标",
            AlertType.MODEL_PERFORMANCE_DROP: "模型性能下降",
            AlertType.MODEL_ERROR: "模型执行错误",
            AlertType.DATA_ANOMALY: "数据异常",
            AlertType.SYSTEM_ERROR: "系统错误",
            AlertType.CONNECTION_LOST: "连接丢失",
            AlertType.API_RATE_LIMIT: "API 限流",
            AlertType.STRATEGY_SIGNAL_SURGE: "信号激增",
            AlertType.STRATEGY_INACTIVITY: "策略不活跃",
            AlertType.RISK_LIMIT_BREACH: "风控限制触发",
            AlertType.MARGIN_CALL: "追加保证金"
        }

        title = templates.get(alert_type, "告警")

        # 添加标的
        if 'symbol' in metadata:
            title = f"[{metadata['symbol']}] {title}"

        return title

    def _generate_message(self, alert_type: AlertType, metadata: Dict) -> str:
        """生成告警消息"""
        messages = {
            AlertType.POSITION_LOSS: "{symbol} 仓位亏损 {loss_ratio:.2%}，浮亏 ${unrealized_pnl:.2f}",
            AlertType.DAILY_LOSS_LIMIT: "当日亏损 {loss_ratio:.2%}，${daily_pnl:.2f}，超过限制 {limit:.2%}",
            AlertType.DAILY_PROFIT_TARGET: "当日盈利 {profit_ratio:.2%}，${daily_pnl:.2f}，达到目标 {target:.2%}",
            AlertType.MAX_DRAWDOWN: "当前回撤 {drawdown:.2%}，峰值权益 ${peak_equity:.2f}，超过阈值 {threshold:.2%}",
            AlertType.MODEL_PERFORMANCE_DROP: "模型准确率 {accuracy:.2%} 低于阈值 {threshold:.2%}",
            AlertType.MODEL_ERROR: "模型执行错误: {error}",
            AlertType.DATA_ANOMALY: "{symbol} 数据延迟 {age} 秒，超过限制 {limit} 秒",
            AlertType.STRATEGY_SIGNAL_SURGE: "短时间内生成 {count} 个信号，超过阈值 {threshold}",
            AlertType.STRATEGY_INACTIVITY: "{symbol} 策略 {hours:.1f} 小时无信号"
        }

        message = messages.get(alert_type, "告警触发")

        try:
            return message.format(**metadata)
        except KeyError as e:
            logger.warning(f"生成告警消息失败: {e}")
            return str(metadata)


class TradingMonitor:
    """交易监控器

    实时监控交易状态，自动触发告警
    """

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        check_interval_seconds: int = 60
    ):
        """初始化监控器

        Args:
            alert_manager: 告警管理器
            check_interval_seconds: 检查间隔
        """
        self.alert_manager = alert_manager or AlertManager()
        self.check_interval = check_interval_seconds

        # 监控状态
        self._running = False
        self._last_check_time = None
        self._peak_equity = Decimal("0")
        self._daily_start_equity = Decimal("0")
        self._daily_pnl = Decimal("0")
        self._recent_signals = []

    def start_monitoring(self, broker, data_source, symbols):
        """开始监控（后台线程）

        Args:
            broker: 经纪商
            data_source: 数据源
            symbols: 监控标的列表
        """
        import threading

        def monitor_loop():
            self._running = True
            self._daily_start_equity = broker.get_total_equity()
            self._peak_equity = self._daily_start_equity

            while self._running:
                try:
                    self._check_cycle(broker, data_source, symbols)
                    import time
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"监控循环异常: {e}")
                    self.alert_manager.check_and_alert(
                        AlertType.SYSTEM_ERROR,
                        severity="critical",
                        error=str(e)
                    )

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

        logger.info("交易监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self._running = False
        logger.info("交易监控已停止")

    def _check_cycle(self, broker, data_source, symbols):
        """执行一次检查周期"""
        now = datetime.now()
        self._last_check_time = now

        # 获取当前状态
        current_equity = broker.get_total_equity()
        positions = broker.get_positions()
        trades = broker.get_trades()

        # 计算当日盈亏
        daily_pnl = current_equity - self._daily_start_equity
        self._daily_pnl = daily_pnl

        # 更新峰值权益
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # 计算当前回撤
        drawdown = (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else Decimal("0")

        # 检查各项规则
        self.alert_manager.check_daily_loss_limit(daily_pnl, self._daily_start_equity)
        self.alert_manager.check_daily_profit_target(daily_pnl, self._daily_start_equity)
        self.alert_manager.check_max_drawdown(drawdown, self._peak_equity)

        # 检查仓位
        current_prices = {}
        for symbol in symbols:
            try:
                bars = data_source.get_bars_until(symbol, now, "1d")
                if bars:
                    current_prices[symbol] = bars[-1].close
            except Exception as e:
                logger.warning(f"获取 {symbol} 价格失败: {e}")

        if positions and current_prices:
            self.alert_manager.check_position_loss(positions, current_prices)

        # 检查数据时效
        for symbol in symbols:
            try:
                bars = data_source.get_bars_until(symbol, now, "1d")
                if bars:
                    data_age = (now - bars[-1].timestamp).total_seconds()
                    self.alert_manager.check_data_anomaly(data_age, symbol)
            except Exception as e:
                self.alert_manager.check_and_alert(
                    AlertType.DATA_ANOMALY,
                    severity="high",
                    symbol=symbol,
                    error=str(e)
                )


if __name__ == "__main__":
    """测试代码"""
    print("=== 告警系统测试 ===\n")

    # 创建告警管理器（只用控制台）
    alert_manager = AlertManager(
        channels=[ConsoleAlertChannel()],
        min_alert_interval=10
    )

    # 测试仓位亏损告警
    print("1. 测试仓位亏损告警:")
    from decimal import Decimal
    from core.models import Position

    positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=100,
            avg_price=Decimal("140"),
            market_value=Decimal("13500"),
            unrealized_pnl=Decimal("-500")
        )
    }
    current_prices = {"AAPL": Decimal("135")}

    alert_manager.check_position_loss(positions, current_prices)

    # 测试单日亏损告警
    print("\n2. 测试单日亏损告警:")
    alert_manager.check_daily_loss_limit(
        daily_pnl=Decimal("-6000"),
        initial_equity=Decimal("100000")
    )

    # 测试数据异常告警
    print("\n3. 测试数据异常告警:")
    alert_manager.check_data_anomaly(
        data_age_seconds=400,
        symbol="AAPL"
    )

    # 显示统计
    print("\n告警统计:")
    stats = alert_manager.get_alert_stats()
    for key, value in stats.items():
        if key != 'config':
            print(f"  {key}: {value}")

    print("\n✓ 测试完成")
