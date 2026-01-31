# 实时监控与告警系统文档

## 概述

实现了完整的实时监控与告警系统，用于在交易过程中监控市场、风险、系统和模型状态，并在异常情况下及时告警。

## 核心组件

### 1. AlertType（告警类型）

支持 14 种告警类型，分为 4 大类：

#### 交易类
- `ORDER_FILLED` - 订单成交
- `ORDER_REJECTED` - 订单被拒
- `POSITION_OPENED` - 开仓
- `POSITION_CLOSED` - 平仓
- `TRADE_SIGNAL` - 交易信号

#### 风险类
- `POSITION_LOSS` - 持仓亏损超限
- `DAILY_LOSS_LIMIT` - 单日亏损限制
- `DRAWDOWN_LIMIT` - 回撤超限

#### 模型类
- `MODEL_CONFIDENCE_LOW` - 模型置信度过低
- `MODEL_PREDICTION_CHANGE` - 模型预测突变

#### 系统类
- `DATA_ANOMALY` - 数据异常
- `CONNECTION_ERROR` - 连接错误
- `SYSTEM_ERROR` - 系统错误

### 2. Alert（告警数据）

```python
@dataclass
class Alert:
    alert_type: AlertType      # 告警类型
    severity: str              # 严重程度 (INFO/LOW/MEDIUM/HIGH/CRITICAL)
    symbol: Optional[str]      # 标的代码
    message: str               # 告警消息
    timestamp: datetime        # 时间戳
    metadata: Dict             # 元数据
```

### 3. AlertChannel（告警渠道）

#### ConsoleAlertChannel（控制台）
- 带颜色的日志输出
- 不同严重程度不同颜色

#### FileAlertChannel（文件）
- 写入 JSON 格式日志
- 支持自动轮转

#### EmailAlertChannel（邮件）
- SMTP 协议发送邮件
- 支持 HTML 格式

#### WebhookAlertChannel（Webhook）
- HTTP POST 发送
- JSON 格式

### 4. AlertManager（告警管理器）

核心告警管理类，提供 9 种检查方法：

#### 检查方法

| 方法 | 说明 | 阈值示例 |
|------|------|---------|
| `check_position_loss()` | 单个持仓亏损 | 亏损 > 5% |
| `check_daily_loss_limit()` | 单日亏损 | 亏损 > 5% |
| `check_drawdown_limit()` | 最大回撤 | 回撤 > 15% |
| `check_model_confidence()` | 模型置信度 | 置信度 < 0.6 |
| `check_prediction_change()` | 预测突变 | 变动 > 0.5 |
| `check_data_anomaly()` | 数据异常 | 价格/成交量异常 |
| `check_order_status()` | 订单状态 | 拒单/部分成交 |
| `check_connection_error()` | 连接状态 | API 连接失败 |
| `check_system_error()` | 系统错误 | 异常捕获 |

### 5. TradingMonitor（交易监控器）

实时监控交易系统：

```python
monitor = TradingMonitor(
    alert_manager=alert_manager,
    check_interval=60          # 检查间隔（秒）
)

monitor.start_monitoring(broker, data_source, symbols)
```

## 使用方法

### 1. 基本使用

```python
from monitoring.alerts import AlertManager, AlertType, AlertSeverity
from monitoring.alerts import ConsoleAlertChannel, FileAlertChannel

# 创建告警管理器
alert_manager = AlertManager()

# 添加告警渠道
alert_manager.add_channel(ConsoleAlertChannel())
alert_manager.add_channel(FileAlertChannel("logs/alerts.log"))

# 检查风险
alert_manager.check_daily_loss_limit(
    daily_pnl=Decimal("-6000"),
    initial_equity=Decimal("100000"),
    limit_percent=0.05
)
```

### 2. 自定义阈值

```python
# 配置阈值
alert_manager.set_threshold('position_loss_percent', 0.05)    # 5%
alert_manager.set_threshold('daily_loss_percent', 0.05)       # 5%
alert_manager.set_threshold('drawdown_percent', 0.15)         # 15%
alert_manager.set_threshold('model_confidence', 0.6)          # 0.6
alert_manager.set_threshold('prediction_change', 0.5)         # 0.5
```

### 3. 邮件告警

```python
from monitoring.alerts import EmailAlertChannel

email_channel = EmailAlertChannel(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="your_email@gmail.com",
    password="your_password",
    from_addr="your_email@gmail.com",
    to_addr=["trader@example.com"]
)

alert_manager.add_channel(email_channel)
```

### 4. Webhook 告警

```python
from monitoring.alerts import WebhookAlertChannel

webhook_channel = WebhookAlertChannel(
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
)

alert_manager.add_channel(webhook_channel)
```

### 5. 实时监控

```python
from monitoring.alerts import TradingMonitor

# 创建监控器
monitor = TradingMonitor(
    alert_manager=alert_manager,
    check_interval=60  # 每 60 秒检查一次
)

# 启动监控
monitor.start_monitoring(
    broker=broker,
    data_source=data_source,
    symbols=["AAPL", "MSFT", "GOOGL"]
)
```

## 告警级别

| 级别 | 颜色 | 说明 | 示例 |
|------|------|------|------|
| INFO | 灰色 | 信息 | 订单成交 |
| LOW | 蓝色 | 低风险 | 置信度略低 |
| MEDIUM | 黄色 | 中等风险 | 持仓亏损 3% |
| HIGH | 橙色 | 高风险 | 数据异常 |
| CRITICAL | 红色 | 严重 | 单日亏损超限 |

## 配置示例

### 保守交易者

```python
alert_manager = AlertManager()
alert_manager.set_threshold('position_loss_percent', 0.03)   # 3%
alert_manager.set_threshold('daily_loss_percent', 0.03)      # 3%
alert_manager.set_threshold('drawdown_percent', 0.10)        # 10%
```

### 激进交易者

```python
alert_manager = AlertManager()
alert_manager.set_threshold('position_loss_percent', 0.10)   # 10%
alert_manager.set_threshold('daily_loss_percent', 0.10)      # 10%
alert_manager.set_threshold('drawdown_percent', 0.25)        # 25%
```

## 集成到回测系统

```python
from agent.replay_engine import ReplayEngine
from monitoring.alerts import AlertManager, ConsoleAlertChannel

# 创建告警管理器
alert_manager = AlertManager()
alert_manager.add_channel(ConsoleAlertChannel())

# 创建回测引擎
engine = ReplayEngine(
    data_source=source,
    strategy=strategy,
    risk_manager=risk_manager,
    broker=broker,
    alert_manager=alert_manager  # 注入告警管理器
)

# 运行回测（会自动触发告警）
engine.runbacktest(
    symbol="AAPL",
    start_date=start,
    end_date=end,
    initial_cash=Decimal("100000")
)
```

## 性能

| 指标 | 数值 |
|------|------|
| 检查延迟 | < 10ms per check |
| 内存占用 | ~100KB per alert |
| 告警发送 | < 100ms (邮件 < 1s) |

## 注意事项

1. **告警风暴**: 避免设置阈值过低，导致告警过多
2. **渠道配置**: 建议至少配置 Console + File 两种渠道
3. **邮件安全**: 使用应用专用密码，不要使用账户密码
4. **监控频率**: 建议 60-300 秒间隔

## 参考资料

- [Python logging 模块](https://docs.python.org/3/library/logging.html)
- [SMTP 邮件发送](https://docs.python.org/3/library/smtplib.html)
- [Webhook 最佳实践](https://slack.dev/building-blocks/tutorials/creating-webhooks)
