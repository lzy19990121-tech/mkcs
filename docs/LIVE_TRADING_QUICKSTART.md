# 实时交易快速启动指南

## 准备工作（今晚完成）

### 1. 运行预检

```bash
cd /home/neal/mkcs
python scripts/preflight_check.py
```

确保所有检查项通过。如有失败，按提示解决。

### 2. 安装依赖（如需要）

```bash
pip install yfinance pandas numpy pytz
```

### 3. 测试数据源

```bash
python -c "
from skills.market_data.yahoo_source import YahooFinanceSource
source = YahooFinanceSource()
quote = source.get_quote('AAPL')
print(f'AAPL 最新价: ${quote.bid_price}')
"
```

---

## 明晚盘前准备（约 15 分钟）

### 1. 确认市场时段

- **美股交易时间**: 21:30 - 次日 04:00 (北京时间)
- **建议**: 21:00 前完成所有准备

### 2. 运行预检（再次确认）

```bash
python scripts/preflight_check.py
```

### 3. 决定交易标的和策略

**推荐标的**:
- 保守型: `AAPL`, `MSFT` (大盘股，流动性好)
- 进取型: `TSLA`, `NVDA` (波动大)
- 混合型: `AAPL`, `TSLA`, `GOOGL`

**推荐策略**:
- MA(5, 20): 经典双均线
- MA(3, 10): 快速交易
- MA(10, 30): 稳健交易

---

## 启动实时交易

### 基础启��

```bash
python scripts/run_live_with_monitoring.py --symbols AAPL MSFT --cash 100000
```

### 完整参数

```bash
python scripts/run_live_with_monitoring.py \
    --symbols AAPL MSFT GOOGL \
    --cash 100000 \
    --interval 5m \
    --fast 5 \
    --slow 20
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbols` | AAPL MSFT GOOGL | 交易标的 |
| `--cash` | 100000 | 初始资金 |
| `--interval` | 5m | 数据更新间隔（1m/5m/15m） |
| `--fast` | 5 | MA 快线周期 |
| `--slow` | 20 | MA 慢线周期 |

---

## 运行中监控

### 正常输出示例

```
[信号] AAPL: BUY @ $175.50
[订单通过] AAPL: BUY @ $175.50
[状态转换] AAPL: NORMAL → WARNING, 触发指标: envelope_usage (0.75)
[告警] AAPL: [WARNING] 接近 Envelope - 回撤接近 envelope 上限（70%）
```

### 关键监控点

1. **数据获取**: "获取到 XXX 根K线"
2. **信号生成**: "[信号] XXX: BUY/SELL/HOLD"
3. **风控通过**: "[订单通过] XXX: 已通过风控"
4. **风险状态**: "[状态转换] XXX: NORMAL → WARNING"
5. **告警触发**: "[告警] XXX: [WARNING/CRITICAL]"

---

## 停止交易

### 正常停止

```
Ctrl + C
```

系统会打印统计总结并保存数据。

### 紧急停止

如果出现异常，直接关闭终端。系统会在下次重启时恢复。

---

## 事后分析（盘后）

### 1. 查看风险事件

```bash
sqlite3 data/risk_events.db "SELECT * FROM risk_events ORDER BY timestamp DESC LIMIT 10;"
```

### 2. 查看交易记录

交易记录保存在 `broker` 对应的数据库中。

### 3. 生成 Post-mortem

如果触发了 gating 或告警，会自动生成 post-mortem 报告。

---

## 故障排查

### 问题 1: Yahoo Finance 连接失败

**症状**: `获取 AAPL 数据失败: HTTPSConnectionPool`

**解决**:
1. 检查网络连接
2. 使用 VPN（如需要）
3. 尝试使用混合数据源

### 问题 2: 无数据返回

**症状**: `未获取到数据: AAPL`

**解决**:
1. 检查标的代码是否正确（如 `0700.HK` 而非 `0700`）
2. 确认市场开盘时间
3. 尝试更长的 `interval`

### 问题 3: 策略无信号

**症状**: 长时间没有 "[信号]" 输出

**原因**: 正常，策略等待金叉/死叉

**操作**:
- 可以调整 MA 周期（如 `--fast 3 --slow 10`）
- 可以增加标的数量
- 或等待市场变化

---

## 风险提示

### ⚠️ 重要提醒

1. **Paper 模式**: 当前为模拟盘，不涉及真实资金
2. **数据延迟**: Yahoo Finance 数据可能有延迟
3. **网络风险**: 依赖网络稳定性
4. **策略风险**: MA 策略在震荡市场可能亏损
5. **资金管理**: 建议单笔不超过资金的 20%

### 🛑 止损设置

默认紧急止损: **5%**

可通过修改 `LiveTradingConfig` 调整。

---

## 下一步优化

### 1. 调整告警阈值

编辑 `config/alerting_rules.yaml`:

```yaml
envelope_approach:
  threshold: 0.7  # 改为 0.6 提前告警
```

### 2. 添加更多策略

```python
from skills.strategy.breakout import BreakoutStrategy
strategy = BreakoutStrategy(period=20)
```

### 3. 启用 SPL-7b 反事实分析

```bash
python scripts/run_counterfactual_analysis.py \
    --replay-path runs \
    --strategy-ids ma_5_20
```

---

## 技术支持

- 遇到问题查看日志
- 检查 `data/risk_events.db`
- 查看 `docs/` 目录下的文档

---

**祝交易顺利！** 🚀
