# MKCS 更新日志

## 2026-02-10 - 混合实时数据源 + SPL-7a 修复

### 新增功能

#### 1. 混合实时数据源 (`skills/market_data/hybrid_realtime_source.py`)
- **Yahoo Finance** - 历史K线数据（用于策略计算）
- **Finnhub** - 实时报价 + WebSocket 推送（用于快速交易）
- 更新频率：WebSocket 实时推送（1秒级别）+ API 检查（2秒间隔）

**优势：**
| 特性 | Yahoo Finance | Finnhub |
|------|--------------|---------|
| 历史数据 | ✅ 完整 | ❌ 需付费 |
| 实时报价 | ⚠️ 有延时 | ✅ 无延时 |
| WebSocket | ❌ | ✅ |
| 频率限制 | 宽松 | 60次/分钟 |

**使用方式：**
```python
from skills.market_data.hybrid_realtime_source import HybridRealtimeSource

source = HybridRealtimeSource(
    use_finnhub=True,
    check_interval=2.0,  # API 调用间隔（秒）
    enable_websocket=True
)
```

#### 2. 更新 `run_live_with_monitoring.py`
- 使用混合数据源
- 新增 `--check-interval` 参数（默认2秒，最快1秒）
- 自动启动/停止 WebSocket 实时数据流

### Bug 修复

#### 1. 时区问题 (`agent/live_runner.py`)
- 修复 `_is_trading_hours()` 使用本地时间判断交易时段的问题
- 修复 `_wait_for_market_open()` 时区转换
- 现在正确支持非美国时区（如北京时间）

#### 2. RunContext 参数不匹配
- 修正 `timestamp` → `now`
- 添加缺失的 `trading_date` 和 `bar_interval` 参数

#### 3. Broker 方法名
- `get_all_positions()` → `get_positions()`

#### 4. datetime 时区感知比较 (`skills/market_data/yahoo_source.py`)
- 添加 `_datetime_in_range()` 方法处理时区感知/无时区 datetime 的比较

#### 5. SPL-7a TrendAlert 属性 (`analysis/online/risk_monitor.py`)
- 修复 `alert.slope` → `alert.magnitude.value`
- 修复 `alert.r_squared` → 使用新属性结构

#### 6. RiskEventStore 方法名
- `store_event()` → `write_event()`

#### 7. AlertingManager 未初始化属性
- 添加 `_current_links` 初始化

### 配置变更

#### 新增依赖
```
pytz>=2023.3
```

#### 环境变量 (`.env`)
```bash
# Finnhub API Key (免费注册: https://finnhub.io/register)
FINNHUB_API_KEY=d612bn9r01qjrrugmfp0d612bn9r01qjrrugmfpg
```

### 使用示例

```bash
# 启动实时交易（2秒检查间隔）
python scripts/run_live_with_monitoring.py \
    --symbols AAPL MSFT \
    --interval 5m \
    --check-interval 2

# 启动实时交易（1秒检查间隔，最快）
python scripts/run_live_with_monitoring.py \
    --symbols AAPL \
    --interval 1m \
    --check-interval 1
```

### 参数说明

| 参数 | 含义 | 默认值 | 范围 |
|------|------|--------|------|
| `--interval` | K线周期 | `5m` | `1m`, `5m`, `1h`, `1d` |
| `--check-interval` | 交易检查间隔（秒） | `2` | `1-60` |

### 频率限制说明

- **Finnhub 免费版**: 60次/分钟
- **推荐设置**:
  - 快速交易: `check-interval=1` (1秒)
  - 正常交易: `check-interval=2` (2秒)
  - 低频交易: `check-interval=5` (5秒)

### 相关提交

- `b9e1053` - Feat: 添加混合实时数据源 (Yahoo + Finnhub)
- `912ab73` - Fix: SPL-7a TrendAlert 属性性和 RiskEventStore 方法名
