# Online Risk Events

**文档版本**: 1.0
**最后更新**: 2026-02-01
**所属模块**: SPL-7a (Online Monitoring)

---

## 📋 概述

本文档定义运行态风险事件的数据结构和存储规范，作为在线监控与离线分析之间的桥梁。

---

## 🎯 设计目标

1. **统一数据格式**: 所有风险事件使用统一的数据结构
2. **持久化存储**: 事件可被后续分析复用（SPL-7b 反事实、SPL-6a 漂移��测）
3. **高效查询**: 支持按时间、类型、策略查询
4. **可扩展性**: 支持新事件类型的添加

---

## 📊 事件类型

### 1. 风险信号事件 (RISK_SIGNAL)

**触发频率**: 每次风险信号更新时

**内容**:
- 滚动收益（1d/5d/20d/60d）
- 回撤指标（当前/最大/持续）
- Spike 指标（最大单步亏损、连续亏损）
- 稳定性指标（波动率、稳定性评分）
- 市场状态特征（volatility bucket, trend strength）
- 最近的 gating/allocator 事件

**用途**:
- SPL-6a: 漂移检测的输入数据
- SPL-7b: 反事实分析的基础数据

**Schema**: 见 `analysis/online/risk_signal_schema.py:RiskSignal`

---

### 2. 状态转换事件 (STATE_TRANSITION)

**触发频率**: 风险状态发生变化时

**内容**:
- 状态转换（from → to）
- 转换类型（upgrade/downgrade/hold）
- 触发指标
- 触发值和阈值
- 上下文（完整的风险信号）

**用途**:
- SPL-7a: Post-mortem 触发条件
- SPL-7b: 反事实场景选择

**Schema**: 见 `analysis/online/risk_state_machine.py:StateTransitionEvent`

---

### 3. 趋势告警事件 (TREND_ALERT)

**触发频率**: 趋势检测器发现异常时

**内容**:
- 告警类型（趋势上升/下降、激增）
- 告警严重程度（info/warning/critical）
- 趋势统计（斜率、拟合优度、增长率）
- 建议措施

**用途**:
- SPL-7a: 早期预警
- SPL-6a: 漂移早期信号

**Schema**: 见 `analysis/online/trend_detector.py:TrendAlert`

---

### 4. Gating 事件 (GATING_EVENT)

**触发频率**: 每次 gating 动作时

**内容**:
- Gating 动作（ALLOW/GATE/REDUCE/DISABLE）
- 触发规则
- 阈值和当前值
- 市场状态快照

**用途**:
- SPL-7a: Post-mortem 触发条件
- SPL-7b: 反事实的 gating 场景

**Schema**: 见 `analysis/online/risk_signal_schema.py:GatingEvent`

---

### 5. Allocator 事件 (ALLOCATOR_EVENT)

**触发频率**: 每次 allocator 重平衡或 fallback 时

**内容**:
- Allocator 类型（rules/optimizer_v2）
- 动作（rebalance/cap_hit/fallback）
- 权重变化
- 触发约束

**用途**:
- SPL-7a: Allocator 行为监控
- SPL-7b: 反事实的 allocator 对比

**Schema**: 见 `analysis/online/risk_signal_schema.py:AllocatorEvent`

---

### 6. Post-mortem 报告事件 (POST_MORTEM)

**触发频率**: 发生重大风险事件后

**内容**:
- 触发类型和事件
- 指标轨迹
- 根本原因分析
- 建议

**用途**:
- SPL-7a: 风险事件归档
- SPL-7b: 反事实分析的场景选择
- SPL-6a: 再标定的输入

**Schema**: 见 `analysis/online/postmortem_generator.py:PostMortemReport`

---

## 💾 存储架构

### 数据库

**实现**: SQLite
**路径**: `data/risk_events.db`
**表**: `risk_events`

**Schema**:
```sql
CREATE TABLE risk_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    data TEXT NOT NULL,              -- JSON 序列化的事件数据
    data_version TEXT,                -- Schema 版本
    source TEXT,                      -- 数据源
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_event_type_timestamp ON risk_events(event_type, timestamp);
CREATE INDEX idx_strategy_timestamp ON risk_events(strategy_id, timestamp);
```

### 写入策略

1. **实时写入**: 关键事件（state_transition, gating_event）立即写入
2. **批量写入**: 高频事件（risk_signal）批量写入（100 条/批）
3. **WAL 模式**: 启用 Write-Ahead Logging，提高并发性能

### 数据保留

**默认保留期**: 1 年
**归档策略**: 超过保留期的数据导出到 JSON 归档

---

## 🔍 查询接口

### 按事件类型查询

```python
event_store = RiskEventStore()

# 查询所有状态转换事件
events = event_store.query_events(
    event_type=EventType.STATE_TRANSITION,
    limit=100
)
```

### 按时间范围查询

```python
# 查询最近 7 天的事件
start_time = datetime.now() - timedelta(days=7)
events = event_store.query_events(
    start_time=start_time,
    limit=1000
)
```

### 按策略查询

```python
# 查询特定策略的事件
events = event_store.query_events(
    strategy_id="strategy_1",
    start_time=start_time,
    end_time=end_time
)
```

### 反事实分析查询

```python
# 获取事件上下文（用于反事实）
context = event_store.get_events_for_counterfactual(
    strategy_id="strategy_1",
    event_id="pm_gate_20260201_120000",
    context_window_hours=24
)

# context 包含：
# - target_event: 目标事件
# - context_events: 上下文窗口内的所有事件
# - window_start/end: 时间窗口
```

---

## 🔄 与其他模块的集成

### SPL-6a: Drift Detection

**输入**: 在线事件（主要是 RISK_SIGNAL）

**用途**:
- 作为 drift detection 的输入数据
- 计算统计指标（PSI, KS test, etc.）
- 识别分布变化

**频率**: 每日/每周

### SPL-7b: Counterfactual Analysis

**输入**: 在线事件 + Replay 数据

**用途**:
- 选择反事实分析的事件
- 提取事件上下文
- 比较实际 vs 反事实结果

**频率**: 按需（重大风险事件后）

---

## 📈 统计与监控

### 事件统计

```python
stats = event_store.get_statistics()

# 返回：
{
#     "total_events": 15234,
#     "by_type": {
#         "risk_signal": 12000,
#         "state_transition": 234,
#         "trend_alert": 123,
#         "gating_event": 56,
#         "allocator_event": 78,
#         "post_mortem": 12
#     },
#     "by_strategy": {
#         "strategy_1": 5123,
#         "strategy_2": 4987,
#         "strategy_3": 5124
#     },
#     "time_range": {
#         "earliest": "2026-01-01T00:00:00",
#         "latest": "2026-02-01T12:00:00"
#     }
# }
```

### 监控指标

- **写入速率**: 事件/秒
- **存储大小**: MB
- **查询延迟**: ms
- **失败率**: %

---

## 🚀 性能优化

### 索引策略

1. **复合索引**: `(event_type, timestamp)` 支持按类型和时间查询
2. **策略索引**: `(strategy_id, timestamp)` 支持按策略查询

### 批量写入

```python
# 批量写入提高性能
events = [RiskEvent.from_signal(s) for s in signals]
count = event_store.write_events_batch(events)
```

### 数据清理

```python
# 清理超过保留期的数据
cutoff_time = datetime.now() - timedelta(days=365)
# 导出到归档
event_store.export_to_json(
    "archive/risk_events_old.json",
    end_time=cutoff_time
)
# 然后删除（TODO: 实现）
```

---

## 📝 版本控制

### Schema 版本

**当前版本**: 1.0

**向后兼容性**:
- 1.0 → 1.0: 完全兼容
- 未来版本需要考虑迁移策略

### 字段变更

重大变更需要记录在 CHANGELOG 中：

```
## [1.1] - 2026-XX-XX
### Added
- 新增字段 `correlation_metrics` 到 RISK_SIGNAL

### Changed
- `stability_score` 类型从 INT 改为 FLOAT

### Deprecated
- `old_field` 将在 2.0 中移除
```

---

## 🔗 相关文档

- **Post-mortem Template**: `docs/POST_MORTEM_TEMPLATE.md`
- **Alerting Rules**: `config/alerting_rules.yaml`
- **Online Metrics**: `config/online_metrics.yaml`
- **Schema Definition**: `analysis/online/risk_signal_schema.py`

---

**生成时间**: 2026-02-01
**维护者**: SPL-7 Team
