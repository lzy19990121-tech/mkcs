# Post-mortem Report Template

**Report ID**: `{{REPORT_ID}}`
**Strategy**: `{{STRATEGY_ID}}`
**Trigger Type**: `{{TRIGGER_TYPE}}`
**Generated**: `{{TIMESTAMP}}`

---

## 📋 Executive Summary

**What Happened**: `{{BRIEF_DESCRIPTION}}`

**Impact**:
- 最大回撤: `{{MAX_DRAWDOWN}}`
- Spike 次数: `{{SPIKE_COUNT}}`
- Gate 触发: `{{GATE_TRIGGERED}}`

---

## 🕒 Timeline

| Time | Event | Details |
|------|-------|---------|
| `{{TIMESTAMP_1}}` | `{{EVENT_1}}` | `{{DETAILS_1}}` |
| `{{TIMESTAMP_2}}` | `{{EVENT_2}}` | `{{DETAILS_2}}` |
| `{{TIMESTAMP_3}}` | **Trigger Event** | `{{TRIGGER_DETAILS}}` |

---

## 📊 Metrics Trajectory

### 收益曲线

```
收益随时间变化：
{{RETURN_PLOT_OR_TABLE}}
```

### 回撤曲线

```
回撤随时间变化：
{{DRAWDOWN_PLOT_OR_TABLE}}
```

### 关键指标

| 指标 | 触发前 | 触发时 | 触发后 | 变化 |
|------|--------|--------|--------|------|
| 收益 | `{{RETURN_BEFORE}}` | `{{RETURN_AT}}` | `{{RETURN_AFTER}}` | `{{RETURN_CHANGE}}` |
| 回撤 | `{{DD_BEFORE}}` | `{{DD_AT}}` | `{{DD_AFTER}}` | `{{DD_CHANGE}}` |
| 波动率 | `{{VOL_BEFORE}}` | `{{VOL_AT}}` | `{{VOL_AFTER}}` | `{{VOL_CHANGE}}` |
| Spike | `{{SPIKE_BEFORE}}` | `{{SPIKE_AT}}` | `{{SPIKE_AFTER}}` | `{{SPIKE_CHANGE}}` |

---

## 🎯 Triggered Rules

### 规则 1: `{{RULE_ID_1}}`

**描述**: `{{RULE_DESCRIPTION_1}}`

**触发条件**:
- 指标: `{{METRIC_NAME_1}}`
- 当前值: `{{CURRENT_VALUE_1}}`
- 阈值: `{{THRESHOLD_1}}`

**动作**: `{{ACTION_TAKEN_1}}`

### 规则 2: `{{RULE_ID_2}}`

**描述**: `{{RULE_DESCRIPTION_2}}`

**触发条件**:
- 指标: `{{METRIC_NAME_2}}`
- 当前值: `{{CURRENT_VALUE_2}}`
- 阈值: `{{THRESHOLD_2}}`

**动作**: `{{ACTION_TAKEN_2}}`

---

## 🌐 Market Regime at Trigger

**Volatility Bucket**: `{{VOLATILITY_BUCKET}}`
**Trend Strength**: `{{TREND_STRENGTH}}`
**Liquidity Level**: `{{LIQUIDITY_LEVEL}}`

**Market Context**:
```
{{MARKET_CONTEXT_DESCRIPTION}}
```

**Correlation with Market**:
- 市场下跌时策略表现: `{{CORRELATION_DESCRIPTION}}`
- 是否与市场协同: `{{CO_MOVEMENT}}`

---

## 🔍 Root Cause Analysis

### Primary Cause

**`{{PRIMARY_CAUSE}}`**

**Explanation**: `{{PRIMARY_CAUSE_EXPLANATION}}`

### Contributing Factors

1. **`{{FACTOR_1}}`**
   - 证据: `{{EVIDENCE_1}}`
   - 影响程度: `{{IMPACT_1}}`

2. **`{{FACTOR_2}}`**
   - 证据: `{{EVIDENCE_2}}`
   - 影响程度: `{{IMPACT_2}}`

3. **`{{FACTOR_3}}`**
   - 证据: `{{EVIDENCE_3}}`
   - 影响程度: `{{IMPACT_3}}`

### Causal Chain

```
{{ROOT_CAUSE}}
    ↓
{{INTERMEDIATE_CAUSE_1}}
    ↓
{{INTERMEDIATE_CAUSE_2}}
    ↓
{{TRIGGER_EVENT}}
```

---

## 💡 Recommendations

### Immediate Actions (Within 1 Hour)

- [ ] `{{IMMEDIATE_ACTION_1}}`
- [ ] `{{IMMEDIATE_ACTION_2}}`
- [ ] `{{IMMEDIATE_ACTION_3}}`

### Short-term Actions (Within 24 Hours)

- [ ] `{{SHORT_TERM_ACTION_1}}`
- [ ] `{{SHORT_TERM_ACTION_2}}`

### Long-term Actions (Within 1 Week)

- [ ] `{{LONG_TERM_ACTION_1}}`
- [ ] `{{LONG_TERM_ACTION_2}}`

---

## 📎 Attachments

1. **Replay Data**: `{{REPLAY_DATA_PATH}}`
2. **Signal Logs**: `{{SIGNAL_LOG_PATH}}`
3. **Metrics Export**: `{{METRICS_EXPORT_PATH}}`

---

## 🔄 Follow-up

**Next Review**: `{{NEXT_REVIEW_TIME}}`

**Follow-up Questions**:
1. `{{QUESTION_1}}`
2. `{{QUESTION_2}}`
3. `{{QUESTION_3}}`

---

## 📝 Analysis Notes

```
{{ANALYST_NOTES}}
```

---

**Report Version**: `{{VERSION}}`
**Analyst**: `{{ANALYST_NAME}}`
**Reviewers**: `{{REVIEWERS}}`

---

*此报告由 SPL-7a Post-mortem Generator 自动生成*
