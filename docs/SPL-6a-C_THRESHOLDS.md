# SPL-6a-C: 漂移阈值与分级响应 - 完成报告

**完成日期**: 2026-02-01
**状态**: ✅ 已完成

---

## 📋 任务完成情况

### 漂移阈值配置（`config/drift_thresholds.yaml`）

#### 1️⃣ 分布型指标阈值
| 指标 | GREEN | YELLOW | CRITICAL |
|------|-------|--------|----------|
| PSI | < 0.1 | 0.1 - 0.25 | - |
| JS divergence | < 0.1 | 0.1 - 0.2 | - |
| Bucket shift | < 5% | 5% - 10% | - |
| KS test | < 0.1 | 0.1 - 0.2 | - |
| Wasserstein | < 0.1 | 0.1 - 0.2 | - |

#### 2️⃣ 统计型指标阈值
| 指标 | GREEN | YELLOW | CRITICAL |
|------|-------|--------|----------|
| Percentile shift | < 2% | 2% - 5% | - |
| Tail change | < 1% | 1% - 3% | - |
| Absolute change | < 0.01 | 0.01 - 0.02 | - |
| Relative change | < 10% | 10% - 20% | - |
| Max drawdown | < 1% | 1% - 2% | - |
| CVaR change | < 2% | 2% - 5% | - |

#### 3️⃣ 行为指标阈值
| 指标 | GREEN | YELLOW | CRITICAL |
|------|-------|--------|----------|
| Gating trigger rate | < 5% | 5% - 10% | > 30% |
| Avg downtime | < 1天 | 1 - 2天 | - |
| Cap hit rate | < 5% | 5% - 10% | - |
| Regime switch | < 0.5次/天 | 0.5 - 1次/天 | - |

#### 4️⃣ 组合指标阈值
| 指标 | GREEN | YELLOW | CRITICAL |
|------|-------|--------|----------|
| Portfolio correlation | < 0.05 | 0.05 - 0.1 | > 0.3 |
| Co-crash frequency | < 0.5 | 0.5 - 1.0 | - |
| Max simultaneous | 不变 | +1 | - |

---

## 🚨 再标定触发条件

### 基本条件（满足任一即触发）

#### 条件1: ��续 RED 检测
- **配置**: 连续 3 次检测到 RED（在 7 天内）
- **适用**: 所有漂移对象
- **触发**: 再标定候选

#### 条件2: 关键风险指标退化
- **Max drawdown**: 恶化 > 5%
- **CVaR**: 恶化 > 10%
- **Worst-case returns**: 恶化 > 10%

#### 条件3: 多个对象同时 YELLOW
- **数量阈值**: >= 5 个对象
- **比例阈值**: 占总对象数 30%

#### 条件4: 组合协同爆炸
- **Co-crash count**: >= 3 个策略同时亏损
- **Frequency**: 最近 30 天内发生 >= 2 次

### 保护机制（防止频繁触发）

- **最小间隔**: 两次再标定之间最少 30 天
- **冷却期**: 再标定后 60 天内不再触发
- **回滚要求**: 必须有 baseline 可以回滚

---

## 🎯 分级响应策略

### GREEN 状态
```yaml
action: none
monitoring: normal
message: "No significant drift detected"
```

### YELLOW 状态
```yaml
action: alert
monitoring: enhanced
message: "Minor drift detected - monitoring closely"
notifications:
  - log (warning level)
  - daily report
```

### RED 状态
```yaml
action: candidate_for_recalibration
monitoring: intensive
message: "Significant drift detected - evaluating recalibration"
notifications:
  - log (error level)
  - Slack alert
  - realtime report
triggers:
  - create_recalibration_ticket
  - run_candidate_evaluation
```

### CRITICAL 状态
```yaml
action: immediate_investigation
monitoring: continuous
message: "Critical drift - may require emergency intervention"
notifications:
  - Slack (high priority)
  - Email to risk-team
triggers:
  - create_emergency_ticket
  - disable_affected_strategies
```

---

## 📊 对象优先级分级

### Critical（5个对象）
- `risk_behavior.max_drawdown`
- `risk_behavior.cvar`
- `risk_behavior.worst_case_returns`
- `portfolio_behavior.co_crash_frequency`
- `portfolio_behavior.max_simultaneous_losses`

### High（4个对象）
- `input_distribution.volatility`
- `input_distribution.returns`
- `model_behavior.gating_trigger_rate`
- `portfolio_behavior.portfolio_correlation`

### Medium（4个对象）
- `input_distribution.adx`
- `input_distribution.spread_cost`
- `risk_behavior.spike_metrics`
- `model_behavior.avg_downtime`

### Low（4个对象）
- `model_behavior.cap_hit_rate`
- `model_behavior.regime_switch_frequency`
- `portfolio_behavior.correlation_spike_frequency`

---

## 📁 创建的文件

### 1. `config/drift_thresholds.yaml`
- 17 个对象的阈值配置
- 再标定触发条件
- 分级响应策略
- 对象优先级分级
- 检测频率配置

### 2. `analysis/drift_threshold_evaluator.py`
核心类：

#### `DriftThresholdEvaluator`
- 加载阈值配置
- 评估漂移结果（添加状态和消息）
- 检查再标定触发条件
- 获取响应行动
- 获取对象优先级

#### `DriftReportGenerator`
- 生成汇总报告
- 按优先级分组统计
- 识别关键漂移
- 判断总体状态

---

## 🎯 关键设计决策

### 1. 配置驱动
- 所有阈值在 YAML 中定义
- 易于调整和维护
- 支持对象级别的覆盖

### 2. 多条件触发
- 4 个独立的触发条件
- 保护机制防止频繁触发
- 冷却期和回滚要求

### 3. 优先级分级
- Critical 对象优先处理
- 资源分配有侧重
- 报告中高亮显示

### 4. 响应行动明确
- 每个状态对应清晰行动
- 自动化触发通知
- 渐进式升级策略

---

## 🚀 下一步：SPL-6a-D

需要实现受控再标定流程：
1. 实现再标定流程脚本（从 runs/ 自动取样本）
2. 数据 eligibility 过滤（复用 SPL-4c）
3. 时间切分 train/valid（不允许随机打散）
4. 候选参数评估（三组对照 + gates）
5. 生成待审查 artifact（不直接覆盖 baseline）

---

**生成时间**: 2026-02-01
**SPL-6a 进度**: 3/5 (60%) ✅
