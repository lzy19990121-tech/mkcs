# SPL-6a-A: 漂移对象定义 - 完成报告

**完成日期**: 2026-02-01
**状态**: ✅ 已完成

---

## 📋 任务完成情况

### 定义的漂移对象（17个）

#### 1️⃣ 输入分布漂移（4个对象）
| 对象名 | 描述 | 窗口 | 桶数量 |
|--------|------|------|--------|
| `returns` | 策略收益分布 | 30d/90d | 6桶 |
| `volatility` | 实现波动率 | 30d/90d | 4桶 |
| `adx` | 平均趋向指数 | 30d/90d | 3桶 |
| `spread_cost` | 价差/成本代理 | 30d/90d | 3桶 |

**统计口径**:
- **时间窗口**: 30天检测窗口 vs 90天基线窗口
- **分桶粒度**: 根据业务意义定义（如收益<-5%为large_loss）
- **指标**: PSI、JS divergence、bucket share shift

#### 2️⃣ 风险行为漂移（5个对象）
| 对象名 | 描述 | 窗口 | 指标 |
|--------|------|------|------|
| `worst_case_returns` | 最坏情况收益分布 | 30d/90d | percentile_shift |
| `cvar` | 条件风险价值 | 30d/90d | absolute/relative_change |
| `max_drawdown` | 最大回撤 | 30d/90d | threshold_breach |
| `drawdown_duration` | 回撤持续时长分布 | 30d/90d | tail_ratio_change |
| `spike_metrics` | 尖刺指标 | 30d/90d | max/consecutive_change |

#### 3️⃣ 模型/规则漂移（4个对象）
| 对象名 | 描述 | 窗口 | 指标 |
|--------|------|------|------|
| `gating_trigger_rate` | Gating触发频率 | 30d/90d | rate_breach |
| `avg_downtime` | 平均停机时长 | 30d/90d | relative_change |
| `cap_hit_rate` | 风险预算cap命中频率 | 30d/90d | threshold_breach |
| `regime_switch_frequency` | 市场状态切换频率 | 30d/90d | stability_check |

#### 4️⃣ 组合层面漂移（4个对象）
| 对象名 | 描述 | 窗口 | 指标 |
|--------|------|------|------|
| `portfolio_correlation` | 策略间平均相关性 | 30d/90d | tail_correlation_change |
| `co_crash_frequency` | 同时尾部亏损频率 | 30d/90d | threshold_breach |
| `correlation_spike_frequency` | 相关性激增事件频率 | 30d/90d | threshold_breach |
| `max_simultaneous_losses` | 最大同时亏损策略数 | 30d/90d | threshold_breach |

---

## 📁 创建的文件

### 1. `config/drift_objects.yaml`
- 漂移对象配置文件
- 定义了17个漂移对象的详细配置
- 包含分桶定义、时间窗口、指标类型

### 2. `analysis/drift_objects.py`
- 漂移对象数据结构定义
- 核心类：
  - `DriftObjectConfig`: 单个漂移对象配置
  - `DriftSnapshot`: 统计快照
  - `DriftResult`: 漂移检测结果
  - `DriftObjectRegistry`: 漂移对象注册表

---

## 🎯 关键设计决策

### 1. 时间窗口策略
- **检测窗口**: 30天（近期观察）
- **基线窗口**: 90天（历史参考）
- **最小样本量**: 100个数据点

### 2. 分桶一致性
```yaml
regime_buckets:
  volatility: ["low", "med", "high"]
  adx: ["weak", "strong"]
  spread_cost: ["low", "high"]
```
确保与 SPL-5a 的 regime buckets 保持一致。

### 3. 指标类型分类
- **分布型指标** (distribution): 需要分桶，使用 PSI、KS test
- **统计型指标** (statistical): 数值比较，使用 absolute/relative change

### 4. 扩展性设计
- 配置文件驱动，易于添加新对象
- 支持自定义桶定义和阈值
- 数据结构清晰，便于实现漂移检测算法

---

## 🚀 下一步：SPL-6a-B

需要实现漂移指标计算：
1. PSI (Population Stability Index)
2. KS test / Wasserstein distance
3. Bucket share shift
4. Percentile shift
5. Absolute/relative change
6. Threshold breach detection

---

**生成时间**: 2026-02-01
**SPL-6a 进度**: 1/5 (20%) ✅
