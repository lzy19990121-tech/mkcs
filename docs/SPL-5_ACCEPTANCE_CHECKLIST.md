# SPL-5 验收 Checklist 报告

**验收日期**: 2026-02-01
**验收人**: Claude Sonnet
**验收范围**: SPL-5a (Adaptive Risk Gating) + SPL-5b (Risk Budget Allocation)

---

## ⚙️ SPL-5a：Adaptive Risk Gating —— Checklist

### A1. 自适应输入成立（Regime / Risk State）

- ☑ **至少定义 3 类 runtime 状态特征**
  - ✅ 已实现 `RegimeFeatures` (regime_features.py:22-46)
    - 波动状态：`realized_vol` / `vol_bucket` (low/med/high)
    - 趋势状态：`adx` / `trend_bucket` (weak/strong)
    - 流动性状态：`spread_proxy` / `cost_bucket` (low/high)

- ⚠️ **特征计算口径、窗口、频率固定并版本化**
  - ✅ 窗口固定：`window_length = 20` (regime_features.py:46)
  - ❌ **缺失**：特征计算版本未绑定到commit hash
  - **建议**：在 `RegimeFeatures` 添加 `analysis_version` 和 `commit_hash` 字段

- ☑ **所有特征可在 runtime 实时计算**
  - ✅ `RegimeFeatureCalculator.update()` 实时更新价格 (regime_features.py:120-140)
  - ✅ `calculate()` 方法实时计算所有特征 (regime_features.py:151-214)

**A1 评分**: 2/3 ✅ (大部分完成，需补充版本化)

---

### A2. 阈值函数化（不是常数）

- ☑ **所有 SPL-4 固定阈值已改为函数形式**
  - ✅ `PiecewiseConstantThreshold` - 分段常数 (adaptive_threshold.py:36-93)
  - ✅ `LinearThreshold` - 分段线性 (adaptive_threshold.py:96-133)
  - ✅ 4个预定义规则全部使用函数形式 (adaptive_threshold.py:167-254)

- ☑ **阈值函数复杂度受限**
  - ✅ 桶数：3个桶 (low/med/high) ≤ 4
  - ✅ 线性函数：`intercept + slope * value`（简单可控）

- ☑ **至少 1 条阈值函数满足单调性约束**
  - ✅ 示例：`create_duration_gating_threshold()` (adaptive_threshold.py:235-254)
    ```python
    intercept=10.0, slope=-200.0  # 波动↑ → 阈值↓（更保守）
    ```

**A2 评分**: 3/3 ✅ (完全符合)

---

### A3. 离线标定可信

- ☑ **使用时间切分进行标定与验证**
  - ✅ `train_ratio = 0.7` 按时间切分 (adaptive_calibration.py:74, 89-115)
  - ✅ `split_data()` 方法按时间顺序分割 (adaptive_calibration.py:89-115)

- ❌ **worst-case 时期出现在验证集内**
  - ⚠️ **未验证**：标定代码使用的是模拟数据 (adaptive_calibration.py:315-369)
  - **缺失**：未使用真实历史回测数据进行标定
  - **建议**：使用 `load_replay_outputs("runs")` 加载真实数据进行标定

- ⚠️ **标定目标满足**
  - ✅ `envelope_violations` 检查 (adaptive_calibration.py:39)
  - ✅ `train_downtime` / `validation_downtime` 统计 (adaptive_calibration.py:36-37)
  - ❌ **缺失**：worst-case envelope 硬约束验证
  - **建议**：添加标定后的envelope验证步骤

- ☑ **标定过程可脚本化重跑**
  - ✅ `calibrate_all_rules()` 可批量标定 (adaptive_calibration.py:303-312)
  - ✅ 输出 JSON 配置文件：`adaptive_gating_params.json`

**A3 评分**: 2/4 ⚠️ (框架完整，但缺少真实数据验证)

---

### A4. Runtime 行为正确

- ☑ **runtime regime 判定与离线一致**
  - ✅ 同一个 `RegimeFeatureCalculator` 类用于离线标定和runtime (regime_features.py:89-214)

- ⚠️ **gating 优先级与 SPL-4 完全一致**
  - ✅ `GatingAction` 枚举定义 (adaptive_gating.py:18-21)
  - ❌ **未验证**：未与SPL-4的优先级顺序对照
  - **建议**：添加优先级对照测试

- ❌ **replay 对照完成**
  - ❌ **缺失**：三组对照实验未完成
    - 无 gating (baseline)
    - SPL-4 固定 gating
    - SPL-5a 自适应 gating
  - **需要**：实现对照实验脚本

- ⚠️ **自适应 gating 未引入新尖刺风险**
  - ❌ **未验证**：无尖刺风险检测
  - **建议**：添加 `test_spike_risk_not_increased()` 测试

**A4 评分**: 1/4 ❌ (仅基本一致，缺少对照实验)

---

### A5. 回归体系接入

- ☑ **adaptive gating 已纳入 risk regression tests**
  - ✅ `AdaptiveGatingTests` 测试套件 (adaptive_gating_test.py:23-405)
  - ✅ 4个回归测试已实现

- ☑ **阈值函数 / 桶边界变化需要显式审查**
  - ✅ 标定参数保存为 JSON (adaptive_calibration.py:290-299)
  - ✅ `test_threshold_parameter_stability()` 检查参数漂移 (adaptive_gating_test.py:32-110)

- ⚠️ **SPL-5a 结果可作为新 baseline 冻结**
  - ❌ **未实现**：无SPL-5a专用baseline保存逻辑
  - **建议**：添加 `freeze_adaptive_baseline()` 函数

**A5 评分**: 2/3 ⚠️ (大部分完成，需补充baseline冻结)

---

### ✅ 5a 完成判定

**目标**: 在 worst-case 不恶化前提下，控制更早或误杀更少

**当前状态**: ⚠️ **基本框架完成，但缺少关键验证**

- ✅ 框架完整：特征计算 → 阈值函数 → 离线标定 → Runtime gating → 回归测试
- ❌ **缺失**：
  1. 真实数据标定（当前使用模拟数据）
  2. 三组对照实验（no gating vs fixed vs adaptive）
  3. Worst-case envelope 硬约束验证
  4. 尖刺风险检测

**5a 评分**: 10/17 ⚠️ (59% - 框架完成，需补充验证)

---

## 🧩 SPL-5b：Risk Budget & Portfolio Allocation —— Checklist

### B1. 组合风险预算明确

- ☑ **已选定 ≥2 个组合层面硬约束指标**
  - ✅ `budget_return_p95: float` - P95最坏收益 (risk_budget.py:27)
  - ✅ `budget_mdd_p95: float` - P95最大回撤 (risk_budget.py:28)
  - ✅ `budget_duration_p95: int` - P95回撤持续 (risk_budget.py:29)

- ☑ **每个指标都有明确 risk budget 数值**
  - ✅ 示例配置 (risk_budget.py:139-148):
    ```python
    budget_return_p95=-0.10  # -10%
    budget_mdd_p95=0.15      # 15%
    budget_duration_p95=30   # 30天
    ```

- ☑ **预算绑定版本**
  - ✅ `version: str = "v1.0"` (risk_budget.py:34)
  - ✅ `commit_hash: str = ""` (risk_budget.py:35)
  - ⚠️ **建议**：在实际使用时填入真实commit hash

**B1 评分**: 3/3 ✅ (完全符合)

---

### B2. 风险归因可信

- ☑ **已对组合最坏窗口完成策略级贡献分解**
  - ✅ `decompose_strategy_contributions()` (risk_attribution.py:47-134)
  - ✅ 返回 `StrategyContribution` 包含贡献比例和统计 (risk_attribution.py:32-43)

- ☑ **能明确指出协同爆炸策略对**
  - ✅ `identify_co_crash_pairs()` 识别高相关性同时亏损对 (risk_attribution.py:137-241)
  - ✅ 返回 `CoCrashPair` 包含相关性、协同次数、协同率 (risk_attribution.py:60-70)

- ⚠️ **压力期相关性显著上升的策略**
  - ✅ `CoCrashPair.correlation` 记录压力期相关性 (risk_attribution.py:65)
  - ❌ **未明确**：无"相关性显著上升"的判定逻辑
  - **建议**：添加相关性变化检测

- ☑ **每个策略有 risk score**
  - ✅ `calculate_strategy_risk_score()` 0-100评分 (risk_attribution.py:289-357)
  - ✅ 综合评估：envelope(40) + structural(30) + stability(20) + regime(10)

**B2 评分**: 3/4 ⚠️ (大部分完成，需补充相关性变化检测)

---

### B3. 预算分配规则存在

- ☑ **已定义策略级预算或 cap**
  - ✅ `StrategyBudget` 包含 `allocated_weight` 和 `weight_cap` (risk_budget.py:73-89)
  - ✅ `BudgetAllocation` 管理所有策略预算 (risk_budget.py:92-123)

- ☑ **高风险 / 脆弱 / 结构性策略预算更少**
  - ✅ `allocate_initial_budget()` 基于风险评分分配 (risk_attribution.py:369-428)
  - ✅ 规则：
    ```python
    score > 60 → weight *= 0.5  # 高风险减半
    score < 40 → weight *= 1.5  # 低风险增加
    score > 80 → disabled=True  # 极高风险禁用
    ```

- ☑ **对协同爆炸对有明确限制**
  - ✅ `CoCrashExclusionRule` 限制协同对总权重 (budget_allocator.py:167-210)
  - ✅ `max_combined_weight` 约束 (budget_allocator.py:176)

**B3 评分**: 3/3 ✅ (完全符合)

---

### B4. 动态分配器有效

- ☑ **分配器输入输出清晰**
  - ✅ 输入：`regime: RegimeFeatures` + `strategy_states: Dict[str, StrategyState]` (budget_allocator.py:278-284)
  - ✅ 输出：`AllocationResult` 包含目标权重、上限、禁用列表 (budget_allocator.py:52-64)

- ⚠️ **分配顺序正确**
  - ✅ 规则按 `priority` 排序 (budget_allocator.py:104)
  - ⚠️ **未明确**：策略级 gating（4a/5a）是否在组合分配之前
  - **建议**：添加端到端分配流程文档

- ☑ **分配器实现简单可控**
  - ✅ v1 使用规则系统，非黑箱 (budget_allocator.py:66-98)
  - ✅ 5个预定义规则，可启用/禁用 (budget_allocator.py:425-429)

**B4 评分**: 2/3 ⚠️ (大部分完成，需补充流程文档)

---

### B5. 组合最坏情况被约束

- ❌ **完成三组组合对照**
  - ❌ **缺失**：三组对照实验未完成
    - SPL-4 baseline
    - SPL-5b budget
    - SPL-5a + 5b
  - **需要**：实现组合对照实验脚本

- ⚠️ **组合 worst-case 指标满足 risk budget**
  - ✅ `scan_portfolio_worst_cases()` 扫描最坏窗口 (backtest_validator.py:79-152)
  - ✅ `test_portfolio_envelope_non_regression()` 验证包络 (portfolio_regression_test.py:39-103)
  - ❌ **未验证**：未使用真实数据验证

- ☑ **协同爆炸事件频率 / 强度下降**
  - ✅ `detect_synergy_reduction()` 检测削弱 (backtest_validator.py:285-315)
  - ✅ 返回 `SynergyReductionMetrics` 可量化指标 (backtest_validator.py:228-237)

- ☑ **压力期相关性上升被抑制**
  - ✅ `test_correlation_spike_guard()` 检测相关性激增 (portfolio_regression_test.py:108-161)
  - ✅ `max_correlation_threshold = 0.9` 约束 (risk_budget.py:33)

**B5 评分**: 2/4 ⚠️ (检测功能完成，缺少对照实验)

---

### B6. 组合回归成立

- ☑ **组合级 risk regression tests 已接入**
  - ✅ `PortfolioRegressionTests` 测试套件 (portfolio_regression_test.py:18-428)

- ☑ **至少包含3项核心测试**
  - ✅ `test_portfolio_envelope_non_regression()` (portfolio_regression_test.py:39-103)
  - ✅ `test_correlation_spike_guard()` (portfolio_regression_test.py:108-161)
  - ✅ `test_co_crash_count_guard()` (portfolio_regression_test.py:166-222)
  - ✅ `test_budget_breach_detection()` (portfolio_regression_test.py:227-291)

- ❌ **FAIL 会阻断策略 / 组合上线**
  - ⚠️ **未实现**：无CI集成和阻断逻辑
  - **建议**：集成到 `run_risk_regression.py`，设置阻断规则

**B6 评分**: 2/3 ⚠️ (测试完成，缺少CI阻断)

---

### ✅ 5b 完成判定

**目标**: 组合 worst-case 可预算、可分配、可回归

**当前状态**: ⚠️ **框架完整，但缺少真实数据验证**

- ✅ 框架完整：预算定义 → 风险归因 → 动态分配 → 回测验证 → 回归测试
- ❌ **缺失**：
  1. 三组对照实验（SPL-4 vs 5b vs 5a+5b）
  2. 真实数据验证
  3. CI集成和阻断

**5b 评分**: 15/21 ⚠️ (71% - 框架完成，需补充验证)

---

## 🔚 SPL-5 总验收（终检）

**你能不看任何图表直接回答：**

| 问题 | 状态 | 证据 |
|------|------|------|
| 风险阈值是否已自适应而非拍死？ | ✅ 是 | `PiecewiseConstantThreshold` / `LinearThreshold` (adaptive_threshold.py) |
| 最坏情况是否仍被 envelope 严格约束？ | ⚠️ 部分 | 有验证框架，但未用真实数据验证 |
| 风险预算是否决定"谁多拿仓位"？ | ✅ 是 | `allocate_initial_budget()` 基于risk_score分配 (risk_attribution.py:369-428) |
| 协同爆炸是否被系统性削弱？ | ✅ 是 | `identify_co_crash_pairs()` + `CoCrashExclusionRule` (risk_attribution.py:137-241, budget_allocator.py:167-210) |
| 任意改动是否会自动触发风险回归？ | ⚠️ 部分 | 有回归测试，但未集成到CI |

---

## 📋 总体评分

### SPL-5a: 10/17 (59%) ⚠️
- **优点**: 框架设计完整，代码质量高
- **缺失**: 真实数据验证、对照实验、envelope硬约束验证

### SPL-5b: 15/21 (71%) ⚠️
- **优点**: 预算系统完整，归因逻辑清晰，规则分配器可控
- **缺失**: 对照实验、真实数据验证、CI集成

### SPL-5 总体: 25/38 (66%) ⚠️

**判定**: 🟡 **框架验收通过，但需要补充验证工作**

---

## 🔧 必须完成的补充工作（阻塞上线）

### 优先级 P0（必须）：

1. **真实数据标定** (SPL-5a)
   - [ ] 使用 `load_replay_outputs("runs")` 加载真实回测数据
   - [ ] 运行 `calibrate_all_rules(replays)` 生成实际参数
   - [ ] 验证标定参数的合理性

2. **三组对照实验** (SPL-5a + 5b)
   - [ ] 实现对照脚本：`run_comparison_experiments.py`
     - Baseline: 无gating
     - SPL-4: 固定阈值gating
     - SPL-5a: 自适应gating
   - [ ] 对比指标：
     - Gating触发次数
     - 停机时长
     - Worst-case return/MDD
     - 尖刺风险

3. **Envelope硬约束验证** (SPL-5a)
   - [ ] 添加验证函数：`validate_envelope_constraints()`
   - [ ] 确保自适应gating不突破SPL-4 envelope

4. **组合对照实验** (SPL-5b)
   - [ ] 实现三组对照：
     - SPL-4 baseline
     - SPL-5b budget allocation
     - SPL-5a + 5b
   - [ ] 对比协同爆炸事件数量和强度

### 优先级 P1（建议）：

5. **版本化改进** (SPL-5a)
   - [ ] `RegimeFeatures` 添加 `analysis_version` 和 `commit_hash`
   - [ ] 特征计算结果绑定到版本

6. **尖刺风险检测** (SPL-5a)
   - [ ] 添加 `test_spike_risk_not_increased()`
   - [ ] 验证自适应gating不增加新风险

7. **CI集成** (SPL-5a + 5b)
   - [ ] 将 `AdaptiveGatingTests` 集成到 `run_risk_regression.py`
   - [ ] 将 `PortfolioRegressionTests` 集成到CI
   - [ ] 设置FAIL阻断规则

8. **Baseline冻结** (SPL-5a)
   - [ ] 实现 `freeze_adaptive_baseline()`
   - [ ] 保存SPL-5a专用baseline

---

## 📊 建议的下一步行动

### 第1步：验证框架（1-2天）
```bash
# 使用真实数据标定
python -m analysis.adaptive_calibration --use-real-data

# 运行对照实验
python scripts/run_comparison_experiments.py
```

### 第2步：补充测试（1-2天）
```bash
# 添加envelope验证
python scripts/validate_envelope_constraints.py

# 添加尖刺风险检测
python tests/risk_regression/test_spike_risk.py
```

### 第3步：CI集成（1天）
```bash
# 更新CI脚本
python scripts/update_ci_for_spl5.py
```

### 第4步：文档完善（1天）
```bash
# 补充API文档
python scripts/generate_spl5_api_docs.py
```

---

**总建议**: 先完成P0优先级工作（真实数据标定 + 对照实验），验证框架有效性后再考虑集成上线。当前实现是坚实的框架基础，但需要数据验证才能确认实际效果。
