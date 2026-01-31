# 工作状态与下一步计划

**日期**: 2026-02-01
**当前状态**: SPL-4 实现完成，待集成测试

---

## ✅ 已完成工作

### SPL-4: Risk Control & Portfolio Hardening

#### Phase C: 风险回归测试 ✅
- [x] 基线冻结系统 (`analysis/risk_baseline.py`, `analysis/baseline_manager.py`)
- [x] 5大回归测试 (`tests/risk_regression/risk_baseline_test.py`)
- [x] CI集成 (`tests/risk_regression/run_risk_regression.py`)
- [x] GitHub Actions配置 (`.github/workflows/risk_regression.yml`)
- [x] 基线存储结构 (`baselines/risk/`)

#### Phase A: 运行时风控 ✅
- [x] 实时风险指标计算 (`skills/risk/runtime_metrics.py`)
- [x] 风控规则执行引擎 (`skills/risk/risk_gate.py`)
- [x] Agent集成 (`agent/runner.py`)
- [x] 风控效果验证 (`tests/risk_regression/gating_verification.py`)

#### Phase B: 组合分析 ✅
- [x] 组合构建器 (`analysis/portfolio/portfolio_builder.py`)
- [x] 组合窗口扫描 (`analysis/portfolio/portfolio_scanner.py`)
- [x] 协同风险分析 (`analysis/portfolio/synergy_analyzer.py`)
- [x] 组合风险报告 (`analysis/portfolio/portfolio_risk_report.py`)

#### 文档 ✅
- [x] 实现文档 (`SPL-4_IMPLEMENTATION.md`)
- [x] 验收报告 (`SPL-4_ACCEPTANCE.md`)
- [x] 交付文档 (`SPL-4_DELIVERY_SUMMARY.md`)
- [x] README更新

### 验收结果
- **C阶段**: 10/11 通过 (1 SKIP)
- **A阶段**: 9/10 通过 (1 待验证)
- **B阶段**: 12/12 通过
- **终检**: 4/4 核心问题可答
- **总体**: ✅ PASS

---

## 📋 明天工作计划

### 优先级1: 生成实际基线数据 (30分钟)

**目标**: 为现有策略生成SPL-4c基线

```bash
# 1. 确认有运行数据
ls runs/

# 2. 冻结基线
PYTHONPATH=/home/neal/mkcs python -c "
from analysis.baseline_manager import BaselineManager
mgr = BaselineManager()
snapshot = mgr.freeze_baselines(
    replay_dir='runs',
    output_dir='baselines/risk',
    window_lengths=['20d', '60d']
)
print(f'✓ 已冻结 {len(snapshot.baselines)} 个基线')
"

# 3. 验证基线文件
cat baselines/risk/baseline_manifest.json
```

**预期结果**:
- `baselines/risk/baselines_v1.json` 包含实���策略基线数据
- 每个策略有worst_windows, risk_patterns, envelopes, rules

---

### 优先级2: 运行完整回归测试 (20分钟)

**目标**: 验证所有策略通过回归测试

```bash
# 运行回归测试
PYTHONPATH=/home/neal/mkcs python tests/risk_regression/run_risk_regression.py \
    --replay-dir runs \
    --baseline-dir baselines/risk \
    --output-dir reports/risk_regression

# 查看结果
cat reports/risk_regression/report.md
```

**预期结果**:
- 所有测试PASS或SKIP（无FAIL）
- 生成完整报告

---

### 优先级3: 测试Runtime Gating (40分钟)

**目标**: 验证风控在实际回测中有效

```bash
# 1. 选择一个策略测试
cd /home/neal/mkcs

# 2. 创建测试脚本
cat > test_gating.py << 'EOF'
from agent.runner import create_default_agent
from analysis.actionable_rules import RiskRuleset
from skills.risk.risk_gate import RiskGate
from analysis import load_replay_outputs
import json

# 加载规则
with open('runs/deep_analysis_v3b/exp_xxx_actionable_rules.json') as f:
    ruleset_data = json.load(f)
    ruleset = RiskRuleset.from_dict(ruleset_data)

# 创建agent
agent = create_default_agent()

# 添加风控
gate = RiskGate(ruleset)
agent.set_risk_gate(gate)

# 运行回测
agent.run_replay_backtest(...)

# 检查风控统计
stats = gate.get_statistics()
print(f"风控触发次数: {stats['gate_triggers']}")
print(f"触发率: {stats['trigger_rate']*100:.2f}%")
EOF

# 3. 运行测试
PYTHONPATH=/home/neal/mkcs python test_gating.py
```

**预期结果**:
- 风控在worst-case期间触发
- 回测成功完成
- 生成gate statistics

---

### 优先级4: 构建2策略组合 (30分钟)

**目标**: 测试组合分析功能

```bash
# 创建组合构建脚本
cat > test_portfolio.py << 'EOF'
from analysis.portfolio import PortfolioBuilder, PortfolioConfig
from analysis.portfolio import PortfolioWindowScanner, SynergyAnalyzer
from datetime import date

# 配置组合
config = PortfolioConfig(
    strategy_ids=["ma_5_20", "breakout"],  # 使用实际策略ID
    weights={"ma_5_20": 0.6, "breakout": 0.4},
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    alignment_method="inner"
)

# 构建组合
builder = PortfolioBuilder()
portfolio = builder.build_portfolio(config, replay_dir="runs")

# 扫描最坏窗口
scanner = PortfolioWindowScanner()
worst_windows = scanner.find_worst_portfolio_windows(
    portfolio,
    window_lengths=["20d", "60d"],
    top_k=5
)

# 协同分析
analyzer = SynergyAnalyzer()
synergy_report = analyzer.generate_synergy_report(
    portfolio,
    worst_windows,
    risk_budget=-0.10
)

# 生成报告
from analysis.portfolio import PortfolioRiskReportGenerator
generator = PortfolioRiskReportGenerator()
report = generator.generate_report(
    portfolio,
    worst_windows,
    synergy_report,
    output_path=Path("reports/portfolio_analysis.md")
)

print("✓ 组合分析完成")
print(f"不安全组合: {len(synergy_report.unsafe_combinations)}")
print(f"尾部损失事件: {len(synergy_report.simultaneous_tail_losses)}")
EOF

# 运行
PYTHONPATH=/home/neal/mkcs python test_portfolio.py
```

**预期结果**:
- 成功构建组合
- 识别worst windows
- 生成协同分析报告

---

## 🔍 故障排查准备

### 如果基线生成失败

**问题**: `freeze_baselines()` 报错
```bash
# 检查replay数据
python -c "
from analysis.replay_schema import load_replay_outputs
replays = load_replay_outputs('runs')
print(f'找到 {len(replays)} 个replay')
for r in replays:
    print(f'  - {r.run_id}: {r.strategy_id}')
"
```

**解决方案**:
- 确保runs/目录有有效的replay数据
- 检查replay schema是否完整

### 如果回归测试FAIL

**问题**: 某个策略测试失败
```bash
# 查看详细报告
cat reports/risk_regression/report.md | grep -A 10 "FAIL"
```

**解决方案**:
- 检查是真实退化还是容差太严
- 如果是预期变更，更新基线
- 如果是真实退化，修复代码

### 如果风控未触发

**问题**: 风控statistics显示0次触发
```bash
# 检查规则配置
python -c "
from analysis.actionable_rules import RiskRuleset
import json
ruleset = RiskRuleset.from_dict(json.load(open('rules_file.json')))
for rule in ruleset.rules:
    print(f'{rule.rule_name}: {rule.trigger_metric} {rule.trigger_operator} {rule.trigger_threshold}')
"
```

**解决方案**:
- 检查规则阈值是否合理
- 检查RuntimeRiskCalculator是否正确计算指标
- 调整规则阈值

---

## 📚 参考文档

### 核心文档
- `SPL-4_IMPLEMENTATION.md` - 完整实现细节
- `SPL-4_ACCEPTANCE.md` - 验收检查清单
- `SPL-4_DELIVERY_SUMMARY.md` - 快速开始指南

### 代码文件
- `analysis/baseline_manager.py` - 基线管理
- `skills/risk/risk_gate.py` - 风控引擎
- `analysis/portfolio/` - 组合分析模块

### 测试脚本
- `tests/risk_regression/run_risk_regression.py` - 回归测试
- `tests/risk_regression/gating_verification.py` - 风控验证

---

## ✅ 检查清单

明天开始前，确认：

- [ ] 所有代码已提交到git
- [ ] 有可用的replay数据在 `runs/`
- [ ] 已阅读 `SPL-4_DELIVERY_SUMMARY.md`
- [ ] 已准备好测试策略的配置文件

---

## 💡 快速命令参考

```bash
# 设置环境变量
export PYTHONPATH=/home/neal/mkcs

# 查看所有replay
python -c "from analysis.replay_schema import load_replay_outputs; \
  replays = load_replay_outputs('runs'); \
  [print(f'{r.run_id}: {r.strategy_id}') for r in replays]"

# 冻结基线
python -c "from analysis.baseline_manager import BaselineManager; \
  mgr = BaselineManager(); \
  mgr.freeze_baselines('runs', 'baselines/risk')"

# 运行回归测试
python tests/risk_regression/run_risk_regression.py

# 查看基线
python -c "from analysis.baseline_manager import BaselineManager; \
  mgr = BaselineManager(); \
  snapshot = mgr.load_all_baselines(); \
  print(f'基线数: {len(snapshot.baselines)}')"

# 查看风控统计
python -c "from skills.risk.risk_gate import RiskGate; \
  gate = RiskGate(ruleset); \
  print(gate.get_statistics())"
```

---

**明天的主要目标**: 完成SPL-4的实际数据集成测试，验证所有功能在真实数据上的表现。

**预计时间**: 2-3小时

**成功标准**:
- ✅ 至少1个策略的基线已生成
- ✅ 回归测试全部通过
- ✅ 风控在回测中有效触发
- ✅ 成功构建并分析1个组合
