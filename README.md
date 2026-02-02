# MKCS - 自动交易Agent系统

可扩展的自动交易agent（美股+港股），采用 skills + agent 编排架构。

## 核心原则

LLM不参与实时下单，只做盘后分析和策略配置。

## 技术栈

- Python 3.9+
- 数据结构: dataclass
- 存储: SQLite
- 版本控制: Git
- 测试: pytest

## 项目结构

```
mkcs/
├── core/               # 核心数据模型
├── skills/             # 技能模块
│   ├── market_data/    # 数据源
│   ├── strategy/       # 交易策略
│   └── risk/          # 风控管理
├── broker/             # 模拟交易执行
├── agent/              # 任务编排
│   ├── runner.py              # 回测Agent
│   ├── replay_engine.py       # 回放引擎
│   ├── live_runner.py         # 实时交易
│   └── health_monitor.py      # 健康监控
├── web/                # Web UI
│   ├── app.py                 # Flask后端
│   ├── api/                   # REST API
│   ├── services/              # 业务服务
│   ├── db/                    # 数据库模型
│   ├── socketio_server.py     # WebSocket服务
│   └── frontend/              # React前端
│       ├── src/
│       │   ├── components/    # 组件
│       │   ├── pages/         # 页面
│       │   ├── hooks/         # 自定义Hooks
│       │   ├── stores/        # 状态管理
│       │   └── services/      # API服务
│       └── package.json
├── analysis/           # 风险分析模块
│   ├── replay_schema.py       # ���一输出格式
│   ├── window_scanner.py      # 窗口扫描器
│   ├── stability_analysis.py  # 稳定性分析
│   ├── multi_strategy_comparison.py  # 多策略对比
│   ├── risk_card_generator.py # Risk Card生成器
│   ├── perturbation_test.py   # 扰动测试 (SPL-3b)
│   ├── structural_analysis.py # 结构性风险分析 (SPL-3b)
│   ├── risk_envelope.py       # 风险包络计算 (SPL-3b)
│   ├── actionable_rules.py    # 可执行风控规则 (SPL-3b)
│   ├── deep_analysis_report.py # 深度分析报告生成 (SPL-3b)
│   ├── risk_baseline.py       # 风险基线 (SPL-4c)
│   ├── baseline_manager.py    # 基线管理 (SPL-4c)
│   └── portfolio/             # 组合风险分析 (SPL-4b)
│       ├── portfolio_builder.py    # 组合构建
│       ├── portfolio_scanner.py    # 组合扫描
│       ├── synergy_analyzer.py     # 协同风险分析
│       └── portfolio_risk_report.py # 组合报告
├── runs/               # 实验运行目录
│   └── <experiment_id>/
│       ├── run_manifest.json
│       ├── summary.json
│       ├── trades.csv
│       └── equity_curve.csv
├── scripts/            # 工具脚本
│   └── compare_strategies.py
├── config.py           # 回测配置
├── run_manifest.py    # 运行清单
└── tests/              # 测试用例
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行回放回测

#### Python API方式

```python
from config import create_mock_config, create_csv_config
from agent.runner import run_backtest_with_config

# Mock数据回测
config = create_mock_config(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2024-01-01',
    end_date='2024-06-30',
    seed=42
)
result = run_backtest_with_config(config)
print(f"总收益: {result['metrics']['total_return']*100:.2f}%")
print(f"交易次数: {result['metrics']['trade_count']}")

# CSV真实数据回测
config = create_csv_config(
    data_path='data/aapl_2023.csv',
    symbols=['AAPL'],
    start_date='2023-01-01',
    end_date='2023-04-30',
    strategy_params={'fast_period': 3, 'slow_period': 10}
)
result = run_backtest_with_config(config)
```

#### 命令行方式

```bash
# 基础回放
python -m agent.runner --mode replay \
    --start 2024-01-01 --end 2024-01-31 \
    --symbols AAPL,MSFT --cash 100000

# 指定输出目录
python -m agent.runner --mode replay \
    --start 2024-01-01 --end 2024-01-31 \
    --output-dir reports/my_backtest
```

### 3. 运行策略对比

```bash
python scripts/compare_strategies.py
```

对比结果输出到 `runs/comparison/`：
- `compare_report.md` - Markdown格式报告
- `compare_table.csv` - CSV格式表格

### 4. 生成风险分析报告

```bash
# 为所有回测结果生成Risk Card
python -c "from analysis import generate_risk_cards; generate_risk_cards('runs', 'runs/risk_analysis')"
```

风险分析报告输出到 `runs/risk_analysis/`：
- `<experiment_id>_risk_card.md` - 单个策略的详细风险报告
- `comparison_risk_card.md` - 多策略风险对比报告

### 5. Paper模式（实时数据）

#### 命令行方式

```bash
python -m agent.live_runner --mode paper \
    --symbols AAPL MSFT \
    --interval 1m \
    --cash 100000
```

#### Web UI方式（推荐）

启动后端服务：

```bash
# 设置Python路径并启动
export PYTHONPATH=/home/neal/mkcs:$PYTHONPATH
python web/app.py --host 0.0.0.0 --port 5000 --debug
```

启动前端开发服务器：

```bash
cd web/frontend
npm install
npm run dev
```

访问 `http://localhost:5173` 打开实时交易界面。

**Web UI功能**：
- 📊 实时行情和K线图
- 📈 策略信号显示（目标价/止损价区间）
- 🔔 价格触及提醒（观察列表高亮）
- 💰 模拟交易（买入/卖出）
- 📋 持仓和风控状态
- 📝 买卖点标注
- ⏰ 时间周期切换（1分/15分/日K）

**⚠️ 重要提示**：
- Paper模式使用Yahoo Finance实时数据
- 仅用于模拟交易，不涉及真实资金
- 建议先在非交易时段测试连接
- 注意API调用频率限制

## 功能特性

### 可复现回测系统

| 功能 | 说明 | 验证 |
|------|------|------|
| 配置哈希 | 相同配置产生相同哈希 | ✅ |
| 数据哈希 | 数据文件完整性验证 | ✅ |
| Git Commit | 记录代码版本 | ✅ |
| 100%复现 | 相同配置+seed结果一致 | ✅ |

```bash
# 验证可复现性
python test_reproducibility.py
```

### 支持的策略

#### 1. 移动平均线策略 (MA)

```python
from skills.strategy.moving_average import MAStrategy

strategy = MAStrategy(fast_period=5, slow_period=20)
```

- 金叉买入：快线上穿慢线
- 死叉卖出：快线下穿慢线
- 可配置快慢周期

#### 2. 突破策略 (Breakout)

```python
from skills.strategy.breakout import BreakoutStrategy

strategy = BreakoutStrategy(period=20, threshold=0.01)
```

- 价格突破N日高点买入
- 价格跌破N日低点卖出
- 可配置周期和阈值

### 数据源支持

| 数据源 | 类型 | 用途 |
|--------|------|------|
| Mock | 模拟数据 | 测试、工程验证 |
| CSV | 本地文件 | 回测、策略验证 |
| Yahoo Finance | 在线数据 | Paper模式 |

### 成本模型

```python
# BPS滑点模型 (1 BPS = 0.01%)
config = BacktestConfig(
    slippage_bps=5,  # 0.05% 滑点
    commission_per_share=0.01  # 每股$0.01手续费
)
```

- **BPS滑点**: 买入价格增加，卖出价格减少
- **手续费**: 每股固定费用

## 风险分析系统

### 概述

风险分析系统提供最坏情况场景检测和评估，帮助识别策略的潜在风险点。

### 核心功能

#### 1. 时间窗口扫描 (Window Scanner)

扫描多个时间窗口（1d, 5d, 20d, 60d, 120d, 250d），找到最坏情况窗口：

```python
from analysis import WindowScanner, load_replay_outputs

# 加载回测结果
replays = load_replay_outputs("runs")

# 创建窗口扫描器
scanner = WindowScanner(windows=["5d", "20d", "60d"], top_k=5)

# 找到最坏窗口
worst_windows = scanner.find_worst_windows(replays[0])

for w in worst_windows:
    print(f"{w.window_id}: {w.total_return*100:.2f}% (MDD={w.max_drawdown*100:.1f}%)")
```

#### 2. 风险指标

| 指标 | 说明 |
|------|------|
| 最大回撤 (MDD) | 窗口内最大权益回撤 |
| 回撤持续 | 最大回撤持续天数 |
| 回撤恢复时间 | 从最大回撤恢复所需天数 |
| Ulcer指数 | 回撤面积指标 |
| 下行波动 | 负收益波动率 |
| 95% CVaR | 条件风险价值 |
| 尾部均值 | 最差5%日均收益 |
| 尖刺风险 | 最大单步亏损、最长连亏 |
| 回撤形态 | sharp/slow_recovery/unrecovered |

#### 3. 稳定性分析

评估策略在不同窗口长度下的稳定性：

```python
from analysis import StabilityAnalyzer, analyze_all_stability

# 分析所有策略的稳定性
reports = analyze_all_stability("runs")

for report in reports:
    print(f"{report.strategy_id}: 稳定性评分 {report.stability_score:.1f}/100")
```

**稳定性评分组成**：
- 收益方差（0-40分惩罚）
- MDD一致性（0-30分惩罚）
- 最差CVaR（0-30分惩罚）
- 满分100分，越高越稳定

#### 4. 多策略对比

比较不同策略的最坏情况表现：

```python
from analysis import MultiStrategyComparator, compare_all_strategies

# 对比所有策略
summaries = compare_all_strategies("runs")

# 生成对比表格
comparator = MultiStrategyComparator()
df = comparator.generate_comparison_table(summaries)
print(df)

# 找到各窗口最优策略
best_20d = comparator.find_best_for_window(summaries, "20d")
print(f"20d窗口最优: {best_20d.strategy_name}")
```

#### 5. Risk Card 生成

自动生成Markdown格式的风险分析报告：

```python
from analysis import generate_risk_cards, RiskCardGenerator

# 为所有replay生成Risk Card
generate_risk_cards("runs", "runs/risk_analysis")

# 或使用生成器
generator = RiskCardGenerator()
generator.generate_for_replay(replay, "output/risk_card.md")
generator.generate_for_comparison(replays, "output/comparison.md")
```

**Risk Card内容包括**：
- 基本信息（策略、日期、收益、git commit、config hash）
- 稳定性评分和指标分解
- Top-K最坏窗口详情
- 各窗口长度最坏情况汇总
- 可复现性审计指令
- 完整配置快照

### Risk Card 示例

```markdown
# 风险分析报告
**策略**: ma_5_20
**运行ID**: exp_abc123
**生成时间**: 2026-01-31 23:26:07

## 1. 基本信息
- **回测期间**: 2024-01-01 至 2024-06-30
- **总收益**: -1.71%
- **Git Commit**: 0232e5c9
- **Config Hash**: cfg_abc123

## 2. 稳定性评分
### 总分: 5.3 / 100

| 指标 | 数值 |
|------|------|
| 收益方差 | 8.7478% |
| 收益极差 | 86.26% |
| MDD一致性 | 0.824 |
| 最差CVaR | 5.415 |

## 3. Top-K 最坏窗口

### #1 - 20d 窗口
- **时间范围**: 2024-05-31 至 2024-06-28
- **窗口收益**: -27.24%
- **最大回撤**: 1.93%
- **回撤形态**: slow_recovery
...
```

## SPL-3b 深度风险分析系统

### 概述

SPL-3b（Stress Perturbation Level-3b）是在基础风险分析之上的深度扰动测试系统，通过系统性压力测试识别策略的结构性风险并生成可执行的风控规则。

### 核心组件

#### 1. 扰动测试 (Perturbation Testing)

测试最坏情况窗口在不同微小扰动下的稳定性：

```python
from analysis import PerturbationTester, PerturbationConfig
from analysis.replay_schema import load_replay_outputs

# 加载回测结果
replays = load_replay_outputs("runs")
replay = replays[0]

# 配置扰动测试
config = PerturbationConfig(
    cost_epsilon=0.01,  # ±1% 成本扰动
    capital_epsilon=0.01,  # ±1% 资金扰动
    enable_replay_order=True,  # 重放顺序扰动
    enable_cost_epsilon=True,  # 成本扰动
    enable_capital_epsilon=True  # 资金扰动
)

# 运行扰动测试
tester = PerturbationTester(config)
results = tester.test_perturbations(replay, window_length="20d")

for r in results:
    print(f"{r.perturbation_type}: {r.stability_label}")
    print(f"  原始收益: {r.original_return*100:.2f}%")
    print(f"  扰动后: {r.perturbed_return*100:.2f}%")
```

**稳定性分类**：
- **Stable**: 扰动后最坏窗口仍在同一时间区间（±2天）
- **Weakly Stable**: 扰动后最坏窗口在原始Top-K中
- **Fragile**: 扰动后最坏窗口不在Top-K中

**扰动类型**：
| 类型 | 说明 | 扰动值 |
|------|------|--------|
| replay_order | 重放顺序变化 | 随机打乱 |
| cost_epsilon | 交易成本变化 | ±1% |
| capital_epsilon | 初始资金变化 | ±1% |

#### 2. 结构性风险分析 (Structural Analysis)

分析Top-K最坏窗口是否存在相似的风险模式：

```python
from analysis import StructuralAnalyzer

# 分析20d窗口的结构性风险
analyzer = StructuralAnalyzer(top_k=5)
result = analyzer.analyze(replay, window_length="20d")

print(f"风险类型: {result.risk_pattern_type}")
print(f"形态相似度: {result.metrics.pattern_similarity:.3f}")
print(f"MDD变异系数: {result.metrics.mdd_cv:.3f}")

if result.risk_pattern_type == "structural":
    print("⚠️ 检测到结构性风险！最坏窗口高度相似")
    print("建议在震荡市场时禁用该策略")
```

**风险类型**：
- **Structural**: Top-K窗口高度相似（相似度 > 0.7），表示风险是结构性的、会重复发生
- **Single-Outlier**: Top-K窗口差异大，最坏情况是单一异常事件

**关键指标**：
| 指标 | 说明 | 阈值 |
|------|------|------|
| pattern_similarity | 形态相似度 | >0.7为结构性 |
| mdd_cv | MDD变异系数 | <0.3为稳定 |
| avg_mdd | 平均最大回撤 | Top-K均值 |

#### 3. 风险包络 (Risk Envelope)

构建最坏情况的统计边界（P95/P99分位数）：

```python
from analysis import RiskEnvelopeBuilder

# 构建风险包络
builder = RiskEnvelopeBuilder(confidence_level=0.95)
envelope = builder.build_envelope(replay, window_length="20d")

print("=== 收益包络 ===")
print(f"P50 (中位数): {envelope.return_percentiles['p50']*100:.2f}%")
print(f"P95 (95%边界): {envelope.return_percentiles['p95']*100:.2f}%")
print(f"P99 (99%边界): {envelope.return_percentiles['p99']*100:.2f}%")

print("\n=== MDD包络 ===")
print(f"P95 MDD: {envelope.mdd_percentiles['p95']*100:.2f}%")

print("\n=== 持续时间包络 ===")
print(f"P95 持续: {envelope.duration_percentiles['p95']:.1f}天")
```

**风险包络用途**：
- 设定止损阈值
- 评估极端情况损失
- 制定仓位管理规则

#### 4. 可执行风控规则 (Actionable Rules)

基于深度分析生成可执行的if-then风控规则：

```python
from analysis import RiskRuleGenerator, DeepAnalysisReportGenerator

# 生成深度分析报告（包含规则）
generator = DeepAnalysisReportGenerator()
report = generator.generate_full_report(replay, window_lengths=["20d", "60d"])

# 保存报告
with open("runs/deep_analysis_v3b/exp_abc123_deep_analysis_v3b.md", "w") as f:
    f.write(report.markdown_report)

with open("runs/deep_analysis_v3b/exp_abc123_deep_analysis_v3b.json", "w") as f:
    f.write(report.json_report)
```

**规则类型**：
| 类型 | 说明 | 示例 |
|------|------|------|
| Gating | 暂停交易 | 稳定性评分 < 30 → 暂停 |
| Position Reduction | 降低仓位 | 窗口收益 < -10% → 减仓50% |
| Disable | 禁用策略 | 震荡市场 → 禁用 |

**规则示例**：
```python
# 规则1: 低稳定性暂停交易
if stability_score < 30:
    pause_trading(reason="稳定性不足")

# 规则2: 极端亏损降仓
if window_return < -0.10:  # -10%
    reduce_position_by(0.50)  # 降低50%仓位

# 规则3: 结构性风险禁用
if market_regime == "ranging" and adx < 25:
    disable_strategy(reason="震荡市场，结构性风险")
```

### 完整工作流程

```python
from analysis import (
    load_replay_outputs,
    DeepAnalysisReportGenerator
)

# 1. 加载所有回测结果
replays = load_replay_outputs("runs")

# 2. 生成深度分析报告
generator = DeepAnalysisReportGenerator()

for replay in replays:
    # 生成Markdown + JSON双格式报告
    report = generator.generate_full_report(
        replay,
        window_lengths=["20d", "60d"]
    )

    # 保存报告
    run_id = replay.run_id
    md_path = f"runs/deep_analysis_v3b/{run_id}_deep_analysis_v3b.md"
    json_path = f"runs/deep_analysis_v3b/{run_id}_deep_analysis_v3b.json"

    with open(md_path, "w") as f:
        f.write(report.markdown_report)

    with open(json_path, "w") as f:
        f.write(report.json_report)

    print(f"✅ {run_id}: 深度分析完成")
```

### 报告内容

#### Markdown报告 (`*_deep_analysis_v3b.md`)

```
╔════════════════════════════════════════════════════════════════╗
║         SPL-3b 深度风险分析报告                                    ║
║         深度扰动测试 + 结构分析 + 风险包络 + 可执行规则           ║
╚════════════════════════════════════════════════════════════════╝

# 20d 窗口 - 深度风险分析

## 一、Worst-Case 稳定性检验（扰动测试）
**稳定性标签**: Stable

### 扰动测试详情
| 扰动类型 | 扰动值 | 原始收益 | 扰动后收益 | 差异 | 同一窗口 | 在Top-K |
|----------|--------|----------|------------|------|----------|---------|
| replay_order | +0.0% | 4.15% | 4.15% | +0.00% | ✓ | ✓ |
| cost_epsilon | +1.0% | 4.15% | 4.19% | +0.04% | ✓ | ✓ |
| capital_epsilon | -1.0% | 4.15% | 4.19% | +0.04% | ✓ | ✓ |

## 二、Worst-Case 结构确认
**风险类型**: 结构性风险 (Structural Risk Pattern)

### 形态指标
| 指标 | 数值 | 说明 |
|------|------|------|
| 形态相似度 | 0.897 | >0.7为高度相似 |
| MDD变异系数 | 0.000 | <0.3为稳定 |

## 三、Worst-Case Envelope（风险边界）
### Worst-Case Return 分位数
| 分位数 | 数值 | 说明 |
|--------|------|------|
| P95    |   -26.97% | 95%置信边界 |
| P99    |   -26.96% | 99%置信边界 |

## 四、可执行风险规则
### 规则 #1: 结构性风险禁用
- **触发条件**: `market_regime < 0.0`
- **描述**: 策略表现出结构性风险（形态相似度0.90）
- **伪代码**: `if market_regime < 0.0: disable_strategy()`
```

#### JSON报告 (`*_deep_analysis_v3b.json`)

```json
{
  "strategy_id": "ma_5_20",
  "run_id": "exp_1677b52a",
  "commit_hash": "0232e5c9",
  "config_hash": "cfg_1677b52a",
  "windows": {
    "20d": {
      "stability_label": "Stable",
      "risk_pattern_type": "structural",
      "pattern_metrics": {
        "pattern_similarity": 0.753,
        "mdd_cv": 0.155
      },
      "envelope": {
        "return_percentiles": {
          "p95": -0.2697,
          "p99": -0.2696
        }
      },
      "rules": [
        {
          "rule_id": "ma_5_20_regime_disable",
          "rule_type": "disable",
          "trigger_metric": "market_regime",
          "trigger_threshold": 0.0,
          "description": "策略表现出结构性风险"
        }
      ]
    }
  }
}
```

### 自我评估问题

使用SPL-3b系统后，可以回答以下关键问题：

1. **最坏情况是否稳定？**
   - 通过扰动测试验证
   - Stable表示不是偶然事件

2. **风险是重复的还是偶然的？**
   - 通过结构分析判断
   - Structural表示会重复发生

3. **风险上限是多少？**
   - 通过风险包络量化
   - P95/P99给出明确边界

4. **应该优先应用什么控制？**
   - 通过可执行规则指导
   - 明确的if-then逻辑

### 使用场景

1. **策略上线前验证**: 确保策略在压力测试下表现稳定
2. **风控规则设计**: 基于实际风险特征制定规则
3. **风险评估**: 量化极端情况下的潜在损失
4. **策略优化**: 识别结构性弱点并改进

### 示例输出

报告保存在 `runs/deep_analysis_v3b/` 目录：

```
runs/deep_analysis_v3b/
├── exp_1677b52a_deep_analysis_v3b.md   # MA(5,20)分析
├── exp_1677b52a_deep_analysis_v3b.json
├── exp_3f38beb1_deep_analysis_v3b.md
├── exp_3f38beb1_deep_analysis_v3b.json
└── ...
```

### 使用场景

1. **策略验证**: 识别策略的最坏情况表现
2. **参数调优**: 比较不同参数下的风险特征
3. **策略选择**: 选择稳定性更好的策略
4. **风险评估**: 评估最大潜在损失
5. **可复现性审计**: 确保结果可完全复现

## SPL-7 在线监控与反事实分析系统

### 概述

SPL-7 是完整的运行态风险监控与反事实分析系统，实现了从 CI-only 到 continuous online monitoring 的转变，并能回答"如果当时换了规则会怎样"的问题。

**两个核心模块**:
- **SPL-7a**: Online Monitoring & Post-mortem Attribution
- **SPL-7b**: Counterfactual & What-If Risk Analysis

### SPL-7a: 在线监控与事后归因

#### 核心功能

**运行态数据采集**:
```python
from analysis.online import RiskMetricsCollector, RiskSignal

# 初始化采集器
collector = RiskMetricsCollector(strategy_id="my_strategy")

# 每个时间步更新
signal = collector.update(price=150.0, timestamp=now, position=100.0)

# RiskSignal 包含:
# - RollingReturnMetrics: 1d/5d/20d/60d 收益
# - DrawdownMetrics: 当前回撤、最大回撤、持续时间
# - SpikeMetrics: 最大单步亏损、聚类评分
# - StabilityMetrics: 稳定性评分、波动率
# - RegimeFeatures: vol/ADX/spread_cost
# - GatingEvent/AllocatorEvent: 决策事件
```

**风险状态机**:
```python
from analysis.online import RiskStateMachine, RiskState

# 创建状态机
machine = RiskStateMachine(strategy_id="my_strategy")

# 更新状态
transition = machine.update_state(signal)
if transition:
    print(f"状态转换: {transition.from_state} → {transition.to_state}")

# RiskState: NORMAL / WARNING / CRITICAL
# envelope_usage = current_drawdown / envelope_limit
```

**趋势检测**:
```python
from analysis.online import TrendDetector

detector = TrendDetector()
trend = detector.detect_trend("volatility", timestamps, values)
# TrendIndicator: direction (UP/DOWN/STABLE), slope, R²
```

**多渠道告警**:
```python
from analysis.online import AlertingManager

manager = AlertingManager()
alerts = manager.process_risk_update(signal, state, trends, transition)
# 告警渠道: LOG/Slack/Webhook/Email
# 告警规则: envelope_approach, spike_surge, gating_frequency_high, etc.
```

**Post-mortem 自动生成**:
```python
from analysis.online import PostMortemGenerator

generator = PostMortemGenerator()
report = generator.generate_from_gate_event(gating_event)
# 包含: 触发时间、上下文窗口、指标轨迹、根因分析、建议
```

**风险事件存储**:
```python
from analysis.online import RiskEventStore

store = RiskEventStore("data/risk_events.db")
store.store_event(event_type="GATING", data=gating_event.to_dict())
# SQLite 持久化，支持查询和导出到 SPL-7b/SPL-6a
```

#### 使用场景

1. **运行态监控**: 实时追踪风险指标，早期预警
2. **事后归因**: 自动生成 post-mortem，理解风险事件
3. **趋势分析**: 识别风险上升趋势，提前干预
4. **事件桥接**: 在线事件 → 离线分析（SPL-7b, SPL-6a）

---

### SPL-7b: 反事实与假设分析

#### 核心功能

**场景配置**:
```python
from analysis.counterfactual import CounterfactualScenarioLibrary

# 获取预定义场景
scenarios = CounterfactualScenarioLibrary.get_all_scenarios(["strategy_1"])

# 场景包括:
# - actual: 真实发生
# - cf_earlier_gating: 更早/更强的 gating
# - cf_later_gating: 更晚/更弱的 gating
# - cf_no_gating: 完全禁用 gating
# - cf_optimizer: 使用优化器分配
# - cf_equal_weight: 等权重组合
```

**并行执行**:
```python
from analysis.counterfactual import CounterfactualExperiment

experiment = CounterfactualExperiment(
    replay_path="runs",
    strategy_ids=["strategy_1"],
    max_workers=4
)
results = experiment.run_experiment()
# 同一 replay + 多 decision config 并行运行
```

**效果量化**:
```python
from analysis.counterfactual import EffectCalculator

calculator = EffectCalculator()
effects = calculator.calculate_effects(actual_result, cf_results)

# EffectMetrics 包含:
# - avoided_drawdown: 避免的回撤
# - lost_return: 牺牲的收益
# - eliminated_spikes: 消除的尖刺
# - tradeoff_ratio: 权衡比（风险改善/收益牺牲）
```

**规则评估**:
```python
from analysis.counterfactual import RuleEvaluator

evaluator = RuleEvaluator()
evaluations = evaluator.evaluate_rules(actual_result, cf_results)

# 找出:
# - 最值钱的规则 (overall_value > 70)
# - 几乎无贡献的规则 (overall_value < 30)
# - 需要调整的规则 (overall_value 30-70)
```

**反馈回流**:
```python
from scripts.run_counterfactual_analysis import (
    run_counterfactual_analysis_and_feedback
)

report_path, feedback = run_counterfactual_analysis_and_feedback(
    replay_path="runs",
    strategy_ids=["strategy_1"]
)

# feedback["spl6a"]: SPL-6a 再标定建议
# feedback["spl5"]: SPL-5 规则调整建议
```

#### 使用场景

1. **What-If 分析**: 如果当时早一点 gating，会不会更好？
2. **规则优化**: 哪条规则最值钱？哪条可以删？
3. **Allocator 选择**: 优化器 vs 规则 vs 等权重？
4. **组合调整**: 排除某个策略对 co-crash 的影响？

---

### 完整工作流程

```python
# 1. 运行时监控
collector = RiskMetricsCollector(strategy_id)
signal = collector.update(price, timestamp, position)

# 2. 状态机判定
machine = RiskStateMachine(strategy_id)
transition = machine.update_state(signal)

# 3. 告警
alerts = alerting_manager.process_risk_update(signal, state, trends)

# 4. Post-mortem 生成
if gating_event:
    report = postmortem_generator.generate_from_gate_event(gating_event)

# 5. 事件存储
risk_event_store.store_event("GATING", gating_event.to_dict())

# 6. 反事实分析
cf_results = counterfactual_experiment.run_experiment()

# 7. 效果量化
effects = effect_calculator.calculate_effects(actual, cf_results)

# 8. 规则评估
evaluations = rule_evaluator.evaluate_rules(actual, cf_results)

# 9. 反馈回流
spl6a_feedback = feedback_looper.generate_spl6a_feedback(report)
spl5_feedback = feedback_looper.generate_spl5_feedback(report)
```

---

### 核心价值

| 能力 | SPL-7a | SPL-7b |
|------|--------|--------|
| 今天系统有没有在逼近风险上界？ | ✅ envelope_usage | - |
| 为什么某次 gate 被触发？ | ✅ PostMortemReport | - |
| 如果当时早一点 / 换个规则，会不会更好？ | - | ✅ tradeoff_ratio |
| 哪条规则最值钱？ | - | ✅ overall_value |
| 哪条规则可以删？ | - | ✅ identify_weak_rules() |
| 结论都有数据与 replay 证据？ | ✅ RiskEventStore | ✅ CounterfactualResult |

---

### 完整文档

详细实现请参见：
- **✅ SPL-7验收报告**: `docs/SPL7_ACCEPTANCE_REPORT.md`
- **📘 Online Risk Events Schema**: `docs/ONLINE_RISK_EVENTS.md`
- **📋 Post-mortem 模板**: `docs/POST_MORTEM_TEMPLATE.md`

---

## Live Trading 实时交易系统 ⚡️

### 概述

实时交易系统将 SPL-7a 在线监控集成到 Live Trader，实现每个交易周期的风险快照、告警触发和事件记录。系统支持 **SHADOW 模式**（模拟，不下单）和 **LIVE 模式**（真实下单）。

**P0 验收状态**: ✅ 已完成
**验收文档**: `docs/P0_ACCEPTANCE.md`

### 核心特性

#### 1. SPL-7a 集成监控

**3 个 Hook 点**:
- **Pre-Decision**: 更新市场特征（vol/ADX/cost proxy）
- **Post-Decision**: 记录 gating/allocator 决策结果
- **Post-Fill**: 更新 PnL/DD/spike 并触发告警

**实时输出**:
- 每周期输出风险快照到 `data/live_monitoring/snapshots.jsonl`
- 所有事件存储到 SQLite 数据库 `risk_events.db`
- 严重告警自动生成 Post-mortem 报告

#### 2. 配置化运行

**Live Config** (`config/live/live_config.yaml`):
```yaml
mode: SHADOW  # SHADOW（模拟）/ LIVE（真实）
symbols: [AAPL, MSFT]
kill_switch:
  max_daily_loss: 0.03  # 3% 止损
  max_drawdown: 0.10
position_limits:
  max_single_position: 0.10  # 10%
```

**Alerts Config** (`config/live/alerts.yaml`):
```yaml
envelope_approach:
  threshold: 0.70  # 70% envelope 使用率告警
envelope_critical:
  threshold: 0.90  # 90% 严重告警
spike_surge:
  threshold: 5      # 5 个 spike 告警
```

#### 3. 环境验证

**验证脚本**: `scripts/verify_live_env.py`

30 项自动检查：
- ✅ Python 版本 >= 3.9
- ✅ 核心依赖包（yfinance, pandas, numpy）
- ✅ 项目模块（7 个核心模块）
- ✅ SPL-7a 组件（7 个）
- ✅ 数据源连接（Yahoo Finance）
- ✅ LiveTrader 初始化
- ✅ RiskMonitor 功能测试

### 快速启动

#### 1. 环境验证

```bash
cd /home/neal/mkcs

# 安装依赖
pip install -r requirements-live.txt

# 验证环境
python scripts/verify_live_env.py
# 应该看到：🎉 所有验证通过！Live Trading 环境就绪
```

#### 2. 启动实时交易

**Shadow 模式**（今晚测试，不下单）:
```bash
python scripts/run_live_with_monitoring.py \
    --symbols AAPL MSFT \
    --cash 100000 \
    --interval 5m
```

**LIVE 模式**（明晚真实交易）:
```bash
python scripts/run_live_with_monitoring.py \
    --symbols AAPL MSFT \
    --cash 100000 \
    --interval 5m \
    --mode live
```

#### 3. 停止交易

```
Ctrl + C
```

系统会自动打印统计并关闭所有组件。

### 运行时监控

**正常输出示例**:
```
[数据获取] AAPL: 获取到 60 根K线
[RiskMonitor] 已生成 10 个快照, 告警 0, Gating 0
[信号] AAPL: BUY @ $175.50
[订单通过] AAPL: 已通过风控
[状态转换] AAPL: NORMAL → WARNING
[告警] AAPL: [WARNING] 接近 Envelope - 回撤接近 70%
```

**输出文件**:
- `data/live_monitoring/snapshots.jsonl` - 风险快照
- `data/live_monitoring/risk_events.db` - 事件数据库
- `data/live_monitoring/postmortem_*.md` - Post-mortem 报告（触发时生成）

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbols` | AAPL MSFT GOOGL | 交易标的 |
| `--cash` | 100000 | 初始资金 |
| `--interval` | 5m | 数据更新间隔（1m/5m/15m） |
| `--mode` | paper | paper（模拟）或 live（真实） |
| `--fast` | 5 | MA 快线周期 |
| `--slow` | 20 | MA 慢线周期 |

### 风险指标说明

**实时监控的指标**:
- **Rolling Returns**: 1d/5d/20d/60d 收益率
- **Drawdown**: 当前回撤、最大回撤、回撤持续时间
- **Spike**: 最大单步亏损、连续亏损次数、聚类评分
- **Stability**: 稳定性评分、波动率
- **Regime**: 市场状态（volatility_bucket, trend_strength, liquidity_level）
- **Risk State**: NORMAL / WARNING / CRITICAL

**告警阈值**:
- Envelope 接近: 70%（WARNING），90%（CRITICAL）
- Spike 激增: > 5 次（WARNING）
- Gating 频率: > 20%（WARNING）
- 单日亏损: > 3%（CRITICAL，Kill Switch）

### 完整文档

- **✅ P0 验收报告**: `docs/P0_ACCEPTANCE.md`
- **📘 快速启动指南**: `docs/LIVE_TRADING_QUICKSTART.md`
- **✅ SPL-7 验收**: `docs/SPL7_ACCEPTANCE_REPORT.md`

---

## SPL-4 风险控制与组合加固系统

### 概述

SPL-4（Safety Protection Level-4）是完整的风险控制与组合加固系统，通过三个阶段保护策略和组合免受最坏情况影响。

**执行流程**: C → A → B（回归测试 → 运行时风控 → 组合分析）

```
SPL-3b (深度分析)
    ↓
SPL-4c (回归测试) ← FREEZE 3b结论
    ↓
SPL-4a (运行时风控) ← 执行3b规则
    ↓
SPL-4b (组合分析) ← 分析4a合格策略
```

### SPL-4c: 风险回归测试

冻结SPL-3b分析结果作为基线，运行5大回归测试防止风险退化。

**核心功能**:
- 基线冻结（`analysis/risk_baseline.py`, `analysis/baseline_manager.py`）
- 5大回归测试（`tests/risk_regression/risk_baseline_test.py`）
- CI集成（`.github/workflows/risk_regression.yml`）

**使用方法**:
```bash
# 冻结基线
PYTHONPATH=/home/neal/mkcs python -c "
from analysis.baseline_manager import BaselineManager
mgr = BaselineManager()
snapshot = mgr.freeze_baselines('runs', 'baselines/risk')
"

# 运行回归测试
PYTHONPATH=/home/neal/mkcs python tests/risk_regression/run_risk_regression.py
```

### SPL-4a: 运行时风险风控

将SPL-3b规则转换为运行时可执行约束，实时保护策略。

**核心功能**:
- 实时指标计算（`skills/risk/runtime_metrics.py`）
- 风险风控器（`skills/risk/risk_gate.py`）
- Agent��成（`agent/runner.py`）

**风控动作**:
- `PAUSE_TRADING`: 暂停交易
- `REDUCE_POSITION`: 减少50%仓位
- `DISABLE_STRATEGY`: 禁用策略

**使用方法**:
```python
from skills.risk.risk_gate import RiskGate
gate = RiskGate(ruleset)
agent.set_risk_gate(gate)
agent.run_replay_backtest(...)
```

### SPL-4b: 组合协同风险分析

分析多策略组合层面的最坏情况协同风险。

**核心功能**:
- 组合构建（`analysis/portfolio/portfolio_builder.py`）
- 组合窗口扫描（`analysis/portfolio/portfolio_scanner.py`）
- 协同风险分析（`analysis/portfolio/synergy_analyzer.py`）
- 组合报告生成（`analysis/portfolio/portfolio_risk_report.py`）

**风险类型**:
- 相关性尖峰
- 同时性尾部损失
- 风险预算违规

**使用方法**:
```python
from analysis.portfolio import PortfolioBuilder, SynergyAnalyzer

builder = PortfolioBuilder()
portfolio = builder.build_portfolio(config, 'runs')

analyzer = SynergyAnalyzer()
synergy_report = analyzer.generate_synergy_report(portfolio, worst_windows)
```

### 完整文档

详细实现请参见：
- **📘 SPL-4实现文档**: `SPL-4_IMPLEMENTATION.md`
- **✅ SPL-4验收报告**: `SPL-4_ACCEPTANCE.md`
- **📦 SPL-4交付文档**: `SPL-4_DELIVERY_SUMMARY.md`

## 实验资产化 (runs/ 目录)

每次回测自动生成实验ID并保存到独立目录：

```
runs/
├── exp_abc123/              # 实验1
│   ├── run_manifest.json    # 运行清单
│   ├── summary.json          # 结果摘要
│   ├── trades.csv            # 成交记录
│   └── equity_curve.csv     # 权益曲线
├── exp_def456/              # 实验2
│   └── ...
├── risk_analysis/           # 风险分析报告
│   ├── exp_abc123_risk_card.md
│   ├── exp_def456_risk_card.md
│   └── comparison_risk_card.md
├── deep_analysis_v3b/       # SPL-3b深度分析
│   ├��─ exp_abc123_deep_analysis_v3b.md
│   ├── exp_abc123_deep_analysis_v3b.json
│   └── ...
└── comparison/             # 策略对比
    ├── compare_report.md
    └── compare_table.csv
```

### run_manifest.json 示例

```json
{
  "experiment_id": "exp_abc123",
  "started_at": "2026-01-31T23:00:00",
  "ended_at": "2026-01-31T23:05:00",
  "status": "completed",
  "git_commit": "7bfce06",
  "config_hash": "sha256:abc123",
  "data_hash": "sha256:def456",
  "mode": "replay_mock",
  "symbols": ["AAPL", "MSFT"],
  "metrics": {
    "total_return": 0.05,
    "trade_count": 10
  },
  "artifacts": [
    "summary.json",
    "trades.csv",
    "equity_curve.csv"
  ]
}
```

### 查询实验

```python
from run_manifest import list_runs

# 列出所有实验
runs = list_runs()
for r in runs:
    print(f"{r.experiment_id}: {r.status}")
    print(f"  收益: {r.metrics.get('total_return', 0):.2%}")
    print(f"  交易: {r.metrics.get('trade_count', 0)}笔")

# 只列出完成的实验
completed = list_runs(status="completed")
```

## 策略对比报告示例

```markdown
# 策略对比报告

| 策略 | 总收益 | 最大回撤 | 交易次数 | 胜率 | 平均持仓(天) |
|------|--------|----------|----------|------|-------------|
| ma_5_20 | -1.71% | 4.79% | 8 | 66.7% | 15.3 |
| ma_3_10 | -0.63% | 2.33% | 15 | 28.6% | 30.3 |
| breakout_20 | -1.31% | 3.29% | 4 | 50.0% | 84.5 |
| breakout_10 | -2.93% | 3.48% | 10 | 20.0% | 21.8 |
```

## 输出文件说明

### summary.json

回测结果摘要，包含：
- `backtest_date`: 回测时间
- `replay_mode`: 回放模式
- `data_source`: 数据源信息
- `data_hash`: 数据文件哈希
- `config`: 完整配置
- `config_hash`: 配置哈希
- `git_commit`: Git提交哈希
- `metrics`: 性能指标

### equity_curve.csv

每日权益曲线，列：
- `date`: 日期
- `cash`: 现金余额
- `equity`: 总权益
- `pnl`: 盈亏

### trades.csv

成交记录，列：
- `trade_id`: 成交ID
- `timestamp`: 成交时间
- `symbol`: 标的
- `side`: 方向 (BUY/SELL)
- `price`: 成交价
- `quantity`: 数量
- `commission`: 手续费

### risk_rejects.csv

风控拒绝记录，列：
- `date`: 日期
- `symbol`: 标的
- `action`: 方向
- `reason`: 拒绝原因

## 配置示例

### Mock数据回测

```python
from config import BacktestConfig
from agent.runner import run_backtest_with_config

config = BacktestConfig(
    data_source="mock",
    seed=42,
    symbols=["AAPL", "MSFT"],
    start_date="2024-01-01",
    end_date="2024-06-30",
    strategy_name="ma",
    strategy_params={"fast_period": 5, "slow_period": 20},
    slippage_bps=2,
    commission_per_share=0.01,
    initial_cash=100000.0
)
result = run_backtest_with_config(config)
```

### CSV数据回测

```python
config = BacktestConfig(
    data_source="csv",
    data_path="data/aapl_2023.csv",
    symbols=["AAPL"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    strategy_params={"fast_period": 10, "slow_period": 30},
    slippage_bps=5,
    initial_cash=100000.0
)
result = run_backtest_with_config(config)
```

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# CI回归测试
python tests/test_ci_regression.py

# 可复现性验证
python test_reproducibility.py
```

## 常见问题

### Q: 如何验证回测结果可复现？

A: 运行两次相同配置，比较 `config_hash` 和 `metrics`：

```python
result1 = run_backtest_with_config(config, output_dir="runs/test1")
result2 = run_backtest_with_config(config, output_dir="runs/test2")

assert result1['config_hash'] == result2['config_hash']
assert result1['metrics']['final_equity'] == result2['metrics']['final_equity']
```

### Q: 如何添加自定义策略？

A: 继承 `Strategy` 基类：

```python
from skills.strategy.base import Strategy
from core.models import Bar, Signal

class MyStrategy(Strategy):
    def get_min_bars_required(self):
        return 20

    def generate_signals(self, bars, position=None):
        # 生成信号逻辑
        if bars[-1].close > bars[-2].close:
            return [Signal(...)]
        return []
```

### Q: Paper模式会真实交易吗？

A: 不会。Paper模式使用实时数据但执行模拟交易，不涉及真实资金。

### Q: 如何修改手续费和滑点？

A: 在 `BacktestConfig` 中设置：

```python
config = BacktestConfig(
    commission_per_share=0.01,  # 每股手续费
    slippage_bps=2             # 2 BPS = 0.02%
)
```

### Q: 如何生成风险分析报告？

A: 使用风险分析模块：

```python
from analysis import generate_risk_cards

# 为所有回测结果生成Risk Card
generate_risk_cards("runs", "runs/risk_analysis")

# 报告将包含：
# - 每个策略的详细风险报告
# - 多策略对比报告
# - 可复现性审计信息
```

### Q: 稳定性评分如何计算？

A: 稳定性评分基于三个指标：
- **收益方差**: 不同窗口最坏收益的方差（越小越好）
- **MDD一致性**: 最大回撤的标准差/均值比（越小越好）
- **最差CVaR**: 各窗口中最差的CVaR值（越小越好）

满分100分，每项指标超出阈值会扣分，最终得分越高越稳定。

## 许可证

MIT License
