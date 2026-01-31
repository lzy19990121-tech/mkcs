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
├── analysis/           # 风险分析模块
│   ├── replay_schema.py       # 统一输出格式
│   ├── window_scanner.py      # 窗口扫描器
│   ├── stability_analysis.py  # 稳定性分析
│   ├── multi_strategy_comparison.py  # 多策略对比
│   └── risk_card_generator.py # Risk Card生成器
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

```bash
python -m agent.live_runner --mode paper \
    --symbols AAPL MSFT \
    --interval 1m \
    --cash 100000
```

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

### 使用场景

1. **策略验证**: 识别策略的最坏情况表现
2. **参数调优**: 比较不同参数下的风险特征
3. **策略选择**: 选择稳定性更好的策略
4. **风险评估**: 评估最大潜在损失
5. **可复现性审计**: 确保结果可完全复现

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
