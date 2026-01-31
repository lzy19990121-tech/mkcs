# MKCS - 自动交易Agent系统

可扩展的自动交易agent（美股+港股），采用 skills + agent 编排架构。

## 核心原则

LLM不参与实时下单，只做盘后分析和策略配置。

## 技术栈

- Python 3.9+
- 数据结构: dataclass
- 存储: SQLite
- 版本控制: Git
- 测试: 每个模块有独立的main函数用于自测

## 项目结构

```
mkcs/
├── core/           # 核心数据模型
├── skills/         # 各种技能模块（市场数据、策略、风控等）
├── broker/         # 模拟交易执行
├── agent/          # 任务编排和状态管理
│   ├── runner.py           # 交易Agent
│   ├── replay_engine.py    # 回放引擎
│   ├── live_runner.py      # 实时交易（paper模式）
│   └── health_monitor.py   # 健康监控
├── storage/        # 数据持久化
├── reports/        # 报告生成
├── runs/           # 实验运行目录
│   └── <experiment_id>/    # 每次运行的独立目录
│       ├── run_manifest.json
│       ├── summary.json
│       ├── trades.csv
│       └── equity_curve.csv
├── scripts/        # 工具脚本
│   └── compare_strategies.py  # 策略对比
├── config.py       # 回测配置类
├── run_manifest.py # 运行清单管理
└── tests/          # 测试用例
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行回放回测

#### 使用配置文件（推荐）

```python
from config import create_mock_config, create_csv_config
from agent.runner import run_backtest_with_config

# Mock数据回测
config = create_mock_config(
    symbols=['AAPL', 'MSFT'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    seed=42
)
result = run_backtest_with_config(config)

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

基础回放（使用模拟数据）：
```bash
python -m agent.runner --mode replay --start 2024-01-01 --end 2024-01-31 --interval 1d --cash 100000
```

### 运行策略对比

```bash
python scripts/compare_strategies.py
```

对比多个策略/参数组合，生成：
- `runs/comparison/compare_report.md` - Markdown报告
- `runs/comparison/compare_table.csv` - CSV表格

### Paper 模式（实时数据）

```bash
python -m agent.live_runner --mode paper --symbols AAPL MSFT --interval 1m
```

**注意事项**：
- Paper模式使用实时数据（Yahoo Finance）
- 每次调用有API限制，建议设置合理的间隔
- 仅用于模拟和验证，不涉及真实资金
- 首次使用建议在交易时段外测试

### 运行状态监控

Paper模式运行时，可以通过以下方式监控状态：

1. **事件日志**: `logs/events_YYYYMMDD.jsonl`
2. **健康指标**: 检查 `data_lag_seconds`, `error_count`, `orders_pending`
3. **TUI界面**: `python -m tui.watchlist` （仅订阅，不直接访问）

## 功能特性

### 可复现回测系统

- **配置哈希**: 相同配置产生相同哈希
- **数据哈希**: 数据文件完整性验证
- **Git Commit**: 记录代码版本
- **100%复现**: 相同配置+seed = 完全相同结果

```bash
# 验证可复现性
python test_reproducibility.py
```

### 支持的策略

1. **移动平均线 (MA)**: `skills/strategy/moving_average.py`
   - 金叉买入，死叉卖出
   - 可配置快慢周期

2. **突破策略**: `skills/strategy/breakout.py`
   - 突破N日高点买入
   - 突破N日低点卖出

### 数据源支持

- **Mock**: 模拟数据（测试用）
- **CSV**: 本地CSV文件（回测用）
- **Yahoo Finance**: 在线数据（paper模式）

### 成本模型

```python
# BPS滑点模型 (1 BPS = 0.01%)
config = BacktestConfig(
    slippage_bps=5,  # 0.05% 滑点
    commission_per_share=0.01
)
```

### 实验资产化 (runs/ 目录)

每次回测自动生成实验ID并保存到独立目录：

```
runs/
├── exp_abc123/           # 实验1
│   ├── run_manifest.json # 运行清单
│   ├── summary.json      # 结果摘要
│   ├── trades.csv        # 成交记录
│   └── equity_curve.csv  # 权益曲线
├── exp_def456/           # 实验2
└── comparison/           # 策略对比
    ├── compare_report.md
    └── compare_table.csv
```

查看所有实验：
```python
from run_manifest import list_runs
runs = list_runs()
for r in runs:
    print(f"{r.experiment_id}: {r.status} - {r.metrics.get('total_return', 0):.2%}")
```

## 回放输出文件

每次回测输出以下文件：

- `summary.json` - 回测摘要（包含git_commit/config_hash/data_hash等）
- `equity_curve.csv` - 每日权益曲线
- `trades.csv` - 所有成交记录
- `risk_rejects.csv` - 风控拒绝记录

## 开发状态

- [x] Phase 1: 基础设施
- [x] Phase 2: 数据层
- [x] Phase 3: 业务逻辑层
- [x] Phase 4: 编排层
- [x] Phase 5: 集成测试
- [x] Phase 6: 可复现回测系统
- [x] Phase 7: 策略对比框架
- [x] Phase 8: Paper模式稳定性

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# CI回放回归测试
python tests/test_ci_regression.py
```

## 许可证

MIT License
