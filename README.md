# MKCS - 自动交易Agent系统

可扩展的自动交易agent（美股+港股），采用 skills + agent 编排架构。

## 核心原则

LLM不参与实时下单，只做盘后分析和策略配置。

## 技术栈

- Python 3.10+
- 数据结构: dataclass
- 存储: SQLite
- 版本控制: Git
- 测试: 每个模块有独立的main函数用于自测

## 项目结构

```
mkcs/
├── core/           # 核心数据模型
├── skills/         # 各种技能模块（市场数据、策略、风控等）
│   └── market_data/
│       ├── mock_source.py    # 模拟数据源
│       ├── csv_source.py     # CSV数据源（回测）
│       └── yahoo_source.py   # Yahoo Finance数据源
├── broker/         # 模拟交易执行
├── agent/          # 任务编排和状态管理
│   ├── runner.py           # 交易Agent
│   └── replay_engine.py    # 回放引擎
├── storage/        # 数据持久化
├── reports/        # 报告生成
├── events/         # 事件日志系统
├── tui/            # 终端UI界面
├── utils/          # 工具函数
│   └── hash.py     # 哈希计算工具
├── config.py       # 回测配置类
├── data/           # 数据文件目录
└── tests/          # 测���用例
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行回放回测

#### 使用配置文件（推荐）

```python
from config import create_mock_config
from agent.runner import run_backtest_with_config

# 创建配置
config = create_mock_config(
    symbols=['AAPL', 'MSFT'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    seed=42
)

# 运行回测
result = run_backtest_with_config(config)
```

#### 使用CSV真实数据

```python
from config import BacktestConfig
from agent.runner import run_backtest_with_config

config = BacktestConfig(
    data_source='csv',
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

带数据库持久化的回放：
```bash
python -m agent.runner --mode replay --start 2024-01-01 --end 2024-01-31 --interval 1d --db trading.db
```

指定输出目录：
```bash
python -m agent.runner --mode replay --start 2024-01-01 --end 2024-01-31 --interval 1d --output-dir reports/replay
```

查看帮助：
```bash
python -m agent.runner --help
```

### 运行集成测试

```bash
python tests/test_integration.py
```

### 测试各个模块

每个模块都有独立的main函数用于自测：

```bash
# 测试数据模型
python core/models.py

# 测试市场数据源
PYTHONPATH=/home/neal/mkcs python -m skills.market_data.mock_source

# 测试交易时段管理
PYTHONPATH=/home/neal/mkcs python skills/session/trading_session.py

# 测试策略
PYTHONPATH=/home/neal/mkcs python -m skills.strategy.moving_average

# 测试风控
PYTHONPATH=/home/neal/mkcs python -m skills.risk.basic_risk

# 测试模拟经纪商
PYTHONPATH=/home/neal/mkcs python broker/paper.py

# 测试报告生成
PYTHONPATH=/home/neal/mkcs python -m reports.daily
```

## 功能特性

### 已实现模块

1. **核心数据模型** (core/models.py)
   - Bar: K线数据
   - Quote: 实时报价
   - Signal: 交易信号
   - OrderIntent: 订单意图
   - Trade: 成交记录
   - Position: 持仓信息

2. **市场数据Skill** (skills/market_data/)
   - MockMarketSource: 模拟数据生成器
   - TrendingMockSource: 带趋势的模拟数据
   - get_bars_until: 截止时间点取数
   - 支持美股和港股

3. **交易时段Skill** (skills/session/)
   - 美股交易日历
   - 港股交易日历
   - 交易时段检查

4. **策略Skill** (skills/strategy/)
   - MAStrategy: 移动平均线交叉策略
   - 可扩展的策略基类

5. **风控Skill** (skills/risk/)
   - BasicRiskManager: 基础风控规则
   - 仓位限制
   - 黑名单
   - 资金充足性检查
   - 单日亏损限制

6. **模拟经纪商** (broker/)
   - PaperBroker: 虚拟账户管理
   - 订单提交/撮合 (submit_order/on_bar)
   - 禁止同bar成交 (t+1 open撮合)
   - 持仓管理
   - 资金管理

7. **Agent编排器** (agent/)
   - TradingAgent: 任务编排
   - ReplayEngine: 时间推进器
   - tick(ctx): 单步推进
   - 回放主循环 (run_replay_backtest)

8. **报告生成** (reports/)
   - DailyReport: 每日报告
   - BacktestReport: 回测报告

9. **数据持久化** (storage/)
   - SQLite数据库
   - 交易记录存储
   - K线数据存储
   - 持仓快照存储

## 开发状态

- [x] Phase 1: 基础设施
- [x] Phase 2: 数据层
- [x] Phase 3: 业务逻辑层
- [x] Phase 4: 编排层
- [x] Phase 5: 集成测试

## 回放输出文件

回放结束会输出以下文件到 `--output-dir`：

- `summary.json` - 回测摘要，包含配置哈希、git commit、metrics等
- `equity_curve.csv` - 每日权益曲线
- `trades.csv` - 所有成交记录
- `risk_rejects.csv` - 风控拒绝记录

## 可复现回测系统

系统支持可信、可复现的回测结果：

### 配置哈希
每次回测自动计算配置哈希，确保配置可追踪：
```json
{
  "config_hash": "sha256:7d24743fa7149f3b",
  "git_commit": "15dde232",
  "data_hash": "sha256:48724fba7f560f93"
}
```

### 100%复现
相同配置 + seed = 完全相同的结果：
```bash
# 验证可复现性
python test_reproducibility.py
```

### 回放模式
- `replay_mock` - 工程闭环验证（模拟数据）
- `replay_real` - 策略/风控验证（CSV真实数据）

## 许可证

MIT License
