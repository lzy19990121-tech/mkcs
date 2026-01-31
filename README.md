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
print(f"总收益: {result['metrics']['total_return']*100:.2f}%")
```

#### 运行策略对比

```bash
python scripts/compare_strategies.py
```

对比4个策略，生成报告：
- `runs/comparison/compare_report.md`
- `runs/comparison/compare_table.csv`

### Paper 模式（实时数据）

```bash
python -m agent.live_runner --mode paper --symbols AAPL --interval 1m
```

**注意事项**：
- 使用实时数据（Yahoo Finance）
- 模拟交易，不涉及真实资金
- 建议先在非交易时段测试连接

## 功能特性

### 可复现回测系统

- **配置哈希**: 相同配置产生相同结果
- **数据哈希**: 数据文件完整性验证
- **Git Commit**: 记录代码版本
- **100%复现**: 验证脚本

```bash
python test_reproducibility.py
```

### 支持的策略

1. **移动平均线**: `skills/strategy/moving_average.py`
2. **突破策略**: `skills/strategy/breakout.py`

### 数据源支持

- **Mock**: 模拟数据
- **CSV**: 本地CSV文件
- **Yahoo Finance**: 在线数据

### 成本模型

```python
# BPS滑点 (1 BPS = 0.01%)
config = BacktestConfig(
    slippage_bps=5,  # 0.05%
    commission_per_share=0.01
)
```

### 实验资产化 (runs/ 目录)

```
runs/
├── exp_abc123/
│   ├── run_manifest.json
│   ├── summary.json
│   ├── trades.csv
│   └── equity_curve.csv
```

## 测试

```bash
# 所有测试
python -m pytest tests/ -v

# CI回归测试
python tests/test_ci_regression.py
```

## 许可证

MIT License
