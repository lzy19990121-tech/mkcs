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
├── broker/         # 模拟交易执行
├── agent/          # 任务编排和状态管理
├── storage/        # 数据持久化
├── reports/        # 报告生成
└── tests/          # 测试用例
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行回测

```bash
python -m agent.runner --start 2024-01-01 --end 2024-01-31
```

### 测试各个模块

每个模块都有独立的main函数用于自测：

```bash
python -m skills.market_data.mock_source
python -m skills.strategy.moving_average
python -m skills.risk.basic_risk
python -m broker.paper
```

## 开发状态

- [x] Phase 1: 基础设施
- [x] Phase 2: 数据层
- [x] Phase 3: 业务逻辑层
- [x] Phase 4: 编排层
- [x] Phase 5: 集成测试

## 许可证

MIT License
