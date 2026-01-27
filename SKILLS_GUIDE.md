# 自动交易系统 - Skills使用指南

## 已创建的Claude Code Skills

我们成功将自动交易系统包装成了2个可重用的Claude Code Skills：

### 1. trading-backtest-runner.skill (8.8KB)

**用途**: 完整的回测系统运行器

**包含内容**:
- ✅ 完整的系统架构说明
- ✅ 核心组件API文档
- ✅ 快速开始指南
- ✅ 回测运行命令
- ✅ 数据库集成方法
- ✅ 模块测试方法
- ✅ 扩展指南（新策略、新数据源、新风控规则）
- ✅ 执行脚本
  - `run_quick_backtest.sh` - 快速回测脚本
  - `run_integration_tests.sh` - 集成测试脚本
- ✅ API参考文档 (references/api_reference.md)

**使用场景**:
当用户需要：
- 运行交易回测
- 测试交易策略
- 分析历史表现
- 生成交易报告
- 用历史数据验证策略

**示例触发**:
```
用户: "运行2024年1月的回测"
用户: "测试移动平均线策略"
用户: "生成交易报告"
```

### 2. trading-strategy-analyzer.skill (2.5KB)

**用途**: 策略分析和性能优化工具

**包含内容**:
- ✅ 策略接口定义
- ✅ 移动平均线策略详解
- ✅ 性能指标计算
  - 总收益率
  - 夏普比率
  - 最大回撤
- ✅ 参数优化方法
  - 网格搜索
  - Walk-Forward优化
- ✅ 信号分析工具
- ✅ 测试方法

**使用场景**:
当用户需要：
- 分析交易信号
- 计算策略性能指标
- 优化策略参数
- 比较多个策略
- 计算夏普比率和回撤

**示例触发**:
```
用户: "计算这个策略的夏普比率"
用户: "优化移动平均线参数"
用户: "分析这些信号的置信度"
```

## Skills vs 原始代码

### 优势

1. **可重用性**: Skills可以在不同对话中重复使用，无需重新解释
2. **上下文高效**: 只在需要时加载详细内容，节省token
3. **标准化**: 提供标准的使用模式和最佳实践
4. **可扩展**: 易于添加新的策略和功能

### 架构对应

```
原始项目结构                  →   Skills
─────────────────────────────────────────────
agent/runner.py              →   trading-backtest-runner
  - 完整回测系统                  - 运行回测
  - Agent编排                    - 系统架构说明
  - 报告生成                      - 快速开始指南

skills/strategy/              →   trading-strategy-analyzer
  - MAStrategy                   - 策略分析
  - 基类接口                      - 性能计算
  - 信号生成                      - 参数优化
```

## 使用示例

### 在对话中自动触发

**场景1**: 用户要求运行回测

```
用户: "我想测试一下AAPL和GOOGL在2024年的表现"
Claude: [自动加载 trading-backtest-runner skill]
      → 运行回测命令
      → 生成报告
```

**场景2**: 用户需要优化策略

```
用户: "帮我找到最佳的移动平均线参数组合"
Claude: [自动加载 trading-strategy-analyzer skill]
      → 执行网格搜索
      → 返回最优参数
```

**场景3**: 用户分析性能

```
用户: "计算这个策略的夏普比率和最大回撤"
Claude: [自动加载 trading-strategy-analyzer skill]
      → 使用内置函数计算
      → 返回性能指标
```

## 安装Skills

### 方法1: 手动安装（推荐用于测试）

```bash
# 复制.skill文件到Claude的skills目录
cp /home/gushengdong/mkcs/*.skill ~/.claude/skills/

# 或者安装到全局skills目录
sudo cp /home/gushengdong/mkcs/*.skill /opt/claude/skills/
```

### 方法2: 通过UI安装

1. 打开Claude Code
2. 进入设置 → Skills
3. 点击 "Install Skill"
4. 选择 `.skill` 文件

## Skill内容详解

### trading-backtest-runner

**SKILL.md结构**:
```markdown
1. Quick Start - 3个命令立即开始
2. Core Architecture - 系统组件关系
3. System Components - 7个核心模块详解
4. Running Backtests - 使用方法和参数
5. Testing Individual Modules - 单元测试
6. Example Output - 预期输出示例
7. Extending the System - 扩展指南
8. Design Principles - 设计原则
9. Database Schema - 数据库结构
10. Performance Metrics - 性能指标
```

**scripts/**:
- `run_quick_backtest.sh`: 一键运行示例回测
- `run_integration_tests.sh`: 运行完整测试套件

**references/**:
- `api_reference.md`: 完整API文档（所有类、方法、参数）

### trading-strategy-analyzer

**SKILL.md结构**:
```markdown
1. Quick Start - 3行代码开始分析
2. Strategy Interface - 基类定义
3. Available Strategies - MA策略详解
4. Performance Analysis - 3个核心指标
5. Strategy Optimization - 参数优化方法
6. Signal Analysis - 信号过滤和分析
7. Testing - 测试方法
```

## 实际应用案例

### 案例1: 策略开发流程

```bash
# 1. 用strategy-analyzer设计策略
→ 定义RSI策略
→ 计算理论性能

# 2. 用backtest-runner回测
→ 运行历史回测
→ 获取实际表现

# 3. 用strategy-analyzer优化
→ 网格搜索参数
→ 找到最优配置

# 4. 用backtest-runner验证
→ 运行样本外测试
→ 确认策略有效性
```

### 案例2: 性能分析

```python
# 运行回测
agent.run_backtest(start, end, symbols)

# 使用strategy-analyzer计算指标
from trading_strategy_analyzer import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    analyze_performance
)

metrics = analyze_performance(agent.broker)
sharpe = calculate_sharpe_ratio(returns)
drawdown = calculate_max_drawdown(equity_curve)

print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {drawdown['max_drawdown']:.2f}%")
```

### 案例3: 参数优化

```python
# 使用strategy-analyzer的优化函数
from trading_strategy_analyzer import optimize_ma_strategy

results = optimize_ma_strategy(
    data_source=MockMarketSource(),
    symbols=["AAPL", "GOOGL"],
    fast_range=range(5, 11),
    slow_range=range(20, 31)
)

# 显示最优参数
best = results[0]
print(f"Best: MA({best['fast']}, {best['slow']}) = {best['return']:.2f}%")
```

## 最佳实践

### 1. 选择合适的Skill

- **运行完整回测** → 使用 `trading-backtest-runner`
- **分析策略性能** → 使用 `trading-strategy-analyzer`
- **开发新策略** → 先用 `analyzer` 设计，再用 `runner` 测试

### 2. 组合使用

两个Skills可以相互配合：
```python
# Step 1: 用analyzer设计策略
strategy = create_rsi_strategy()

# Step 2: 用runner测试
agent = TradingAgent(
    data_source=source,
    strategy=strategy,  # 来自analyzer
    risk_manager=risk,
    broker=broker
)

# Step 3: 用analyzer分析结果
metrics = analyze_performance(agent.broker)
```

### 3. 扩展Skills

在 `trading-backtest-runner` 中有完整的扩展指南：
- 添加新策略（RSI, 布林带等）
- 添加真实数据源（yfinance, Alpha Vantage）
- 添加自定义风控规则

## 技术细节

### Token使用

- **trading-backtest-runner**:
  - Frontmatter: ~50 tokens
  - SKILL.md body: ~3,000 tokens (仅在触发时加载)
  - references/api_reference.md: ~4,000 tokens (按需加载)

- **trading-strategy-analyzer**:
  - Frontmatter: ~40 tokens
  - SKILL.md body: ~1,500 tokens (仅在触发时加载)

### 触发机制

Skills通过description中的关键词自动触发：

**trading-backtest-runner**:
- "backtest", "回测"
- "trading system", "交易系统"
- "run strategy", "运行策略"
- "performance report", "性能报告"

**trading-strategy-analyzer**:
- "optimize strategy", "优化策略"
- "Sharpe ratio", "夏普比率"
- "signal analysis", "信号分析"
- "performance metrics", "性能指标"

## 总结

通过创建这两个Skills，我们实现了：

✅ **代码复用**: 核心逻辑封装为可重用组件
✅ **知识传承**: 最佳实践和经验文档化
✅ **效率提升**: 自动加载相关代码和文档
✅ **易于扩展**: 清晰的扩展指南
✅ **标准化**: 统一的使用模式

现在，在任何对话中，当用户提到相关需求时，Claude会自动加载这些Skills，提供专业、完整的解决方案！
