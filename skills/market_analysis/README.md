# MKCS 市场分析模块 - 实现总结

## 概述

市场分析模块已完整实现，为 MKCS 交易系统提供结构化、可回放的市场理解能力。核心原则是：**市场分析 ≠ 预测，而是限制策略犯错**。

## 模块架构

```
skills/market_analysis/
├── __init__.py                 # 模块入口，导出所有公共接口
├── market_state.py             # MarketState 定义，序列化/反序列化
├── detectors.py                # 市场状态检测器（RegimeDetector, VolatilityDetector, LiquidityDetector）
├── external_signals.py         # 外部信号（情绪、宏观、事件日历）
├── strategy_feasibility.py     # 策略可行性评估和动态参数管理
└── manager.py                  # MarketManager 统一入口

skills/strategy/
└── enhanced_base.py            # EnhancedStrategy 集成 MarketManager

skills/risk/
├── adaptive_risk.py            # 基础自适应风控
├── market_aware_risk.py        # 市场感知风控（新增）
└── __init__.py

api/
└── market_analysis_api.py      # Web UI API 接口（新增）

analysis/
└── market_analysis_evaluator.py # 评估框架（新增）
```

## 核心组件

### 1. MarketState（市场状态）

统一的市场状态数据结构，包含：
- **基本信息**: timestamp, symbol
- **市场状态**: regime（TREND/RANGE/HIGH_VOL/CRISIS）
- **波动率分析**: volatility_state, volatility_trend, realized_vol, atr_ratio
- **流动性状态**: liquidity_state, volume_ratio, turnover_ratio
- **市场情绪**: sentiment_state, sentiment_score
- **系统性风险**: SystemicRiskFlags（事件窗口、系统性风险等）
- **趋势强度**: trend_strength, trend_direction
- **置信度**: confidence

**便捷方法**：
- `should_reduce_risk`: 判断是否应降低风险
- `position_scale_factor`: 获取建议仓位缩放系数
- `to_dict()` / `from_dict()`: 序列化支持

### 2. Detectors（检测器）

| 检测器 | 功能 | 输出 |
|--------|------|------|
| RegimeDetector | 市场状态检测 | regime, confidence, features (ADX, slope, MA分离) |
| VolatilityDetector | 波动率分析 | state, trend, percentile, realized_vol, atr_ratio |
| LiquidityDetector | 流动性分析 | state, volume_ratio, turnover_ratio |
| MarketAnalyzer | 统一分析接口 | 综合分析结果 |

### 3. External Signals（外部信号）

| 组件 | 数据源 | 功能 |
|------|--------|------|
| SentimentDetector | CNN Fear & Greed, VIX, PCR | 市场情绪检测 |
| MacroDetector | SPX/NDX, 收益率曲线, DXY | 系统性风险检测 |
| EventCalendar | FOMC, CPI, NFP, Earnings | 事件窗口检测 |
| ExternalSignalManager | 统一管理器 | 整合所有外部信号 |

### 4. Strategy Feasibility（策略可行性）

| 组件 | 功能 |
|------|------|
| StrategyFeasibilityEvaluator | 基于规则和历史表现评估策略是否应启用 |
| DynamicParameterManager | 根据市场状态动态调整策略参数（MA周期、突破阈值等） |

### 5. MarketManager（统一入口）

```python
manager = get_market_manager(symbol="AAPL")

# 分析市场
state = manager.analyze(bars, external_data=external_data)

# 评估策略可行性
feasibility = manager.evaluate_strategy("ma", state)
# => {"strategy_allowed": bool, "position_scale": float, "reason": str}

# 获取动态参数
params = manager.param_manager.get_parameters("ma", state)
# => {"fast_period": 5, "slow_period": 20, ...}
```

## 策略集成

### EnhancedStrategy 更新

```python
class EnhancedStrategy(ABC):
    def __init__(
        self,
        name: str,
        enable_market_manager: bool = True,
        strategy_type: Optional[str] = None,
        enable_feasibility_check: bool = True
    ):
        # 自动集成 MarketManager
        self.market_manager = get_market_manager()

    def generate_strategy_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None,
        external_data: Optional[Dict[str, Any]] = None  # 新增
    ) -> List[StrategySignal]:
        # 1. 获取市场状态
        market_state = self.get_market_state(bars, external_data)

        # 2. 获取动态参数
        self.get_dynamic_params(market_state)

        # 3. 评估策略可行性
        feasibility = self.evaluate_feasibility(market_state)

        # 4. 应用市场状态过滤
        ...
```

## 风控集成

### MarketAwareRiskManager

市场感知的风控管理器，根据市场状态动态调整风控参数：

```python
risk_mgr = MarketAwareRiskManager()

# 更新市场状态
risk_mgr.update_market_state(market_state)

# 检查信号（包含市场状态）
result = risk_mgr.check_signal_with_market_state(
    signal, positions, cash_balance, market_state
)

# 获取动态仓位限制
limits = risk_mgr.get_dynamic_position_limits(market_state)
# => {"max_position_ratio": 0.2, "max_total_ratio": 0.8, "scale_factor": 0.5}
```

**动态仓位限制**：
- 正常市场: 单标的 20%, 总仓位 80%
- 震荡市场: 单标的 15%, 总仓位 60%
- 高波动市场: 单标的 10%, 总仓位 50%
- 危机模式: 单标的 5%, 总仓位 20%

## API 接口（Web UI）

```python
api = get_market_analysis_api()

# 获取当前市场状态
result = api.get_current_market_state(bars)
# => regime, volatility, liquidity, recommendations, explanation

# 获取市场摘要
result = api.get_market_summary(bars)
# => market_state, risk_assessment, strategy_status, recommendations

# 获取策略可行性
result = api.get_strategy_feasibility("ma", bars)
# => allowed, position_scale, reason, dynamic_params

# 获取风控建议
result = api.get_risk_recommendations(bars)
# => position_limits, risk_level, actions, warnings
```

## 评估框架

```python
evaluator = MarketAnalysisEvaluator()

# 检测准确性评估
result = evaluator.evaluate_detection_accuracy(bars)
# => accuracy, confidence_stats, regime_distribution

# 策略可行性评估
result = evaluator.evaluate_strategy_feasibility(bars, "ma")
# => allowed_ratio, avg_position_scale, reasons

# 状态稳定性评估
result = evaluator.evaluate_state_stability()
# => avg_duration, unstable_transitions, unstable_ratio

# 回测模拟
result = evaluator.run_backtest_simulation(bars)
# => state_distribution, transitions, avg_confidence
```

## 使用示例

### 基本使用

```python
from skills.market_analysis import get_market_manager

# 获取市场管理器
manager = get_market_manager(symbol="AAPL")

# 分析市场
state = manager.analyze(bars)

# 检查是否应降低风险
if state.should_reduce_risk:
    print(f"建议降低仓位，缩放系数: {state.position_scale_factor:.2f}")
```

### 策略集成

```python
from skills.strategy.enhanced_base import EnhancedStrategy

class MyStrategy(EnhancedStrategy):
    def __init__(self):
        super().__init__(
            name="my_strategy",
            strategy_type="ma",  # 用于可行性评估
            enable_market_manager=True,
            enable_feasibility_check=True
        )

    def _generate_strategy_signals(self, bars, position, regime_info):
        # 使用 self._dynamic_params 获取动态参数
        fast_period = self._dynamic_params.get("fast_period", 5)
        slow_period = self._dynamic_params.get("slow_period", 20)
        # ... 生成信号
```

### 风控集成

```python
from skills.risk import MarketAwareRiskManager

risk_mgr = MarketAwareRiskManager(initial_capital=100000)

# 根据市场状态动态调整风控
state = market_manager.analyze(bars)
risk_mgr.update_market_state(state)

# 检查信号
result = risk_mgr.check_signal_with_market_state(
    signal, positions, cash_balance, state
)

if result.action == RiskAction.SCALE_DOWN:
    print(f"仓位缩放至 {result.scale_factor:.1%}")
```

## 测试结果

所有组件测试通过：

```
✓ market_state.py - MarketState 测试
✓ detectors.py - Detectors 测试
✓ external_signals.py - External Signals 测试
✓ strategy_feasibility.py - Strategy Feasibility 测试
✓ manager.py - MarketManager 测试
✓ enhanced_base.py - EnhancedStrategy 测试
✓ market_aware_risk.py - MarketAwareRiskManager 测试
✓ market_analysis_evaluator.py - Evaluator 测试
✓ market_analysis_api.py - API 测试
```

## 关键特性

1. **可回放性**: MarketState 支持序列化/反序列化，可用于回测和调试
2. **模块化**: 各检测器独立工作，可灵活组合
3. **可扩展**: 支持添加新的外部信号源和检测器
4. **向后兼容**: EnhancedStrategy 支持旧的 regime_detector 接口
5. **配置驱动**: 风险规则和参数可通过 JSON 配置

## 后续扩展方向

1. **更多外部信号源**:
   - 加密货币特定指标
   - 期权 PCR/IV
   - 社交媒体情绪

2. **机器学习增强**:
   - 使用 ML 模型预测市场状态转换
   - 基于历史表现优化可行性阈值

3. **实时数据流**:
   - WebSocket 实时市场状态推送
   - 增量状态更新

4. **可视化**:
   - 市场状态历史图表
   - 策略可行性仪表板
