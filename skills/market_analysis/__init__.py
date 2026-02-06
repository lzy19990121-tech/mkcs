"""
市场分析模块

为策略提供结构化、可回放的市场理解
核心原则：市场分析 ≠ 预测，而是限制策略犯错
"""

from skills.market_analysis.market_state import (
    MarketState,
    MarketStateBuilder,
    RegimeType,
    VolatilityState,
    VolatilityTrend,
    LiquidityState,
    SentimentState,
    SystemicRiskFlags
)

from skills.market_analysis.detectors import (
    RegimeDetector,
    VolatilityDetector,
    LiquidityDetector,
    MarketAnalyzer
)

from skills.market_analysis.external_signals import (
    SentimentDetector,
    MacroDetector,
    EventCalendar,
    ExternalSignalManager
)

from skills.market_analysis.strategy_feasibility import (
    StrategyFeasibilityEvaluator,
    DynamicParameterManager
)

from skills.market_analysis.manager import (
    MarketManager,
    get_market_manager
)

__all__ = [
    # Market State
    'MarketState',
    'MarketStateBuilder',
    'RegimeType',
    'VolatilityState',
    'VolatilityTrend',
    'LiquidityState',
    'SentimentState',
    'SystemicRiskFlags',

    # Detectors
    'RegimeDetector',
    'VolatilityDetector',
    'LiquidityDetector',
    'MarketAnalyzer',

    # External Signals
    'SentimentDetector',
    'MacroDetector',
    'EventCalendar',
    'ExternalSignalManager',

    # Strategy Integration
    'StrategyFeasibilityEvaluator',
    'DynamicParameterManager',

    # Manager
    'MarketManager',
    'get_market_manager',
]
