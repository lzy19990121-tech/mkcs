"""
策略分析模块

提供市场状态检测、仓位管理、动态止盈止���等功能
"""

from skills.analysis.regime_detector import (
    RegimeDetector,
    RegimeDetectorConfig,
    RegimeState,
    RegimeType,
    RegimeInfo,
    is_regime_allowed,
    get_regime_position_modifier
)

from skills.analysis.position_sizing import (
    PositionSizer,
    PositionConfig,
    RiskParitySizer,
    KellySizer
)

from skills.analysis.dynamic_stops import (
    DynamicStopManager,
    PositionTracker,
    StopLevel,
    ExitReason,
    TimeBasedExit
)

__all__ = [
    # Regime Detection
    'RegimeDetector',
    'RegimeDetectorConfig',
    'RegimeState',
    'RegimeType',
    'RegimeInfo',
    'is_regime_allowed',
    'get_regime_position_modifier',

    # Position Sizing
    'PositionSizer',
    'PositionConfig',
    'RiskParitySizer',
    'KellySizer',

    # Dynamic Stops
    'DynamicStopManager',
    'PositionTracker',
    'StopLevel',
    'ExitReason',
    'TimeBasedExit',
]
