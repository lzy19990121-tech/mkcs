"""
风险分析模块

提供最坏窗口扫描、稳定性分析、多策略对比和Risk Card生成功能
"""

from analysis.replay_schema import (
    ReplayOutput,
    StepRecord,
    TradeRecord,
    load_replay_outputs
)

from analysis.window_scanner import (
    WindowScanner,
    WindowMetrics,
    scan_all_replays
)

from analysis.stability_analysis import (
    StabilityAnalyzer,
    StabilityReport,
    analyze_all_stability
)

from analysis.multi_strategy_comparison import (
    MultiStrategyComparator,
    StrategyWorstCaseSummary,
    compare_all_strategies
)

from analysis.risk_card_generator import (
    RiskCardGenerator,
    RiskCardConfig,
    generate_risk_cards
)

__all__ = [
    # Schema
    'ReplayOutput',
    'StepRecord',
    'TradeRecord',
    'load_replay_outputs',

    # Window Scanner
    'WindowScanner',
    'WindowMetrics',
    'scan_all_replays',

    # Stability Analysis
    'StabilityAnalyzer',
    'StabilityReport',
    'analyze_all_stability',

    # Multi-Strategy Comparison
    'MultiStrategyComparator',
    'StrategyWorstCaseSummary',
    'compare_all_strategies',

    # Risk Card Generator
    'RiskCardGenerator',
    'RiskCardConfig',
    'generate_risk_cards',
]
