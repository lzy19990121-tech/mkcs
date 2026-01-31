"""
Portfolio Analysis Module for SPL-4b

Provides portfolio-level risk analysis capabilities:
- Portfolio construction from multiple strategies
- Portfolio worst-case window scanning
- Synergy risk analysis (correlation spikes, simultaneous tail losses)
- Portfolio risk report generation
"""

from analysis.portfolio.portfolio_builder import (
    PortfolioConfig,
    Portfolio,
    PortfolioBuilder,
    create_equal_weight_portfolio
)

from analysis.portfolio.portfolio_scanner import (
    PortfolioWindowMetrics,
    PortfolioWindowScanner
)

from analysis.portfolio.synergy_analyzer import (
    SynergyRiskReport,
    SynergyAnalyzer
)

from analysis.portfolio.portfolio_risk_report import (
    PortfolioRiskReportGenerator
)

__all__ = [
    # Portfolio Builder
    "PortfolioConfig",
    "Portfolio",
    "PortfolioBuilder",
    "create_equal_weight_portfolio",

    # Portfolio Scanner
    "PortfolioWindowMetrics",
    "PortfolioWindowScanner",

    # Synergy Analyzer
    "SynergyRiskReport",
    "SynergyAnalyzer",

    # Report Generator
    "PortfolioRiskReportGenerator"
]
