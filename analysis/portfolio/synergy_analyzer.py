"""
Synergy Analyzer for SPL-4b

Analyzes portfolio-level worst-case synergies between strategies.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Set
from pandas import DataFrame
import pandas as pd
import numpy as np

from analysis.portfolio.portfolio_builder import Portfolio
from analysis.portfolio.portfolio_scanner import PortfolioWindowMetrics


@dataclass
class SynergyRiskReport:
    """B3: Report on portfolio-level synergy risks"""
    unsafe_combinations: List[Tuple[str, str]]  # Strategy pairs that fail together
    correlation_spike_periods: List[Dict]  # Periods with high correlation
    simultaneous_tail_losses: List[Dict]  # Periods with multiple tail losses
    risk_budget_breaches: List[Dict]  # Portfolio risk budget violations

    # Summary metrics
    total_pairs_analyzed: int
    unsafe_pair_count: int
    avg_correlation_baseline: float
    avg_correlation_stress: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "unsafe_combinations": [
                {"strategy1": s1, "strategy2": s2}
                for s1, s2 in self.unsafe_combinations
            ],
            "correlation_spike_periods": self.correlation_spike_periods,
            "simultaneous_tail_losses": self.simultaneous_tail_losses,
            "risk_budget_breaches": self.risk_budget_breaches,
            "total_pairs_analyzed": self.total_pairs_analyzed,
            "unsafe_pair_count": self.unsafe_pair_count,
            "avg_correlation_baseline": self.avg_correlation_baseline,
            "avg_correlation_stress": self.avg_correlation_stress
        }


class SynergyAnalyzer:
    """B3: Analyze portfolio-level worst-case synergies

    Identifies risks that emerge when combining strategies:
    1. Correlation spikes during stress periods
    2. Simultaneous tail losses
    3. Risk budget breaches
    """

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        tail_loss_threshold: float = -0.05
    ):
        """Initialize synergy analyzer

        Args:
            correlation_threshold: Correlation level considered "high" (default 0.7)
            tail_loss_threshold: Return level considered "tail loss" (default -5%)
        """
        self.correlation_threshold = correlation_threshold
        self.tail_loss_threshold = tail_loss_threshold

    def analyze_correlation_dynamics(
        self,
        portfolio: Portfolio,
        worst_window: PortfolioWindowMetrics
    ) -> Dict[str, float]:
        """Calculate correlation matrix during worst window

        Identifies strategies with spiked correlation.

        Args:
            portfolio: Portfolio to analyze
            worst_window: Worst-case window

        Returns:
            Dictionary mapping (strategy1, strategy2) to correlation
        """
        # Get portfolio data
        df = portfolio.to_dataframe()

        if df.empty:
            return {}

        # Filter to worst window period
        start = pd.Timestamp(worst_window.start_date)
        end = pd.Timestamp(worst_window.end_date)
        window_df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

        # Calculate correlations
        correlations = {}

        for i, s1 in enumerate(portfolio.config.strategy_ids):
            for j, s2 in enumerate(portfolio.config.strategy_ids):
                if i >= j:
                    continue  # Only upper triangle

                col1 = f"{s1}_return"
                col2 = f"{s2}_return"

                if col1 in window_df.columns and col2 in window_df.columns:
                    corr = window_df[col1].corr(window_df[col2])

                    if not pd.isna(corr):
                        correlations[f"{s1}__{s2}"] = float(corr)

        return correlations

    def identify_simultaneous_tail_losses(
        self,
        portfolio: Portfolio,
        worst_windows: List[PortfolioWindowMetrics]
    ) -> List[Dict]:
        """Find periods where multiple strategies enter tail loss together

        Args:
            portfolio: Portfolio to analyze
            worst_windows: List of worst windows to analyze

        Returns:
            List of simultaneous tail loss events
        """
        events = []

        for window in worst_windows:
            # Count strategies in tail loss
            tail_loss_strategies = []

            for strategy_id, contribution in window.strategy_contributions.items():
                if contribution <= self.tail_loss_threshold:
                    tail_loss_strategies.append({
                        "strategy_id": strategy_id,
                        "contribution": contribution
                    })

            # If multiple strategies in tail loss, record event
            if len(tail_loss_strategies) >= 2:
                events.append({
                    "window_id": window.window_id,
                    "start_date": window.start_date.isoformat(),
                    "end_date": window.end_date.isoformat(),
                    "strategies_in_tail_loss": tail_loss_strategies,
                    "count": len(tail_loss_strategies),
                    "portfolio_return": window.window_return
                })

        # Sort by count (most strategies first)
        events.sort(key=lambda e: e["count"], reverse=True)

        return events

    def check_risk_budget_breach(
        self,
        portfolio: Portfolio,
        worst_windows: List[PortfolioWindowMetrics],
        risk_budget: float = -0.10
    ) -> List[Dict]:
        """Check if portfolio worst-case exceeds risk budget

        Args:
            portfolio: Portfolio to analyze
            worst_windows: List of worst windows
            risk_budget: Maximum acceptable loss (default -10%)

        Returns:
            List of risk budget breaches
        """
        breaches = []

        for window in worst_windows:
            if window.window_return < risk_budget:
                breaches.append({
                    "window_id": window.window_id,
                    "start_date": window.start_date.isoformat(),
                    "end_date": window.end_date.isoformat(),
                    "portfolio_return": window.window_return,
                    "risk_budget": risk_budget,
                    "excess_loss": window.window_return - risk_budget,
                    "worst_performers": window.worst_performers,
                    "strategy_contributions": window.strategy_contributions
                })

        # Sort by excess loss (worst first)
        breaches.sort(key=lambda b: b["excess_loss"])

        return breaches

    def generate_synergy_report(
        self,
        portfolio: Portfolio,
        worst_windows: Dict[str, List[PortfolioWindowMetrics]],
        risk_budget: float = -0.10
    ) -> SynergyRiskReport:
        """Generate complete synergy risk report

        Args:
            portfolio: Portfolio to analyze
            worst_windows: Worst windows by length
            risk_budget: Risk budget threshold

        Returns:
            SynergyRiskReport
        """
        # Collect all worst windows
        all_windows = []
        for windows_list in worst_windows.values():
            all_windows.extend(windows_list)

        # Calculate baseline correlations (normal periods)
        baseline_correlations = self._calculate_baseline_correlations(portfolio)

        # Identify unsafe combinations
        unsafe_combinations = set()

        # Check each worst window
        for window in all_windows:
            # Get correlations during this window
            window_correlations = self.analyze_correlation_dynamics(
                portfolio,
                window
            )

            # Find pairs above threshold
            for pair_key, corr in window_correlations.items():
                if corr >= self.correlation_threshold:
                    s1, s2 = pair_key.split("__")
                    unsafe_combinations.add((s1, s2))

        # Find correlation spike periods
        correlation_spike_periods = []
        for window in all_windows:
            window_correlations = self.analyze_correlation_dynamics(
                portfolio,
                window
            )

            spikes = [
                {"pair": pair, "correlation": corr}
                for pair, corr in window_correlations.items()
                if corr >= self.correlation_threshold
            ]

            if spikes:
                correlation_spike_periods.append({
                    "window_id": window.window_id,
                    "start_date": window.start_date.isoformat(),
                    "end_date": window.end_date.isoformat(),
                    "spikes": spikes
                })

        # Find simultaneous tail losses
        simultaneous_tail_losses = self.identify_simultaneous_tail_losses(
            portfolio,
            all_windows
        )

        # Check risk budget breaches
        risk_budget_breaches = self.check_risk_budget_breach(
            portfolio,
            all_windows,
            risk_budget
        )

        # Calculate summary metrics
        strategy_ids = portfolio.config.strategy_ids
        total_pairs = len(strategy_ids) * (len(strategy_ids) - 1) // 2

        # Average correlations
        baseline_corr_values = list(baseline_correlations.values())
        avg_correlation_baseline = (
            np.mean(baseline_corr_values) if baseline_corr_values else 0.0
        )

        # Calculate stress correlations
        stress_corr_values = []
        for window in all_windows:
            window_corrs = self.analyze_correlation_dynamics(portfolio, window)
            stress_corr_values.extend(window_corrs.values())

        avg_correlation_stress = (
            np.mean(stress_corr_values) if stress_corr_values else 0.0
        )

        return SynergyRiskReport(
            unsafe_combinations=list(unsafe_combinations),
            correlation_spike_periods=correlation_spike_periods,
            simultaneous_tail_losses=simultaneous_tail_losses,
            risk_budget_breaches=risk_budget_breaches,
            total_pairs_analyzed=total_pairs,
            unsafe_pair_count=len(unsafe_combinations),
            avg_correlation_baseline=avg_correlation_baseline,
            avg_correlation_stress=avg_correlation_stress
        )

    def _calculate_baseline_correlations(
        self,
        portfolio: Portfolio
    ) -> Dict[str, float]:
        """Calculate baseline correlations (full period)

        Args:
            portfolio: Portfolio to analyze

        Returns:
            Dictionary mapping pair to correlation
        """
        df = portfolio.to_dataframe()

        if df.empty:
            return {}

        correlations = {}

        for i, s1 in enumerate(portfolio.config.strategy_ids):
            for j, s2 in enumerate(portfolio.config.strategy_ids):
                if i >= j:
                    continue

                col1 = f"{s1}_return"
                col2 = f"{s2}_return"

                if col1 in df.columns and col2 in df.columns:
                    corr = df[col1].corr(df[col2])

                    if not pd.isna(corr):
                        correlations[f"{s1}__{s2}"] = float(corr)

        return correlations


if __name__ == "__main__":
    """Test synergy analyzer"""
    print("=== SynergyAnalyzer Test ===\n")

    analyzer = SynergyAnalyzer(
        correlation_threshold=0.7,
        tail_loss_threshold=-0.05
    )

    print(f"Synergy Analyzer initialized:")
    print(f"  Correlation threshold: {analyzer.correlation_threshold}")
    print(f"  Tail loss threshold: {analyzer.tail_loss_threshold}")

    print("\n✓ Test passed")
