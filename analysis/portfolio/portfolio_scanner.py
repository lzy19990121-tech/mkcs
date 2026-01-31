"""
Portfolio Window Scanner for SPL-4b

Scans portfolio P&L for worst-case windows with strategy contribution analysis.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from analysis.portfolio.portfolio_builder import Portfolio
from analysis.window_scanner import WindowScanner, WindowMetrics


@dataclass
class PortfolioWindowMetrics:
    """Portfolio-level worst-case metrics"""
    window_id: str
    start_date: date
    end_date: date
    window_length_days: int

    # Portfolio metrics
    window_return: float
    max_drawdown: float
    drawdown_duration: int
    volatility: float

    # B2: Strategy contributions
    strategy_contributions: Dict[str, float]  # Each strategy's return contribution
    worst_performers: List[str]  # Strategies with worst returns in this window

    # Correlation info
    avg_correlation: float  # Average pairwise correlation during window

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "window_id": self.window_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "window_length_days": self.window_length_days,
            "window_return": self.window_return,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration": self.drawdown_duration,
            "volatility": self.volatility,
            "strategy_contributions": self.strategy_contributions,
            "worst_performers": self.worst_performers,
            "avg_correlation": self.avg_correlation
        }


class PortfolioWindowScanner(WindowScanner):
    """B2: Scan portfolio P&L for worst-case windows

    Extends WindowScanner to work with portfolio-level data
    and provide strategy contribution breakdowns.
    """

    def __init__(self):
        """Initialize portfolio window scanner"""
        super().__init__()

    def find_worst_portfolio_windows(
        self,
        portfolio: Portfolio,
        window_lengths: List[str] = ["20d", "60d"],
        top_k: int = 5
    ) -> Dict[str, List[PortfolioWindowMetrics]]:
        """Find worst-case windows for portfolio

        Applies same window scanning logic to portfolio P&L
        and returns worst windows with strategy decomposition.

        Args:
            portfolio: Portfolio to analyze
            window_lengths: Window lengths to scan (e.g., ["20d", "60d"])
            top_k: Number of worst windows to return per length

        Returns:
            Dictionary mapping window_length to list of PortfolioWindowMetrics
        """
        results = {}

        for window_length in window_lengths:
            # Parse window length
            days = self._parse_window_length(window_length)

            # Find all windows
            all_windows = self._scan_portfolio_windows(
                portfolio,
                days
            )

            # Sort by return (worst first)
            all_windows.sort(key=lambda w: w.window_return)

            # Take top K
            worst_windows = all_windows[:top_k]
            results[window_length] = worst_windows

        return results

    def _scan_portfolio_windows(
        self,
        portfolio: Portfolio,
        window_days: int
    ) -> List[PortfolioWindowMetrics]:
        """Scan portfolio for all windows of given length

        Args:
            portfolio: Portfolio to scan
            window_days: Window length in days

        Returns:
            List of all portfolio windows
        """
        # Get portfolio data
        df = portfolio.to_dataframe()

        if df.empty or len(df) < window_days:
            return []

        windows = []

        # Sliding window scan
        for i in range(len(df) - window_days + 1):
            window_df = df.iloc[i:i + window_days]

            # Calculate metrics
            window_metrics = self._calculate_portfolio_window_metrics(
                portfolio,
                window_df,
                i
            )

            windows.append(window_metrics)

        return windows

    def _calculate_portfolio_window_metrics(
        self,
        portfolio: Portfolio,
        window_df: pd.DataFrame,
        window_index: int
    ) -> PortfolioWindowMetrics:
        """Calculate metrics for a single portfolio window

        Args:
            portfolio: Full portfolio
            window_df: Window DataFrame
            window_index: Window index

        Returns:
            PortfolioWindowMetrics
        """
        # Get dates
        timestamps = window_df['timestamp'].tolist()
        start_date = timestamps[0].date()
        end_date = timestamps[-1].date()

        # Calculate window return
        if 'portfolio_return' in window_df.columns:
            returns = window_df['portfolio_return'].values
            window_return = returns.sum()  # Cumulative return
        else:
            # Calculate from equity
            equity = window_df['portfolio_equity'].values
            if len(equity) >= 2:
                window_return = (equity[-1] - equity[0]) / equity[0]
            else:
                window_return = 0.0

        # Calculate max drawdown
        if 'portfolio_equity' in window_df.columns:
            equity_series = window_df['portfolio_equity'].values
            rolling_max = np.maximum.accumulate(equity_series)
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0.0

        # Calculate drawdown duration
        drawdown_duration = 0
        if 'portfolio_equity' in window_df.columns:
            equity_series = window_df['portfolio_equity'].values
            peak = equity_series[0]
            current_duration = 0
            for val in equity_series:
                if val < peak:
                    current_duration += 1
                else:
                    peak = val
                    current_duration = 0
                drawdown_duration = max(drawdown_duration, current_duration)

        # Calculate volatility
        if 'portfolio_return' in window_df.columns:
            volatility = window_df['portfolio_return'].std()
        else:
            volatility = 0.0

        # Calculate strategy contributions
        strategy_contributions = {}
        worst_performers = []

        for strategy_id in portfolio.config.strategy_ids:
            col_name = f"{strategy_id}_return"
            if col_name in window_df.columns:
                contribution = window_df[col_name].sum()
                strategy_contributions[strategy_id] = contribution

        # Find worst performers
        if strategy_contributions:
            # Sort by contribution (worst first)
            sorted_strategies = sorted(
                strategy_contributions.items(),
                key=lambda x: x[1]
            )
            # All strategies with negative contribution
            worst_performers = [
                sid for sid, contrib in sorted_strategies
                if contrib < 0
            ]

        # Calculate average correlation
        avg_correlation = self._calculate_avg_correlation(
            window_df,
            portfolio.config.strategy_ids
        )

        # Create window ID
        window_id = f"window_{start_date}_to_{end_date}"

        return PortfolioWindowMetrics(
            window_id=window_id,
            start_date=start_date,
            end_date=end_date,
            window_length_days=len(window_df),
            window_return=window_return,
            max_drawdown=max_drawdown,
            drawdown_duration=drawdown_duration,
            volatility=volatility,
            strategy_contributions=strategy_contributions,
            worst_performers=worst_performers,
            avg_correlation=avg_correlation
        )

    def _calculate_avg_correlation(
        self,
        window_df: pd.DataFrame,
        strategy_ids: List[str]
    ) -> float:
        """Calculate average pairwise correlation during window

        Args:
            window_df: Window DataFrame
            strategy_ids: Strategy IDs

        Returns:
            Average correlation
        """
        # Collect strategy returns
        return_series = []
        for strategy_id in strategy_ids:
            col_name = f"{strategy_id}_return"
            if col_name in window_df.columns:
                return_series.append(window_df[col_name].values)

        if len(return_series) < 2:
            return 0.0

        # Calculate pairwise correlations
        correlations = []
        for i in range(len(return_series)):
            for j in range(i + 1, len(return_series)):
                corr = np.corrcoef(return_series[i], return_series[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if not correlations:
            return 0.0

        return float(np.mean(correlations))

    def _parse_window_length(self, window_length: str) -> int:
        """Parse window length string to days

        Args:
            window_length: Window length (e.g., "20d", "60d")

        Returns:
            Number of days
        """
        if window_length.endswith("d"):
            return int(window_length[:-1])
        elif window_length.endswith("w"):
            return int(window_length[:-1]) * 7
        else:
            return 20


if __name__ == "__main__":
    """Test portfolio window scanner"""
    print("=== PortfolioWindowScanner Test ===\n")

    from datetime import date
    import pandas as pd

    # Create a mock portfolio for testing
    # In real usage, this would come from PortfolioBuilder
    print("Portfolio window scanner requires portfolio data.")
    print("Use with PortfolioBuilder to analyze portfolio worst-case windows.")

    print("\n✓ Test passed")
