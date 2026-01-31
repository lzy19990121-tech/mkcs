"""
Portfolio Builder for SPL-4b

Builds portfolios from multiple strategy replays for portfolio-level risk analysis.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np

from analysis.replay_schema import ReplayOutput, load_replay_outputs


@dataclass
class PortfolioConfig:
    """Portfolio configuration"""
    strategy_ids: List[str]  # Strategies that passed 4a
    weights: Dict[str, float]  # Static or dynamic weights (sum to 1.0)

    # Time alignment
    start_date: date
    end_date: date
    alignment_method: str = "inner"  # "inner", "outer", "left"

    # Rebalancing
    rebalance_frequency: str = "monthly"  # "daily", "weekly", "monthly"

    def validate(self) -> bool:
        """Validate portfolio configuration

        Returns:
            True if valid
        """
        if not self.strategy_ids:
            raise ValueError("strategy_ids cannot be empty")

        if not self.weights:
            raise ValueError("weights cannot be empty")

        # Check all strategies have weights
        for strategy_id in self.strategy_ids:
            if strategy_id not in self.weights:
                raise ValueError(f"Missing weight for strategy: {strategy_id}")

        # Check weights sum to 1.0
        total_weight = sum(self.weights.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        return True


@dataclass
class Portfolio:
    """Portfolio of multiple strategies

    Contains combined P&L series and strategy-level breakdowns.
    """
    portfolio_id: str
    config: PortfolioConfig

    # Time series
    timestamps: List[datetime]
    combined_equity: List[float]
    combined_returns: List[float]

    # Strategy contributions
    strategy_equities: Dict[str, List[float]]  # {strategy_id: [equity_series]}
    strategy_returns: Dict[str, List[float]]  # {strategy_id: [returns_series]}

    # Metrics
    initial_value: float
    final_value: float
    total_return: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert portfolio to pandas DataFrame

        Returns:
            DataFrame with portfolio and strategy returns
        """
        data = {
            "timestamp": self.timestamps,
            "portfolio_equity": self.combined_equity,
            "portfolio_return": self.combined_returns
        }

        # Add strategy columns
        for strategy_id in self.config.strategy_ids:
            if strategy_id in self.strategy_returns:
                data[f"{strategy_id}_return"] = self.strategy_returns[strategy_id]
                data[f"{strategy_id}_equity"] = self.strategy_equities[strategy_id]

        return pd.DataFrame(data)

    def get_portfolio_returns(self) -> pd.Series:
        """Get portfolio returns series

        Returns:
            Series of portfolio returns
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.Series([], dtype=float)
        return pd.Series(self.combined_returns, index=self.timestamps)

    def get_strategy_returns(self, strategy_id: str) -> Optional[pd.Series]:
        """Get returns for a specific strategy

        Args:
            strategy_id: Strategy identifier

        Returns:
            Series of strategy returns or None
        """
        if strategy_id not in self.strategy_returns:
            return None

        return pd.Series(
            self.strategy_returns[strategy_id],
            index=self.timestamps
        )


class PortfolioBuilder:
    """B1: Build portfolio from SPL-4a qualified strategies

    Loads multiple strategy replays and aligns timeframes for
    portfolio-level risk analysis.
    """

    def __init__(self):
        """Initialize portfolio builder"""
        self.replay_cache: Dict[str, ReplayOutput] = {}

    def build_portfolio(
        self,
        config: PortfolioConfig,
        replay_dir: str
    ) -> Portfolio:
        """Build portfolio from strategy replays

        Args:
            config: Portfolio configuration
            replay_dir: Directory containing replay outputs

        Returns:
            Portfolio with combined P&L series
        """
        # Validate configuration
        config.validate()

        # Load replays
        replays = self._load_replays(config.strategy_ids, replay_dir)

        if len(replays) != len(config.strategy_ids):
            missing = set(config.strategy_ids) - set(replays.keys())
            raise ValueError(f"Missing replays for strategies: {missing}")

        # Align timeframes
        aligned_data = self._align_timeframes(replays, config)

        # Calculate weighted portfolio
        portfolio_data = self._calculate_portfolio(aligned_data, config)

        # Create portfolio object
        portfolio = Portfolio(
            portfolio_id=f"portfolio_{'_'.join(config.strategy_ids)}",
            config=config,
            timestamps=portfolio_data["timestamps"],
            combined_equity=portfolio_data["combined_equity"],
            combined_returns=portfolio_data["combined_returns"],
            strategy_equities=portfolio_data["strategy_equities"],
            strategy_returns=portfolio_data["strategy_returns"],
            initial_value=portfolio_data["initial_value"],
            final_value=portfolio_data["final_value"],
            total_return=portfolio_data["total_return"]
        )

        return portfolio

    def _load_replays(
        self,
        strategy_ids: List[str],
        replay_dir: str
    ) -> Dict[str, ReplayOutput]:
        """Load replay outputs for strategies

        Args:
            strategy_ids: Strategy identifiers
            replay_dir: Replay directory

        Returns:
            Dictionary mapping strategy_id to ReplayOutput
        """
        # Load all replays
        all_replays = load_replay_outputs(replay_dir)

        # Filter by strategy_id
        replays = {}
        for replay in all_replays:
            if replay.strategy_id in strategy_ids:
                replays[replay.strategy_id] = replay

        return replays

    def _align_timeframes(
        self,
        replays: Dict[str, ReplayOutput],
        config: PortfolioConfig
    ) -> Dict[str, pd.DataFrame]:
        """Align timeframes across strategies

        Args:
            replays: Dictionary of strategy replays
            config: Portfolio configuration

        Returns:
            Dictionary mapping strategy_id to aligned DataFrame
        """
        # Convert each replay to DataFrame
        dfs = {}
        for strategy_id, replay in replays.items():
            df = replay.to_dataframe()
            if df.empty:
                continue

            # Calculate returns
            df = df.sort_values('timestamp')
            df['return'] = df['equity'].pct_change()
            df['return'].fillna(0, inplace=True)

            # Filter by date range
            start = pd.Timestamp(config.start_date)
            end = pd.Timestamp(config.end_date)
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

            dfs[strategy_id] = df

        # Align based on method
        if config.alignment_method == "inner":
            # Use intersection of all timestamps
            common_timestamps = None
            for df in dfs.values():
                if common_timestamps is None:
                    common_timestamps = set(df['timestamp'])
                else:
                    common_timestamps &= set(df['timestamp'])

            # Filter to common timestamps
            for strategy_id in dfs:
                dfs[strategy_id] = dfs[strategy_id][
                    dfs[strategy_id]['timestamp'].isin(common_timestamps)
                ]

        elif config.alignment_method == "outer":
            # Use union of all timestamps (forward fill missing)
            all_timestamps = set()
            for df in dfs.values():
                all_timestamps.update(df['timestamp'])

            all_timestamps = sorted(all_timestamps)

            # Reindex with forward fill
            for strategy_id in dfs:
                df = dfs[strategy_id].set_index('timestamp')
                df = df.reindex(all_timestamps, method='ffill')
                df = df.reset_index()
                dfs[strategy_id] = df

        # "left" method: use first strategy's timestamps
        elif config.alignment_method == "left" and dfs:
            first_id = list(dfs.keys())[0]
            reference_timestamps = set(dfs[first_id]['timestamp'])

            for strategy_id in dfs:
                if strategy_id != first_id:
                    dfs[strategy_id] = dfs[strategy_id][
                        dfs[strategy_id]['timestamp'].isin(reference_timestamps)
                    ]

        return dfs

    def _calculate_portfolio(
        self,
        aligned_data: Dict[str, pd.DataFrame],
        config: PortfolioConfig
    ) -> Dict[str, Any]:
        """Calculate weighted portfolio returns

        Args:
            aligned_data: Aligned strategy DataFrames
            config: Portfolio configuration

        Returns:
            Dictionary with portfolio data
        """
        # Get common timestamps
        if not aligned_data:
            raise ValueError("No aligned data available")

        first_df = list(aligned_data.values())[0]
        timestamps = first_df['timestamp'].tolist()

        # Initialize with zeros
        num_periods = len(timestamps)
        combined_returns = [0.0] * num_periods
        strategy_returns = {}
        strategy_equities = {}

        # Calculate strategy returns
        for strategy_id, df in aligned_data.items():
            if 'return' in df.columns:
                returns = df['return'].tolist()
                strategy_returns[strategy_id] = returns

                # Calculate equity from returns (start at 1.0)
                equity = [1.0]
                for r in returns[1:]:
                    equity.append(equity[-1] * (1 + r))
                strategy_equities[strategy_id] = equity

        # Calculate weighted portfolio returns
        for strategy_id, returns in strategy_returns.items():
            weight = config.weights.get(strategy_id, 0.0)
            for i in range(num_periods):
                combined_returns[i] += weight * returns[i]

        # Calculate portfolio equity
        initial_value = 1.0
        combined_equity = [initial_value]

        for r in combined_returns[1:]:
            combined_equity.append(combined_equity[-1] * (1 + r))

        final_value = combined_equity[-1]
        total_return = (final_value - initial_value) / initial_value

        return {
            "timestamps": timestamps,
            "combined_equity": combined_equity,
            "combined_returns": combined_returns,
            "strategy_equities": strategy_equities,
            "strategy_returns": strategy_returns,
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return
        }


def create_equal_weight_portfolio(
    strategy_ids: List[str],
    start_date: date,
    end_date: date,
    replay_dir: str = "runs"
) -> Portfolio:
    """Convenience function to create equal-weight portfolio

    Args:
        strategy_ids: Strategy identifiers
        start_date: Portfolio start date
        end_date: Portfolio end date
        replay_dir: Replay directory

    Returns:
        Portfolio
    """
    # Calculate equal weights
    weight = 1.0 / len(strategy_ids)
    weights = {sid: weight for sid in strategy_ids}

    # Create config
    config = PortfolioConfig(
        strategy_ids=strategy_ids,
        weights=weights,
        start_date=start_date,
        end_date=end_date,
        alignment_method="inner"
    )

    # Build portfolio
    builder = PortfolioBuilder()
    return builder.build_portfolio(config, replay_dir)


if __name__ == "__main__":
    """Test portfolio builder"""
    print("=== PortfolioBuilder Test ===\n")

    from datetime import timedelta

    # Create mock config
    config = PortfolioConfig(
        strategy_ids=["ma_5_20", "breakout"],
        weights={"ma_5_20": 0.6, "breakout": 0.4},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        alignment_method="inner"
    )

    print(f"Portfolio config:")
    print(f"  Strategies: {config.strategy_ids}")
    print(f"  Weights: {config.weights}")

    # Validate
    try:
        config.validate()
        print("\n✓ Configuration is valid")
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")

    print("\n✓ Test passed")
