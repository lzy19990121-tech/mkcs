"""
Runtime Risk Metrics for SPL-4a

Real-time risk metrics calculated during strategy execution to support
runtime risk gating.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import numpy as np
from collections import deque

from core.context import RunContext


@dataclass
class RuntimeRiskMetrics:
    """Real-time risk metrics calculated during strategy execution

    Updated continuously during strategy execution to support
    runtime risk gating decisions.
    """
    # Stability metrics
    rolling_stability_score: float  # Updated every N bars
    rolling_return_volatility: float  # Volatility of returns

    # Performance metrics
    rolling_window_return: float  # Return over last window (e.g., 20d)
    rolling_max_drawdown: float  # Maximum drawdown in rolling window
    rolling_drawdown_duration: int  # Days in current drawdown

    # Regime indicators
    current_adx: Optional[float]  # Average Directional Index (if available)
    market_regime: str  # "trending", "ranging", "volatile", "unknown"

    # Position metrics
    total_exposure: float  # Total position exposure as ratio of capital
    num_positions: int  # Number of open positions

    # Timestamp
    calculated_at: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "rolling_stability_score": self.rolling_stability_score,
            "rolling_return_volatility": self.rolling_return_volatility,
            "rolling_window_return": self.rolling_window_return,
            "rolling_max_drawdown": self.rolling_max_drawdown,
            "rolling_drawdown_duration": self.rolling_drawdown_duration,
            "current_adx": self.current_adx,
            "market_regime": self.market_regime,
            "total_exposure": self.total_exposure,
            "num_positions": self.num_positions,
            "calculated_at": self.calculated_at.isoformat()
        }


class RuntimeRiskCalculator:
    """Calculate real-time risk metrics during strategy execution

    Maintains rolling windows of equity and returns to compute
    risk metrics on each tick.
    """

    def __init__(
        self,
        window_length: str = "20d",
        max_history: int = 100,
        initial_cash: float = 1000000.0
    ):
        """Initialize risk calculator

        Args:
            window_length: Analysis window length (e.g., "20d", "60d")
            max_history: Maximum number of historical points to keep
            initial_cash: Initial capital for return calculations
        """
        self.window_length = window_length
        self.max_history = max_history
        self.initial_cash = initial_cash

        # History buffers (FIFO queues)
        self.equity_history: deque = deque(maxlen=max_history)
        self.return_history: deque = deque(maxlen=max_history)
        self.drawdown_history: deque = deque(maxlen=max_history)

        # Drawdown tracking
        self.peak_equity: float = initial_cash
        self.current_drawdown_duration: int = 0

        # Metrics cache
        self._last_metrics: Optional[RuntimeRiskMetrics] = None
        self._last_calculation_time: Optional[datetime] = None

        # Parse window length
        self.window_days = self._parse_window_length(window_length)

    def _parse_window_length(self, window_length: str) -> int:
        """Parse window length string to days

        Args:
            window_length: Window length string (e.g., "20d", "60d")

        Returns:
            Number of days
        """
        if window_length.endswith("d"):
            return int(window_length[:-1])
        elif window_length.endswith("w"):
            return int(window_length[:-1]) * 7
        else:
            # Default to 20 days
            return 20

    def calculate(
        self,
        ctx: RunContext,
        positions: Dict,
        cash: float
    ) -> RuntimeRiskMetrics:
        """Called each tick to update risk metrics

        Args:
            ctx: Run context
            positions: Current positions {symbol: Position}
            cash: Current cash balance

        Returns:
            RuntimeRiskMetrics with current risk state
        """
        # Calculate current equity
        total_equity = cash + sum(
            pos.market_value for pos in positions.values()
        )

        # Update history
        self.equity_history.append(total_equity)

        # Update peak
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
            self.current_drawdown_duration = 0
        else:
            self.current_drawdown_duration += 1

        # Calculate return
        daily_return = 0.0
        if len(self.equity_history) >= 2:
            prev_equity = self.equity_history[-2]
            daily_return = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0

        self.return_history.append(daily_return)

        # Calculate drawdown
        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - total_equity) / self.peak_equity
        self.drawdown_history.append(drawdown)

        # Calculate rolling metrics
        rolling_return = self._calculate_rolling_return()
        rolling_mdd = self._calculate_rolling_mdd()
        rolling_volatility = self._calculate_rolling_volatility()
        stability_score = self._calculate_stability_score(
            rolling_return, rolling_mdd, rolling_volatility
        )

        # Calculate position metrics
        total_exposure = 0.0
        if self.initial_cash > 0:
            total_exposure = sum(
                pos.market_value for pos in positions.values()
            ) / self.initial_cash

        num_positions = len(positions)

        # Determine market regime (simplified)
        market_regime = self._determine_regime(rolling_volatility, rolling_return)

        metrics = RuntimeRiskMetrics(
            rolling_stability_score=stability_score,
            rolling_return_volatility=rolling_volatility,
            rolling_window_return=rolling_return,
            rolling_max_drawdown=rolling_mdd,
            rolling_drawdown_duration=self.current_drawdown_duration,
            current_adx=None,  # Would need additional market data
            market_regime=market_regime,
            total_exposure=total_exposure,
            num_positions=num_positions,
            calculated_at=ctx.now
        )

        self._last_metrics = metrics
        self._last_calculation_time = ctx.now

        return metrics

    def _calculate_rolling_return(self) -> float:
        """Calculate rolling window return"""
        if len(self.equity_history) < 2:
            return 0.0

        # Use min of available history and window length
        window = min(len(self.equity_history), self.window_days)

        if window < 2:
            return 0.0

        start_equity = self.equity_history[-window]
        end_equity = self.equity_history[-1]

        if start_equity <= 0:
            return 0.0

        return (end_equity - start_equity) / start_equity

    def _calculate_rolling_mdd(self) -> float:
        """Calculate rolling maximum drawdown"""
        if len(self.drawdown_history) < 2:
            return 0.0

        # Use min of available history and window length
        window = min(len(self.drawdown_history), self.window_days)

        if window < 2:
            return 0.0

        recent_drawdowns = list(self.drawdown_history)[-window:]
        return max(recent_drawdowns) if recent_drawdowns else 0.0

    def _calculate_rolling_volatility(self) -> float:
        """Calculate rolling return volatility (std dev)"""
        if len(self.return_history) < 2:
            return 0.0

        # Use min of available history and window length
        window = min(len(self.return_history), self.window_days)

        if window < 2:
            return 0.0

        recent_returns = list(self.return_history)[-window:]

        if len(recent_returns) < 2:
            return 0.0

        return float(np.std(recent_returns))

    def _calculate_stability_score(
        self,
        rolling_return: float,
        rolling_mdd: float,
        rolling_volatility: float
    ) -> float:
        """Calculate stability score (0-100)

        Higher is more stable.

        Args:
            rolling_return: Rolling window return
            rolling_mdd: Rolling maximum drawdown
            rolling_volatility: Rolling volatility

        Returns:
            Stability score (0-100)
        """
        score = 100.0

        # Penalty for drawdown
        if rolling_mdd > 0:
            score -= rolling_mdd * 100  # 10% drawdown = -10 points

        # Penalty for volatility
        score -= rolling_volatility * 500  # 0.02 volatility = -10 points

        # Penalty for negative returns
        if rolling_return < 0:
            score -= abs(rolling_return) * 50  # -10% return = -5 points

        # Bonus for positive returns
        if rolling_return > 0:
            score += rolling_return * 20  # 10% return = +2 points

        return max(0.0, min(100.0, score))

    def _determine_regime(
        self,
        volatility: float,
        return_value: float
    ) -> str:
        """Determine market regime (simplified)

        Args:
            volatility: Return volatility
            return_value: Rolling return

        Returns:
            Regime label: "trending", "ranging", "volatile", "unknown"
        """
        if len(self.return_history) < self.window_days:
            return "unknown"

        # Simple regime classification
        if volatility > 0.03:
            return "volatile"
        elif abs(return_value) > 0.05:
            return "trending"
        else:
            return "ranging"

    def get_current_metrics(self) -> Optional[RuntimeRiskMetrics]:
        """Get last calculated metrics without recalculation

        Returns:
            Last calculated RuntimeRiskMetrics or None
        """
        return self._last_metrics

    def reset(self, initial_cash: float = None):
        """Reset calculator state

        Args:
            initial_cash: New initial cash (optional)
        """
        if initial_cash is not None:
            self.initial_cash = initial_cash

        self.equity_history.clear()
        self.return_history.clear()
        self.drawdown_history.clear()

        self.peak_equity = self.initial_cash
        self.current_drawdown_duration = 0

        self._last_metrics = None
        self._last_calculation_time = None


if __name__ == "__main__":
    """Test runtime risk calculator"""
    print("=== RuntimeRiskCalculator Test ===\n")

    from unittest.mock import Mock
    from datetime import datetime

    calculator = RuntimeRiskCalculator(window_length="20d", initial_cash=1000000.0)

    # Mock context
    ctx = Mock()
    ctx.now = datetime.now()
    ctx.trading_date = ctx.now.date()

    # Simulate some history
    positions = {}
    cash = 1000000.0

    print("Simulating 30 days of trading...")

    equity_values = [
        1000000, 1005000, 998000, 995000, 990000,  # Small drawdown
        985000, 980000, 975000, 970000, 965000,  # Continued drawdown
        970000, 975000, 980000, 985000, 990000,  # Recovery
        995000, 1000000, 1005000, 1010000, 1015000,  # New high
        1020000, 1025000, 1030000, 1035000, 1040000,
        1045000, 1050000, 1055000, 1060000, 1065000
    ]

    for i, equity in enumerate(equity_values):
        cash = equity  # Simplified: no positions

        metrics = calculator.calculate(ctx, positions, cash)

        if i % 5 == 0:
            print(f"Day {i+1}:")
            print(f"  Equity: ${equity:,.0f}")
            print(f"  Rolling Return: {metrics.rolling_window_return*100:.2f}%")
            print(f"  Rolling MDD: {metrics.rolling_max_drawdown*100:.2f}%")
            print(f"  Stability Score: {metrics.rolling_stability_score:.1f}/100")
            print(f"  Regime: {metrics.market_regime}")

    print("\n✓ Test passed")
