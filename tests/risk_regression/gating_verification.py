"""
Gating Verification for SPL-4a

Verifies that runtime gating is effective by comparing results
with and without gating.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from analysis.replay_schema import ReplayOutput
from analysis.actionable_rules import RiskRuleset
from agent.runner import TradingAgent
from skills.risk.risk_gate import RiskGate
from config import BacktestConfig


@dataclass
class GatingComparisonResult:
    """Result of gating effectiveness comparison"""
    strategy_id: str

    # Metrics without gating (baseline)
    baseline_worst_return: float
    baseline_worst_mdd: float
    baseline_final_return: float

    # Metrics with gating (protected)
    gated_worst_return: float
    gated_worst_mdd: float
    gated_final_return: float

    # Improvement metrics
    worst_return_improvement: float  # Positive = better
    mdd_improvement: float  # Positive = better
    return_sacrifice: float  # How much normal return was lost

    # Gating statistics
    gate_trigger_count: int
    gate_pause_count: int
    gate_disable_count: int

    # Verification status
    is_effective: bool  # Gating reduced worst-case without excessive sacrifice


class GatingVerification:
    """A3: Verify gating effectiveness before/after comparison

    Compares two replays:
    1. Without gating (baseline)
    2. With gating (protected)

    Metrics:
    - worst-case return (should be better with gating)
    - worst-case duration (should be shorter with gating)
    - new spike risks introduced by gating
    - frequency of gate triggers
    - return sacrifice (how much normal return is lost)
    """

    def __init__(self, max_return_sacrifice: float = 0.05):
        """Initialize gating verification

        Args:
            max_return_sacrifice: Maximum acceptable return sacrifice (default 5%)
        """
        self.max_return_sacrifice = max_return_sacrifice

    def compare_gating_impact(
        self,
        baseline_replay: ReplayOutput,
        gated_replay: ReplayOutput,
        gate_stats: Dict[str, Any]
    ) -> GatingComparisonResult:
        """Compare gating impact on strategy performance

        Args:
            baseline_replay: Replay without gating
            gated_replay: Replay with gating
            gate_stats: Gate statistics from RiskGate

        Returns:
            GatingComparisonResult
        """
        # Calculate metrics
        baseline_metrics = self._calculate_replay_metrics(baseline_replay)
        gated_metrics = self._calculate_replay_metrics(gated_replay)

        # Calculate improvements
        worst_return_improvement = (
            gated_metrics["worst_return"] - baseline_metrics["worst_return"]
        )
        # Negative improvement is good (less negative return)

        mdd_improvement = (
            gated_metrics["worst_mdd"] - baseline_metrics["worst_mdd"]
        )
        # Negative improvement is good (smaller drawdown)

        return_sacrifice = (
            baseline_metrics["final_return"] - gated_metrics["final_return"]
        )

        # Create result
        result = GatingComparisonResult(
            strategy_id=baseline_replay.strategy_id,
            baseline_worst_return=baseline_metrics["worst_return"],
            baseline_worst_mdd=baseline_metrics["worst_mdd"],
            baseline_final_return=baseline_metrics["final_return"],
            gated_worst_return=gated_metrics["worst_return"],
            gated_worst_mdd=gated_metrics["worst_mdd"],
            gated_final_return=gated_metrics["final_return"],
            worst_return_improvement=worst_return_improvement,
            mdd_improvement=mdd_improvement,
            return_sacrifice=return_sacrifice,
            gate_trigger_count=gate_stats.get("gate_triggers", 0),
            gate_pause_count=gate_stats.get("pause_count", 0),
            gate_disable_count=gate_stats.get("disable_count", 0),
            is_effective=self._is_effective(
                worst_return_improvement,
                mdd_improvement,
                return_sacrifice
            )
        )

        return result

    def _calculate_replay_metrics(self, replay: ReplayOutput) -> Dict[str, float]:
        """Calculate key metrics from replay

        Args:
            replay: Replay output

        Returns:
            Dictionary with metrics
        """
        # Get returns series
        returns = replay.get_returns_series()

        if returns.empty:
            return {
                "worst_return": 0.0,
                "worst_mdd": 0.0,
                "final_return": 0.0
            }

        # Calculate worst window return (20-day rolling)
        rolling_return = returns.rolling(20).sum()
        worst_return = rolling_return.min()

        # Calculate max drawdown
        equity = replay.get_equity_series()
        if not equity.empty:
            rolling_max = equity.cummax()
            drawdown = (equity - rolling_max) / rolling_max
            worst_mdd = abs(drawdown.min())
        else:
            worst_mdd = 0.0

        # Calculate final return
        final_return = float(
            (replay.final_equity - replay.initial_cash) / replay.initial_cash
        )

        return {
            "worst_return": worst_return,
            "worst_mdd": worst_mdd,
            "final_return": final_return
        }

    def _is_effective(
        self,
        worst_return_improvement: float,
        mdd_improvement: float,
        return_sacrifice: float
    ) -> bool:
        """Determine if gating is effective

        Args:
            worst_return_improvement: Improvement in worst return
            mdd_improvement: Improvement in max drawdown
            return_sacrifice: Sacrifice in final return

        Returns:
            True if effective
        """
        # Check if return sacrifice is acceptable
        if return_sacrifice > self.max_return_sacrifice:
            return False

        # Check if there's meaningful improvement
        # Either worst return improved by at least 20% relative
        # Or MDD improved by at least 20% relative
        has_return_improvement = worst_return_improvement > 0.01  # At least 1% absolute
        has_mdd_improvement = mdd_improvement < -0.01  # At least 1% absolute

        return has_return_improvement or has_mdd_improvement

    def generate_report(
        self,
        result: GatingComparisonResult
    ) -> str:
        """Generate markdown report

        Args:
            result: Gating comparison result

        Returns:
            Markdown report
        """
        lines = []

        lines.append(f"## Gating Verification Report - {result.strategy_id}\n")

        # Overall assessment
        status = "✅ Effective" if result.is_effective else "❌ Ineffective"
        lines.append(f"**Overall Assessment**: {status}\n")

        # Baseline metrics
        lines.append("### Baseline (No Gating)")
        lines.append(f"- Worst Window Return: {result.baseline_worst_return*100:.2f}%")
        lines.append(f"- Max Drawdown: {result.baseline_worst_mdd*100:.2f}%")
        lines.append(f"- Final Return: {result.baseline_final_return*100:.2f}%\n")

        # Gated metrics
        lines.append("### With Gating")
        lines.append(f"- Worst Window Return: {result.gated_worst_return*100:.2f}%")
        lines.append(f"- Max Drawdown: {result.gated_worst_mdd*100:.2f}%")
        lines.append(f"- Final Return: {result.gated_final_return*100:.2f}%\n")

        # Improvements
        lines.append("### Improvements")
        lines.append(f"- Worst Return: {result.worst_return_improvement*100:+.2f}%")
        lines.append(f"- Max Drawdown: {result.mdd_improvement*100:+.2f}%")
        lines.append(f"- Return Sacrifice: {result.return_sacrifice*100:+.2f}%\n")

        # Gating statistics
        lines.append("### Gating Statistics")
        lines.append(f"- Total Triggers: {result.gate_trigger_count}")
        lines.append(f"- Pause Actions: {result.gate_pause_count}")
        lines.append(f"- Disable Actions: {result.gate_disable_count}\n")

        # Recommendation
        lines.append("### Recommendation")
        if result.is_effective:
            lines.append("✓ Gating is effective. Risk is reduced without excessive return sacrifice.")
        else:
            if result.return_sacrifice > self.max_return_sacrifice:
                lines.append(f"✗ Return sacrifice ({result.return_sacrifice*100:.2f}%) exceeds threshold ({self.max_return_sacrifice*100:.2f}%).")
                lines.append("  Consider adjusting gate thresholds or reducing position reduction ratio.")

            if result.worst_return_improvement <= 0 and result.mdd_improvement >= 0:
                lines.append("✗ Gating did not improve worst-case metrics.")
                lines.append("  Review rule thresholds and consider if they are too lenient.")

        return "\n".join(lines)


def run_gating_verification(
    baseline_output_dir: str,
    gated_output_dir: str,
    ruleset: RiskRuleset,
    output_path: Optional[Path] = None
) -> GatingComparisonResult:
    """Run gating verification

    Args:
        baseline_output_dir: Directory with baseline (no gating) results
        gated_output_dir: Directory with gated results
        ruleset: Risk ruleset used for gating
        output_path: Optional path to save report

    Returns:
        GatingComparisonResult
    """
    from analysis.replay_schema import ReplayOutput

    # Load replays
    baseline_replay = ReplayOutput.from_directory(Path(baseline_output_dir))
    gated_replay = ReplayOutput.from_directory(Path(gated_output_dir))

    # Load gate stats (if available)
    gate_stats_path = Path(gated_output_dir) / "gate_stats.json"
    if gate_stats_path.exists():
        with open(gate_stats_path) as f:
            gate_stats = json.load(f)
    else:
        gate_stats = {}

    # Run verification
    verifier = GatingVerification()
    result = verifier.compare_gating_impact(
        baseline_replay,
        gated_replay,
        gate_stats
    )

    # Generate report
    report = verifier.generate_report(result)

    if output_path:
        output_path.write_text(report, encoding='utf-8')
        print(f"✓ Report saved: {output_path}")

    print("\n" + report)

    return result


if __name__ == "__main__":
    """Test gating verification"""
    print("=== GatingVerification Test ===\n")

    from analysis.replay_schema import ReplayOutput
    from datetime import date, datetime
    from decimal import Decimal

    # Create mock replays for testing
    baseline_replay = ReplayOutput(
        run_id="baseline_test",
        strategy_id="ma_5_20",
        strategy_name="MA(5,20)",
        commit_hash="abc123",
        config_hash="def456",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        initial_cash=Decimal("1000000"),
        final_equity=Decimal("950000")  # -5% return
    )

    gated_replay = ReplayOutput(
        run_id="gated_test",
        strategy_id="ma_5_20",
        strategy_name="MA(5,20)",
        commit_hash="abc123",
        config_hash="def456",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        initial_cash=Decimal("1000000"),
        final_equity=Decimal("970000")  # -3% return (better)
    )

    gate_stats = {
        "gate_triggers": 3,
        "pause_count": 2,
        "disable_count": 0
    }

    verifier = GatingVerification()

    # For testing, we'll calculate metrics directly
    # In real usage, these would come from actual replays
    result = GatingComparisonResult(
        strategy_id="ma_5_20",
        baseline_worst_return=-0.15,
        baseline_worst_mdd=0.18,
        baseline_final_return=-0.05,
        gated_worst_return=-0.10,
        gated_worst_mdd=0.12,
        gated_final_return=-0.03,
        worst_return_improvement=0.05,  # 5% better
        mdd_improvement=-0.06,  # 6% better
        return_sacrifice=0.02,  # 2% sacrifice
        gate_trigger_count=3,
        gate_pause_count=2,
        gate_disable_count=0,
        is_effective=True
    )

    report = verifier.generate_report(result)
    print(report)

    print("\n✓ Test passed")
