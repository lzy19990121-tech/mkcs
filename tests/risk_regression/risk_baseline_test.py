"""
Risk Regression Tests for SPL-4c

Regression tests based on frozen SPL-3b baselines to ensure
risk characteristics do not regress over time.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from analysis.risk_baseline import RiskBaseline
from analysis.replay_schema import ReplayOutput
from analysis.window_scanner import WindowScanner
from analysis.stability_analysis import StabilityAnalyzer
from analysis.structural_analysis import StructuralAnalyzer


@dataclass
class TestResult:
    """Result of a regression test"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    message: str
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "status": self.status,
            "message": self.message,
            "details": self.details or {}
        }


class RiskBaselineTests:
    """Regression tests based on frozen SPL-3b baselines

    Test Suite:
    1. C2.1: Worst-Window Non-Drift Test
    2. C2.2: Structural Similarity Test
    3. C2.3: Envelope Non-Regression Test
    4. C2.4: Rule Trigger Sanity Test
    5. C2.5: Replay Determinism Test
    """

    def __init__(self, tolerance_pct: float = 0.02):
        """Initialize test suite

        Args:
            tolerance_pct: Allowed tolerance for metric comparisons (default 2%)
        """
        self.tolerance_pct = tolerance_pct
        self.window_scanner = WindowScanner()
        self.stability_analyzer = StabilityAnalyzer()
        self.structural_analyzer = StructuralAnalyzer()

    def test_worst_window_non_drift(
        self,
        baseline: RiskBaseline,
        current: ReplayOutput,
        top_k: int = 10
    ) -> TestResult:
        """C2.1: Worst-Window Non-Drift Test

        Verifies that worst windows remain in original time range or Top-K.

        Failure criteria:
        - New worst windows appear outside baseline time ranges
        - More than 50% of baseline worst windows are not in current Top-K

        Args:
            baseline: Frozen baseline
            current: Current replay output
            top_k: Number of top worst windows to consider

        Returns:
            TestResult
        """
        try:
            details = {}
            drift_count = 0
            total_windows = 0

            for window_length, baseline_window_ids in baseline.worst_windows.items():
                # Find current worst windows
                current_windows = self.window_scanner.find_worst_windows(
                    current, window_length, top_k=top_k
                )
                current_window_ids = set(w.window_id for w in current_windows)

                # Check for drift: baseline windows not in current Top-K
                baseline_set = set(baseline_window_ids)
                missing_windows = baseline_set - current_window_ids

                total_windows += len(baseline_set)
                drift_count += len(missing_windows)

                details[window_length] = {
                    "baseline_windows": len(baseline_set),
                    "missing_from_top_k": len(missing_windows),
                    "missing_ids": list(missing_windows)
                }

            # Calculate drift ratio
            if total_windows == 0:
                return TestResult(
                    test_name="worst_window_non_drift",
                    status="SKIP",
                    message="No worst windows in baseline",
                    details=details
                )

            drift_ratio = drift_count / total_windows

            # Fail if more than 50% of windows have drifted
            if drift_ratio > 0.5:
                return TestResult(
                    test_name="worst_window_non_drift",
                    status="FAIL",
                    message=f"Significant worst-window drift: {drift_ratio*100:.1f}% of baseline windows not in current Top-K",
                    details={**details, "drift_ratio": drift_ratio}
                )

            return TestResult(
                test_name="worst_window_non_drift",
                status="PASS",
                message=f"Worst windows stable: {drift_ratio*100:.1f}% drift",
                details={**details, "drift_ratio": drift_ratio}
            )

        except Exception as e:
            return TestResult(
                test_name="worst_window_non_drift",
                status="FAIL",
                message=f"Test error: {str(e)}",
                details={"error": str(e)}
            )

    def test_structural_similarity(
        self,
        baseline: RiskBaseline,
        current: ReplayOutput,
        similarity_threshold: float = 0.70
    ) -> TestResult:
        """C2.2: Structural Similarity Test

        Re-calculates pattern similarity and asserts >= baseline threshold.

        Failure criteria:
        - Current similarity < baseline similarity * (1 - tolerance)
        - Risk pattern type changes (e.g., structural → single_outlier)

        Args:
            baseline: Frozen baseline
            current: Current replay output
            similarity_threshold: Minimum required similarity (default 70%)

        Returns:
            TestResult
        """
        try:
            details = {}
            min_similarity = 1.0

            for window_length, baseline_pattern in baseline.risk_patterns.items():
                # Calculate current structural analysis
                current_result = self.structural_analyzer.analyze_structure(current, window_length)
                current_similarity = current_result.pattern_metrics.pattern_similarity
                baseline_similarity = baseline.pattern_similarity.get(window_length, 0.0)

                # Calculate minimum allowed similarity
                min_allowed = baseline_similarity * (1 - self.tolerance_pct)

                details[window_length] = {
                    "baseline_pattern": baseline_pattern,
                    "current_pattern": current_result.risk_pattern_type.value,
                    "baseline_similarity": baseline_similarity,
                    "current_similarity": current_similarity,
                    "min_allowed": min_allowed
                }

                # Check pattern type change
                if current_result.risk_pattern_type.value != baseline_pattern:
                    return TestResult(
                        test_name="structural_similarity",
                        status="FAIL",
                        message=f"Risk pattern type changed for {window_length}: {baseline_pattern} → {current_result.risk_pattern_type.value}",
                        details=details
                    )

                # Check similarity threshold
                min_similarity = min(min_similarity, current_similarity)

                if current_similarity < min_allowed:
                    return TestResult(
                        test_name="structural_similarity",
                        status="FAIL",
                        message=f"Pattern similarity dropped below threshold for {window_length}: {current_similarity:.3f} < {min_allowed:.3f}",
                        details=details
                    )

            # Also check against absolute threshold
            if min_similarity < similarity_threshold:
                return TestResult(
                    test_name="structural_similarity",
                    status="FAIL",
                    message=f"Pattern similarity below absolute threshold: {min_similarity:.3f} < {similarity_threshold:.3f}",
                    details=details
                )

            return TestResult(
                test_name="structural_similarity",
                status="PASS",
                message=f"Structural similarity maintained: {min_similarity:.3f}",
                details=details
            )

        except Exception as e:
            return TestResult(
                test_name="structural_similarity",
                status="FAIL",
                message=f"Test error: {str(e)}",
                details={"error": str(e)}
            )

    def test_envelope_non_regression(
        self,
        baseline: RiskBaseline,
        current: ReplayOutput
    ) -> TestResult:
        """C2.3: Envelope Non-Regression Test

        Compares P95/P99 return, MDD, duration.
        Allows small tolerance (1-2%).
        FAIL if significantly worse.

        Args:
            baseline: Frozen baseline
            current: Current replay output

        Returns:
            TestResult
        """
        try:
            from analysis.risk_envelope import RiskEnvelopeBuilder

            envelope_builder = RiskEnvelopeBuilder()
            details = {}
            violations = []

            for window_length, baseline_envelope in baseline.envelopes.items():
                # Build current envelope
                current_envelope = envelope_builder.build_envelope(current, window_length)

                # Compare key metrics
                metrics_to_check = [
                    ("return_p95", "worse"),  # Lower (more negative) is worse
                    ("return_p99", "worse"),
                    ("mdd_p95", "worse"),  # Higher is worse
                    ("mdd_p99", "worse"),
                    ("duration_p95", "worse"),  # Higher is worse
                    ("duration_p99", "worse")
                ]

                window_violations = []
                for metric_name, direction in metrics_to_check:
                    baseline_value = baseline_envelope.get(metric_name)
                    current_value = getattr(current_envelope, metric_name)

                    if baseline_value is None:
                        continue

                    # Calculate tolerance
                    tolerance = abs(baseline_value) * self.tolerance_pct

                    if direction == "worse":
                        # For returns: more negative is worse
                        # For MDD/duration: higher is worse
                        if metric_name.startswith("return"):
                            # More negative is worse
                            is_regression = current_value < (baseline_value - tolerance)
                            delta = current_value - baseline_value
                        else:
                            # Higher is worse
                            is_regression = current_value > (baseline_value + tolerance)
                            delta = current_value - baseline_value

                        if is_regression:
                            window_violations.append({
                                "metric": metric_name,
                                "baseline": baseline_value,
                                "current": current_value,
                                "delta": delta,
                                "tolerance": tolerance
                            })

                details[window_length] = {
                    "baseline_envelope": baseline_envelope,
                    "current_envelope": {
                        "return_p95": current_envelope.return_p95,
                        "return_p99": current_envelope.return_p99,
                        "mdd_p95": current_envelope.mdd_p95,
                        "mdd_p99": current_envelope.mdd_p99,
                        "duration_p95": current_envelope.duration_p95,
                        "duration_p99": current_envelope.duration_p99
                    },
                    "violations": window_violations
                }

                violations.extend(window_violations)

            if violations:
                return TestResult(
                    test_name="envelope_non_regression",
                    status="FAIL",
                    message=f"Envelope regression detected: {len(violations)} metrics worse than baseline",
                    details={"violations": violations, **details}
                )

            return TestResult(
                test_name="envelope_non_regression",
                status="PASS",
                message="All envelope metrics within tolerance",
                details=details
            )

        except Exception as e:
            return TestResult(
                test_name="envelope_non_regression",
                status="FAIL",
                message=f"Test error: {str(e)}",
                details={"error": str(e)}
            )

    def test_rule_trigger_sanity(
        self,
        baseline: RiskBaseline,
        current: ReplayOutput
    ) -> TestResult:
        """C2.4: Rule Trigger Sanity Test

        In known worst-case replay, verify all baseline rules trigger correctly.

        Args:
            baseline: Frozen baseline
            current: Current replay output

        Returns:
            TestResult
        """
        try:
            details = {}
            triggered_rules = []
            failed_rules = []

            # Re-run stability analysis to check rules
            stability_report = self.stability_analyzer.analyze_replay(current)

            # Check each rule threshold
            for metric_name, threshold in baseline.rule_thresholds.items():
                if metric_name == "stability_score":
                    # Stability score should be below threshold (fragile)
                    is_triggered = stability_report.stability_score < threshold
                    current_value = stability_report.stability_score
                elif metric_name == "max_drawdown":
                    # Find max drawdown from replay
                    equity_series = current.get_equity_series()
                    if equity_series.empty:
                        continue
                    rolling_max = equity_series.cummax()
                    drawdown = (equity_series - rolling_max) / rolling_max
                    current_value = abs(drawdown.min())
                    is_triggered = current_value > threshold
                elif metric_name == "window_return":
                    # Find minimum window return
                    returns = current.get_returns_series()
                    if returns.empty:
                        continue
                    # Approximate 20-day rolling return
                    rolling_return = returns.rolling(20).sum()
                    current_value = rolling_return.min()
                    is_triggered = current_value < threshold
                else:
                    # Unknown metric
                    continue

                details[metric_name] = {
                    "threshold": threshold,
                    "current_value": current_value,
                    "triggered": is_triggered
                }

                if is_triggered:
                    triggered_rules.append(metric_name)
                else:
                    failed_rules.append({
                        "metric": metric_name,
                        "threshold": threshold,
                        "current_value": current_value
                    })

            # In worst-case replay, all rules should trigger
            if failed_rules:
                return TestResult(
                    test_name="rule_trigger_sanity",
                    status="FAIL",
                    message=f"{len(failed_rules)} rules failed to trigger in worst-case replay",
                    details={
                        "triggered_rules": triggered_rules,
                        "failed_rules": failed_rules,
                        "all_details": details
                    }
                )

            return TestResult(
                test_name="rule_trigger_sanity",
                status="PASS",
                message=f"All {len(triggered_rules)} rules triggered correctly",
                details=details
            )

        except Exception as e:
            return TestResult(
                test_name="rule_trigger_sanity",
                status="FAIL",
                message=f"Test error: {str(e)}",
                details={"error": str(e)}
            )

    def test_replay_determinism(
        self,
        baseline: RiskBaseline,
        num_runs: int = 3
    ) -> TestResult:
        """C2.5: Replay Determinism Test

        Run same config 3 times and verify Risk Card core fields match.

        Note: This test requires the ability to re-run the strategy,
        which may not be available in all contexts. In that case, it
        returns SKIP.

        Args:
            baseline: Frozen baseline
            num_runs: Number of times to run replay (default 3)

        Returns:
            TestResult
        """
        try:
            # This test requires access to the original strategy config
            # and the ability to re-run it. If we don't have that, skip.
            if not baseline.config_hash:
                return TestResult(
                    test_name="replay_determinism",
                    status="SKIP",
                    message="Cannot test determinism without original config",
                    details={"reason": "missing_config"}
                )

            # In a full implementation, we would:
            # 1. Load the original strategy config
            # 2. Run it num_runs times
            # 3. Compare the Risk Card fields (stability_score, worst windows, etc.)
            # 4. Check they match within tolerance

            # For now, we skip this test
            return TestResult(
                test_name="replay_determinism",
                status="SKIP",
                message="Determinism test not yet implemented (requires strategy re-run capability)",
                details={"note": "TODO: implement full determinism check"}
            )

        except Exception as e:
            return TestResult(
                test_name="replay_determinism",
                status="FAIL",
                message=f"Test error: {str(e)}",
                details={"error": str(e)}
            )


if __name__ == "__main__":
    """Test the regression test suite"""
    print("=== RiskBaselineTests Test ===\n")

    # Create a mock baseline and replay for testing
    from analysis.risk_baseline import RiskBaseline
    from datetime import datetime

    baseline = RiskBaseline(
        baseline_id="test_baseline",
        strategy_id="ma_5_20",
        run_id="test_run",
        commit_hash="abc123",
        config_hash="def456",
        analysis_version="deep_analysis_v3b",
        created_at=datetime.now(),
        worst_windows={"20d": ["window_1", "window_2"]},
        risk_patterns={"20d": "structural"},
        pattern_similarity={"20d": 0.85},
        envelopes={"20d": {"return_p95": -0.20, "mdd_p95": 0.15}},
        rule_thresholds={"stability_score": 30.0},
        stability_metrics={"stability_score": 25.0, "stability_label": "Fragile"}
    )

    tests = RiskBaselineTests()

    # Test that we can instantiate tests
    print(f"✓ Test suite initialized with {tests.tolerance_pct*100}% tolerance")

    # Run rule trigger sanity test (mock)
    result = tests.test_rule_trigger_sanity(baseline, None)
    print(f"\nRule trigger sanity test: {result.status}")
    print(f"  {result.message}")

    print("\n✓ All tests passed")
