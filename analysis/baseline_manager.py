"""
Baseline Manager for SPL-4c

Manages the lifecycle of risk baselines: freezing, loading, and comparison.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import shutil

from analysis.risk_baseline import RiskBaseline, BaselineSnapshot, generate_baseline_id
from analysis.replay_schema import ReplayOutput, load_replay_outputs
from analysis.window_scanner import WindowScanner, WindowMetrics
from analysis.stability_analysis import StabilityAnalyzer, StabilityReport
from analysis.risk_envelope import RiskEnvelopeBuilder, RiskEnvelope
from analysis.structural_analysis import StructuralAnalyzer, StructuralAnalysisResult
from analysis.actionable_rules import RiskRuleset, RiskRuleGenerator
from utils.hash import get_git_commit


class BaselineManager:
    """Manage risk baseline lifecycle

    Responsibilities:
    1. Freeze current SPL-3b results as baselines
    2. Load baselines from storage
    3. Compare current results vs baselines
    4. Manage baseline versioning
    """

    def __init__(self, baseline_dir: str = "baselines/risk"):
        """Initialize baseline manager

        Args:
            baseline_dir: Directory to store baselines
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def freeze_baselines(
        self,
        replay_dir: str,
        output_dir: Optional[str] = None,
        window_lengths: List[str] = None,
        analysis_version: str = "deep_analysis_v3b"
    ) -> BaselineSnapshot:
        """Freeze current SPL-3b results as baselines

        Args:
            replay_dir: Directory containing replay outputs
            output_dir: Output directory for baselines (default: self.baseline_dir)
            window_lengths: Window lengths to analyze (default: ["20d", "60d"])
            analysis_version: Analysis version identifier

        Returns:
            BaselineSnapshot containing all frozen baselines
        """
        if window_lengths is None:
            window_lengths = ["20d", "60d"]

        output_dir = output_dir or str(self.baseline_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load all replays
        replays = load_replay_outputs(replay_dir)
        if not replays:
            print(f"Warning: No replays found in {replay_dir}")
            return BaselineSnapshot(
                version="1.0",
                created_at=datetime.now(),
                commit_hash=get_git_commit(short=True)
            )

        print(f"Freezing {len(replays)} replay outputs as baselines...")

        # Initialize analyzers
        window_scanner = WindowScanner()
        stability_analyzer = StabilityAnalyzer()
        envelope_builder = RiskEnvelopeBuilder()
        structural_analyzer = StructuralAnalyzer()
        rule_generator = RiskRuleGenerator()

        # Create snapshot
        commit_hash = get_git_commit(short=True)
        snapshot = BaselineSnapshot(
            version="1.0",
            created_at=datetime.now(),
            commit_hash=commit_hash
        )

        # Process each replay
        for replay in replays:
            print(f"\nProcessing: {replay.run_id}")

            # Generate baseline ID
            baseline_id = generate_baseline_id(replay.strategy_id, replay.commit_hash)

            # Run SPL-3b analysis
            worst_windows = {}
            risk_patterns = {}
            pattern_similarity = {}
            envelopes = {}
            rule_thresholds = {}

            # Stability analysis (once for entire replay)
            stability_report = stability_analyzer.analyze_replay(replay)
            stability_metrics = {
                "stability_score": stability_report.stability_score,
                "return_variance": stability_report.return_variance,
                "mdd_consistency": stability_report.mdd_consistency,
                "worst_cvar": stability_report.worst_cvar
            }

            # Store envelope and structural results for rule generation
            primary_envelope = None
            primary_structural = None
            primary_window = window_lengths[0] if window_lengths else "20d"

            for window_length in window_lengths:
                print(f"  Analyzing {window_length} window...")

                # Find worst windows
                all_windows = window_scanner.scan_replay(replay, window_length)
                windows = all_windows[:5] if len(all_windows) >= 5 else all_windows
                worst_windows[window_length] = [w.window_id for w in windows]

                # Risk envelope
                envelope = envelope_builder.build_envelope(replay, window_length)
                envelopes[window_length] = {
                    "return_p95": envelope.return_p95,
                    "return_p99": envelope.return_p99,
                    "mdd_p95": envelope.mdd_p95,
                    "mdd_p99": envelope.mdd_p99,
                    "duration_p95": envelope.duration_p95,
                    "duration_p99": envelope.duration_p99
                }

                # Structural analysis
                structural_result = structural_analyzer.analyze_structure(replay, window_length)
                risk_patterns[window_length] = structural_result.risk_pattern_type.value
                pattern_similarity[window_length] = structural_result.pattern_metrics.pattern_similarity

                # Store primary window results for rule generation
                if window_length == primary_window:
                    primary_envelope = envelope
                    primary_structural = structural_result

            # Generate rules from primary window
            if primary_envelope and primary_structural:
                ruleset = rule_generator.generate_rules(
                    replay, stability_report, primary_envelope, primary_structural
                )
                for rule in ruleset.rules:
                    rule_thresholds[rule.trigger_metric] = rule.trigger_threshold

            # Create baseline
            baseline = RiskBaseline(
                baseline_id=baseline_id,
                strategy_id=replay.strategy_id,
                run_id=replay.run_id,
                commit_hash=replay.commit_hash,
                config_hash=replay.config_hash,
                analysis_version=analysis_version,
                created_at=datetime.now(),
                worst_windows=worst_windows,
                risk_patterns=risk_patterns,
                pattern_similarity=pattern_similarity,
                envelopes=envelopes,
                rule_thresholds=rule_thresholds,
                stability_metrics=stability_metrics
            )

            snapshot.add_baseline(baseline)
            print(f"  ✓ Frozen baseline: {baseline_id}")

        # Save snapshot
        self._save_snapshot(snapshot, output_path)

        return snapshot

    def load_baseline(self, baseline_id: str) -> Optional[RiskBaseline]:
        """Load baseline by ID

        Args:
            baseline_id: Baseline identifier

        Returns:
            RiskBaseline or None if not found
        """
        snapshot = self.load_all_baselines()
        return snapshot.get_baseline(baseline_id)

    def load_all_baselines(self) -> BaselineSnapshot:
        """Load all baselines from directory

        Returns:
            BaselineSnapshot containing all baselines
        """
        baselines_file = self.baseline_dir / "baselines_v1.json"

        if not baselines_file.exists():
            print(f"Warning: No baselines found at {baselines_file}")
            return BaselineSnapshot(
                version="1.0",
                created_at=datetime.now(),
                commit_hash=get_git_commit(short=True)
            )

        with open(baselines_file) as f:
            data = json.load(f)

        return BaselineSnapshot.from_dict(data)

    def compare_vs_baseline(
        self,
        current: ReplayOutput,
        baseline: RiskBaseline,
        tolerance_pct: float = 0.02
    ) -> Dict[str, Any]:
        """Compare current results vs baseline

        Args:
            current: Current replay output
            baseline: Frozen baseline
            tolerance_pct: Allowed tolerance percentage (default 2%)

        Returns:
            Dictionary with test results (PASS/FAIL)
        """
        from tests.risk_regression.risk_baseline_test import RiskBaselineTests

        tests = RiskBaselineTests(tolerance_pct=tolerance_pct)
        results = {}

        # Run all tests
        results["worst_window_drift"] = tests.test_worst_window_non_drift(baseline, current)
        results["structural_similarity"] = tests.test_structural_similarity(baseline, current)
        results["envelope_regression"] = tests.test_envelope_non_regression(baseline, current)
        results["rule_trigger_sanity"] = tests.test_rule_trigger_sanity(baseline, current)
        results["replay_determinism"] = tests.test_replay_determinism(baseline)

        # Overall status
        all_passed = all(r.get("status") == "PASS" for r in results.values())
        results["_overall"] = {
            "status": "PASS" if all_passed else "FAIL",
            "tests_passed": sum(1 for r in results.values() if r.get("status") == "PASS"),
            "tests_total": len(results) - 1  # Exclude _overall
        }

        return results

    def _save_snapshot(self, snapshot: BaselineSnapshot, output_dir: Path):
        """Save snapshot to files

        Args:
            snapshot: BaselineSnapshot to save
            output_dir: Output directory
        """
        # Save baselines
        baselines_file = output_dir / "baselines_v1.json"
        with open(baselines_file, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        # Save manifest
        manifest = {
            "version": snapshot.version,
            "created_at": snapshot.created_at.isoformat(),
            "commit_hash": snapshot.commit_hash,
            "baselines": [
                {
                    "baseline_id": b.baseline_id,
                    "strategy_id": b.strategy_id,
                    "run_id": b.run_id,
                    "file": "baselines_v1.json"
                }
                for b in snapshot.baselines
            ]
        }

        manifest_file = output_dir / "baseline_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n✓ Baselines saved to {output_dir}")
        print(f"  Manifest: {manifest_file}")
        print(f"  Data: {baselines_file}")


def freeze_current_results(
    replay_dir: str = "runs",
    output_dir: str = "baselines/risk",
    window_lengths: List[str] = None
) -> BaselineSnapshot:
    """Convenience function to freeze current results

    Args:
        replay_dir: Directory containing replay outputs
        output_dir: Output directory for baselines
        window_lengths: Window lengths to analyze

    Returns:
        BaselineSnapshot
    """
    manager = BaselineManager()
    return manager.freeze_baselines(replay_dir, output_dir, window_lengths)


if __name__ == "__main__":
    """Test baseline manager"""
    print("=== BaselineManager Test ===\n")

    manager = BaselineManager()

    # Test loading (if baselines exist)
    snapshot = manager.load_all_baselines()
    print(f"Loaded {len(snapshot.baselines)} baselines")
    print(f"Version: {snapshot.version}")
    print(f"Created: {snapshot.created_at}")
    print(f"Commit: {snapshot.commit_hash}")

    if snapshot.baselines:
        baseline = snapshot.baselines[0]
        print(f"\nExample baseline:")
        print(f"  ID: {baseline.baseline_id}")
        print(f"  Strategy: {baseline.strategy_id}")
        print(f"  Run: {baseline.run_id}")
        print(f"  Windows: {list(baseline.worst_windows.keys())}")

    print("\n✓ Test passed")
