"""
Risk Baseline System for SPL-4c

Freezes SPL-3b worst-case findings as immutable regression test baselines.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import hashlib


@dataclass
class RiskBaseline:
    """Frozen risk baseline for regression testing

    Contains all SPL-3b analysis results for a specific strategy run,
    frozen at a point in time to serve as regression test reference.
    """
    # Identification
    baseline_id: str  # Generated from strategy_id + commit_hash
    strategy_id: str
    run_id: str

    # Traceability
    commit_hash: str
    config_hash: str
    analysis_version: str  # e.g., "deep_analysis_v3b"
    created_at: datetime

    # Frozen Worst Windows
    worst_windows: Dict[str, List[str]] = field(default_factory=dict)
    # {"20d": ["window_2024-01-05_to_2024-01-25", ...], "60d": [...]}

    # Frozen Risk Patterns
    risk_patterns: Dict[str, str] = field(default_factory=dict)
    # {"20d": "structural", "60d": "single_outlier"}

    pattern_similarity: Dict[str, float] = field(default_factory=dict)
    # {"20d": 0.897, "60d": 0.234}

    # Frozen Envelopes (P95/P99 worst-case bounds)
    envelopes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # {"20d": {"return_p95": -0.27, "return_p99": -0.35, "mdd_p95": 0.18, ...}}

    # Frozen Rule Thresholds
    rule_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # {"stability_score": 30.0, "max_drawdown": 0.05, "window_return": -0.10}

    # Frozen Stability Metrics
    stability_metrics: Dict[str, Any] = field(default_factory=dict)
    # {"stability_score": 15.0, "stability_label": "Fragile", ...}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "baseline_id": self.baseline_id,
            "strategy_id": self.strategy_id,
            "run_id": self.run_id,
            "commit_hash": self.commit_hash,
            "config_hash": self.config_hash,
            "analysis_version": self.analysis_version,
            "created_at": self.created_at.isoformat(),
            "worst_windows": self.worst_windows,
            "risk_patterns": self.risk_patterns,
            "pattern_similarity": self.pattern_similarity,
            "envelopes": self.envelopes,
            "rule_thresholds": self.rule_thresholds,
            "stability_metrics": self.stability_metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskBaseline":
        """Create from dictionary"""
        created_at = data.get("created_at", datetime.now().isoformat())
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            baseline_id=data["baseline_id"],
            strategy_id=data["strategy_id"],
            run_id=data["run_id"],
            commit_hash=data["commit_hash"],
            config_hash=data["config_hash"],
            analysis_version=data["analysis_version"],
            created_at=created_at,
            worst_windows=data.get("worst_windows", {}),
            risk_patterns=data.get("risk_patterns", {}),
            pattern_similarity=data.get("pattern_similarity", {}),
            envelopes=data.get("envelopes", {}),
            rule_thresholds=data.get("rule_thresholds", {}),
            stability_metrics=data.get("stability_metrics", {})
        )

    def get_envelope_tolerance(self, window_length: str, metric: str, tolerance_pct: float = 0.02) -> float:
        """Get allowed tolerance for envelope metric

        Args:
            window_length: Window length (e.g., "20d")
            metric: Metric name (e.g., "return_p95")
            tolerance_pct: Tolerance percentage (default 2%)

        Returns:
            Absolute tolerance value
        """
        if window_length not in self.envelopes:
            return 0.0

        baseline_value = self.envelopes[window_length].get(metric, 0.0)
        return abs(baseline_value * tolerance_pct)


@dataclass
class BaselineSnapshot:
    """Complete snapshot of all strategy baselines

    Acts as a registry for all frozen baselines in the system.
    """
    version: str
    created_at: datetime
    commit_hash: str
    baselines: List[RiskBaseline] = field(default_factory=list)

    def add_baseline(self, baseline: RiskBaseline):
        """Add a baseline to snapshot"""
        self.baselines.append(baseline)

    def get_baseline(self, baseline_id: str) -> Optional[RiskBaseline]:
        """Get baseline by ID"""
        for baseline in self.baselines:
            if baseline.baseline_id == baseline_id:
                return baseline
        return None

    def get_baselines_by_strategy(self, strategy_id: str) -> List[RiskBaseline]:
        """Get all baselines for a strategy"""
        return [b for b in self.baselines if b.strategy_id == strategy_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "commit_hash": self.commit_hash,
            "baselines": [b.to_dict() for b in self.baselines]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineSnapshot":
        """Create from dictionary"""
        created_at = data.get("created_at", datetime.now().isoformat())
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        baselines = [
            RiskBaseline.from_dict(b_data)
            for b_data in data.get("baselines", [])
        ]

        return cls(
            version=data["version"],
            created_at=created_at,
            commit_hash=data["commit_hash"],
            baselines=baselines
        )


def generate_baseline_id(strategy_id: str, commit_hash: str) -> str:
    """Generate unique baseline ID

    Args:
        strategy_id: Strategy identifier
        commit_hash: Git commit hash

    Returns:
        Unique baseline ID
    """
    # Combine strategy_id and commit_hash
    combined = f"{strategy_id}_{commit_hash}"
    # Create hash to handle special characters
    hash_suffix = hashlib.md5(combined.encode()).hexdigest()[:8]
    return f"{strategy_id}_{hash_suffix}"


if __name__ == "__main__":
    """Test baseline data structures"""
    print("=== RiskBaseline System Test ===\n")

    # Create a test baseline
    baseline = RiskBaseline(
        baseline_id="ma_5_20_a1b2c3d4",
        strategy_id="ma_5_20",
        run_id="exp_1677b52a",
        commit_hash="7c4511b",
        config_hash="abc123",
        analysis_version="deep_analysis_v3b",
        created_at=datetime.now(),
        worst_windows={
            "20d": ["window_1", "window_2", "window_3"],
            "60d": ["window_4"]
        },
        risk_patterns={
            "20d": "structural",
            "60d": "single_outlier"
        },
        pattern_similarity={
            "20d": 0.897,
            "60d": 0.234
        },
        envelopes={
            "20d": {
                "return_p95": -0.27,
                "return_p99": -0.35,
                "mdd_p95": 0.18,
                "mdd_p99": 0.25,
                "duration_p95": 15.0,
                "duration_p99": 22.0
            }
        },
        rule_thresholds={
            "stability_score": 30.0,
            "max_drawdown": 0.05,
            "window_return": -0.10
        },
        stability_metrics={
            "stability_score": 15.0,
            "stability_label": "Fragile",
            "perturbation_passed": 3,
            "perturbation_total": 10
        }
    )

    print(f"Created baseline: {baseline.baseline_id}")
    print(f"  Strategy: {baseline.strategy_id}")
    print(f"  Run: {baseline.run_id}")
    print(f"  Commit: {baseline.commit_hash}")
    print(f"  Worst windows (20d): {len(baseline.worst_windows.get('20d', []))}")

    # Test serialization
    baseline_dict = baseline.to_dict()
    baseline_restored = RiskBaseline.from_dict(baseline_dict)

    assert baseline.baseline_id == baseline_restored.baseline_id
    assert baseline.strategy_id == baseline_restored.strategy_id
    print("\n✓ Serialization test passed")

    # Test tolerance calculation
    tolerance = baseline.get_envelope_tolerance("20d", "return_p95", tolerance_pct=0.02)
    print(f"\nTolerance for return_p95 (2%): {tolerance:.4f}")
    assert tolerance > 0
    print("✓ Tolerance calculation test passed")

    print("\n✓ All tests passed")
