# Risk Baselines (SPL-4c)

This directory contains frozen risk baselines for regression testing of SPL-3b worst-case findings.

## Purpose

Risk baselines serve as immutable reference points to ensure that strategy risk characteristics do not regress over time. They are created by freezing SPL-3b deep analysis results at a specific commit.

## Directory Structure

```
baselines/risk/
├── baseline_manifest.json       # Registry of all baselines
├── baselines_v1.json            # All frozen baselines (v1 format)
└── README.md                    # This file
```

## Files

### baseline_manifest.json

Registry file containing metadata for all baselines:

```json
{
  "version": "1",
  "created_at": "2026-02-01T00:00:00",
  "commit_hash": "7c4511b",
  "baselines": [
    {
      "baseline_id": "ma_5_20_exp_1677b52a",
      "strategy_id": "ma_5_20",
      "run_id": "exp_1677b52a",
      "file": "baselines_v1.json"
    }
  ]
}
```

### baselines_v1.json

Contains all frozen baseline data in v1 format. Each baseline includes:

- **Identification**: baseline_id, strategy_id, run_id
- **Traceability**: commit_hash, config_hash, analysis_version, created_at
- **Frozen Worst Windows**: List of worst-case window IDs per window length
- **Frozen Risk Patterns**: Risk pattern type (structural/single_outlier) per window length
- **Pattern Similarity**: Similarity scores per window length
- **Frozen Envelopes**: P95/P99 worst-case bounds (return, MDD, duration)
- **Frozen Rule Thresholds**: Trigger thresholds for risk rules
- **Stability Metrics**: Stability score, label, and test results

## Creating Baselines

To freeze current SPL-3b results as baselines:

```bash
# From project root
python -c "
from analysis.baseline_manager import BaselineManager

mgr = BaselineManager()
snapshot = mgr.freeze_baselines(
    replay_dir='runs',
    output_dir='baselines/risk',
    window_lengths=['20d', '60d']
)

print(f'Frozen {len(snapshot.baselines)} baselines')
"
```

## Running Regression Tests

To run risk regression tests:

```bash
# Run all tests
python tests/risk_regression/run_risk_regression.py

# With custom settings
python tests/risk_regression/run_risk_regression.py \
    --replay-dir runs \
    --baseline-dir baselines/risk \
    --output-dir reports/risk_regression \
    --tolerance 0.02
```

## Baseline Lifecycle

1. **Creation**: Run SPL-3b deep analysis on strategy replays
2. **Freezing**: Use BaselineManager to freeze results as baselines
3. **Testing**: Run regression tests to detect drift
4. **Versioning**: When baselines need updating, create new version files

## Interpreting Test Results

### PASS

All risk metrics are within tolerance of baseline. Strategy risk characteristics are stable.

### FAIL

Risk metrics have regressed:
- **Worst-Window Drift**: New worst-case windows appeared
- **Structural Change**: Risk pattern type changed
- **Envelope Regression**: Worst-case bounds significantly worse
- **Rule Trigger Failure**: Baseline rules no longer trigger

### SKIP

Test could not be run (e.g., missing data, not implemented)

## Version History

- **v1** (2026-02-01): Initial baseline format

## Maintenance Guidelines

### When to Update Baselines

Baselines should be updated when:
1. Intentional strategy improvements are made
2. Risk characteristics are expected to change
3. Moving to a new major version of the analysis system

### When NOT to Update Baselines

Baselines should NOT be updated when:
1. Test failures are due to bugs or regressions
2. Risk characteristics have unexpectedly degraded
3. The update would mask real problems

### Update Process

1. Investigate the root cause of test failures
2. If intentional, document the reason for updating
3. Create new baseline version file (e.g., `baselines_v2.json`)
4. Update manifest to reference new version
5. Keep old versions for historical comparison

## CI/CD Integration

These baselines are used in CI to block progression on risk regression:

```yaml
# .github/workflows/ci.yml
- name: Run Risk Regression Tests
  run: |
    python tests/risk_regression/run_risk_regression.py

# This will exit with error 1 if any tests FAIL
```

## Contact

For questions about risk baselines, see:
- SPL-3b documentation: `analysis/deep_analysis_report.py`
- SPL-4c test suite: `tests/risk_regression/risk_baseline_test.py`
- Baseline manager: `analysis/baseline_manager.py`
