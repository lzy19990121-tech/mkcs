"""
Pytest fixtures for MKCS tests
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any

# 添加项目根目录
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analysis.replay_schema import ReplayOutput, StepRecord, TradeRecord


@pytest.fixture
def sample_replay_output() -> ReplayOutput:
    """创建示例 ReplayOutput 用于测试"""
    steps = [
        StepRecord(
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            step_pnl=Decimal("10.0"),
            equity=Decimal("100010.0"),
            run_id="test_run",
            commit_hash="abc123",
            config_hash="def456"
        ),
        StepRecord(
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            step_pnl=Decimal("-5.0"),
            equity=Decimal("100005.0"),
            run_id="test_run",
            commit_hash="abc123",
            config_hash="def456"
        )
    ]

    trades = [
        TradeRecord(
            trade_id="trade_1",
            timestamp=datetime.now(),
            symbol="AAPL",
            side="BUY",
            price=Decimal("150.0"),
            quantity=100,
            commission=Decimal("1.0")
        )
    ]

    return ReplayOutput(
        run_id="test_run",
        strategy_id="test_strategy",
        strategy_name="Test Strategy",
        commit_hash="abc123",
        config_hash="def456",
        start_date=date.today(),
        end_date=date.today(),
        initial_cash=Decimal("100000.0"),
        final_equity=Decimal("100005.0"),
        steps=steps,
        trades=trades,
        config={"symbol": "AAPL"}
    )


@pytest.fixture
def baseline() -> ReplayOutput:
    """Baseline replay output for regression tests"""
    # 创建一个 baseline 数据
    steps = []
    for i in range(50):
        pnl = float(np.random.randn() * 50 + 20)  # 平均正收益
        step = StepRecord(
            timestamp=datetime.now(),
            strategy_id="baseline_strategy",
            step_pnl=Decimal(str(pnl)),
            equity=Decimal("100000.0") + Decimal(str(i * 20)),
            run_id="baseline_run",
            commit_hash="base123",
            config_hash="base456"
        )
        steps.append(step)

    return ReplayOutput(
        run_id="baseline_run",
        strategy_id="baseline_strategy",
        strategy_name="Baseline Strategy",
        commit_hash="base123",
        config_hash="base456",
        start_date=date.today(),
        end_date=date.today(),
        initial_cash=Decimal("100000.0"),
        final_equity=Decimal("101000.0"),
        steps=steps,
        config={"symbol": "AAPL", "label": "baseline"}
    )


@pytest.fixture
def current() -> ReplayOutput:
    """Current replay output for regression tests"""
    # 创建一个当前数据，类似 baseline
    steps = []
    for i in range(50):
        pnl = float(np.random.randn() * 50 + 20)  # 平均正收益
        step = StepRecord(
            timestamp=datetime.now(),
            strategy_id="current_strategy",
            step_pnl=Decimal(str(pnl)),
            equity=Decimal("100000.0") + Decimal(str(i * 20)),
            run_id="current_run",
            commit_hash="curr123",
            config_hash="curr456"
        )
        steps.append(step)

    return ReplayOutput(
        run_id="current_run",
        strategy_id="current_strategy",
        strategy_name="Current Strategy",
        commit_hash="curr123",
        config_hash="curr456",
        start_date=date.today(),
        end_date=date.today(),
        initial_cash=Decimal("100000.0"),
        final_equity=Decimal("101000.0"),
        steps=steps,
        config={"symbol": "AAPL", "label": "current"}
    )


@pytest.fixture
def optimizer_metrics() -> Dict[str, float]:
    """Optimizer metrics for testing"""
    return {
        "cvar_95": -0.05,  # -5% (within budget of -10%)
        "cvar_99": -0.08,  # -8% (within budget of -15%)
        "max_drawdown": 0.08,  # 8% (within budget of 12%)
        "tail_correlation": 0.6,
        "co_crash_count": 2
    }


@pytest.fixture
def risk_budgets() -> Dict[str, float]:
    """Risk budgets for testing"""
    return {
        "cvar_95_budget": -0.10,  # -10%
        "cvar_99_budget": -0.15,  # -15%
        "max_drawdown_budget": 0.12  # 12%
    }


@pytest.fixture
def correlation_data() -> Dict[str, Any]:
    """Correlation data for testing"""
    np.random.seed(42)
    n_steps = 100
    n_strategies = 5

    # 生成收益数据
    returns = np.random.randn(n_steps, n_strategies) * 0.02

    # 计算滚动相关性
    window = 20
    correlations = []
    for i in range(window, n_steps):
        window_returns = returns[i-window:i]
        corr_matrix = np.corrcoef(window_returns.T)
        # 取上三角矩阵的平均值
        upper_tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
        correlations.append(np.mean(np.abs(upper_tri)))

    # 添加一些相关性激增
    spike_threshold = 0.8
    spike_indices = [50, 75]

    return {
        "returns": returns,
        "correlations": correlations,
        "spike_threshold": spike_threshold,
        "spike_indices": spike_indices,
        "spike_frequency": len(spike_indices) / len(correlations)
    }


@pytest.fixture
def co_crash_data() -> Dict[str, Any]:
    """Co-crash data for testing"""
    np.random.seed(42)
    n_steps = 100
    n_strategies = 5

    # 生成收益数据，包含一些极端负收益日
    returns = np.random.randn(n_steps, n_strategies) * 0.02

    # 添加协同下跌日
    crash_days = [30, 60, 90]
    for day in crash_days:
        returns[day] = -0.03 + np.random.randn(n_strategies) * 0.005

    crash_threshold = -0.02

    return {
        "returns": returns,
        "crash_threshold": crash_threshold,
        "crash_days": crash_days,
        "co_crash_count": len(crash_days)
    }


@pytest.fixture
def stability_metrics() -> Dict[str, float]:
    """Optimizer stability metrics for testing"""
    return {
        "correlation_spike_frequency": 0.05,  # 5% (below threshold of 10%)
        "co_crash_count": 2,  # (below threshold of 5)
        "tail_correlation": 0.65,  # (below threshold of 0.8)
        "max_weight_jitter": 0.15,  # (below threshold of 0.25)
        "weight_change_frequency": 0.08  # (below threshold of 0.15)
    }


@pytest.fixture
def stability_thresholds() -> Dict[str, float]:
    """Stability thresholds for testing"""
    return {
        "correlation_spike_threshold": 0.10,  # 10%
        "co_crash_threshold": 5,
        "tail_correlation_threshold": 0.80,
        "max_weight_jitter_threshold": 0.25,
        "weight_change_frequency_threshold": 0.15
    }


@pytest.fixture
def current_weights() -> Dict[str, float]:
    """Current portfolio weights for testing"""
    return {
        "strategy_1": 0.20,
        "strategy_2": 0.15,
        "strategy_3": 0.25,
        "strategy_4": 0.20,
        "strategy_5": 0.20
    }


@pytest.fixture
def previous_weights() -> Dict[str, float]:
    """Previous portfolio weights for testing (slightly different)"""
    return {
        "strategy_1": 0.18,  # change of 0.02
        "strategy_2": 0.16,  # change of 0.01
        "strategy_3": 0.26,  # change of 0.01
        "strategy_4": 0.20,  # change of 0.00
        "strategy_5": 0.20   # change of 0.00
    }
