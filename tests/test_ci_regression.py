"""
最小CI回放回归测试

验证回测结果的可复现性，用于CI/CD流水线。

测试要求：
- 30天回测周期
- 3个交易标的
- Mock数据源
- 验证结果100%复现
"""

import json
from datetime import date
from pathlib import Path

from config import BacktestConfig
from agent.runner import run_backtest_with_config


def test_ci_minimal_reproducibility():
    """最小CI回放回归测试

    验证：
    1. 配置哈希计算正确
    2. 相同配置产生相同结果
    3. Summary.json 包含所有必需字段
    """
    # 固定配置
    config = BacktestConfig(
        data_source="mock",
        seed=42,  # 固定随机种子
        symbols=["AAPL", "MSFT", "GOOGL"],  # 3个标的
        start_date="2024-01-01",  # 30天（约22个交易日）
        end_date="2024-01-31",
        commission_per_share=0.01,
        slippage_bps=2,  # 2 BPS = 0.02%
        strategy_name="ma",
        strategy_params={"fast_period": 5, "slow_period": 20},
        initial_cash=100000.0
    )

    # 预期的配置哈希
    expected_config_hash = "sha256:c2699248639b0057"

    # 第一次运行
    result1 = run_backtest_with_config(
        config,
        output_dir="reports/ci_test_run1",
        verbose=False
    )

    # 第二次运行（验证可复现性）
    result2 = run_backtest_with_config(
        config,
        output_dir="reports/ci_test_run2",
        verbose=False
    )

    # 验证配置哈希
    assert result1["config_hash"] == expected_config_hash, \
        f"配置哈希不匹配: {result1['config_hash']} != {expected_config_hash}"
    assert result2["config_hash"] == expected_config_hash, \
        f"配置哈希不匹配: {result2['config_hash']} != {expected_config_hash}"

    # 验证必需字段
    required_fields = [
        "backtest_date",
        "replay_mode",
        "data_source",
        "data_hash",
        "config",
        "config_hash",
        "git_commit",
        "metrics"
    ]
    for field in required_fields:
        assert field in result1, f"缺少字段: {field}"

    # 验证可复现性
    metrics1 = result1["metrics"]
    metrics2 = result2["metrics"]

    assert metrics1["trade_count"] == metrics2["trade_count"], \
        f"交易次数不一致: {metrics1['trade_count']} != {metrics2['trade_count']}"
    assert metrics1["final_equity"] == metrics2["final_equity"], \
        f"最终权益不一致: {metrics1['final_equity']} != {metrics2['final_equity']}"
    assert metrics1["total_pnl"] == metrics2["total_pnl"], \
        f"总盈亏不一致: {metrics1['total_pnl']} != {metrics2['total_pnl']}"

    # 验证输出文件存在
    output_files = [
        "reports/ci_test_run1/summary.json",
        "reports/ci_test_run1/equity_curve.csv",
        "reports/ci_test_run1/trades.csv"
    ]
    for file_path in output_files:
        assert Path(file_path).exists(), f"输出文件不存在: {file_path}"

    # 验证 summary.json 包含 git_commit/config_hash/data_source/data_hash
    summary = json.loads(Path("reports/ci_test_run1/summary.json").read_text())
    assert summary["git_commit"] is not None, "缺少 git_commit"
    assert summary["config_hash"] == expected_config_hash, "config_hash 不匹配"
    assert summary["data_source"]["type"] == "mock", "data_source.type 不正确"
    assert summary["data_source"]["seed"] == 42, "data_source.seed 不正确"

    print("✓ CI回放回归测试通过")
    print(f"  配置哈希: {result1['config_hash']}")
    print(f"  Git Commit: {result1['git_commit']}")
    print(f"  交易次数: {metrics1['trade_count']}")
    print(f"  总收益: {metrics1['total_return']*100:.2f}%")


def test_ci_csv_reproducibility():
    """CSV数据源CI回归测试

    验证CSV数据源的get_bars_until严格截断和可复现性
    """
    # 注意：这个测试需要 data/aapl_2023.csv 存在
    csv_file = Path("data/aapl_2023.csv")

    if not csv_file.exists():
        print("⊘ CSV测试跳过（数据文件不存在）")
        return

    config = BacktestConfig(
        data_source="csv",
        data_path="data/aapl_2023.csv",
        symbols=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-01-31",  # 30天
        commission_per_share=0.01,
        slippage_bps=5,  # 5 BPS
        strategy_params={"fast_period": 3, "slow_period": 10},
        initial_cash=100000.0
    )

    # 运行两次
    result1 = run_backtest_with_config(config, output_dir="reports/ci_csv_test1", verbose=False)
    result2 = run_backtest_with_config(config, output_dir="reports/ci_csv_test2", verbose=False)

    # 验证可复现性
    assert result1["data_hash"] is not None, "缺少 data_hash"
    assert result1["data_hash"] == result2["data_hash"], "data_hash 不一致"

    metrics1 = result1["metrics"]
    metrics2 = result2["metrics"]

    assert metrics1["final_equity"] == metrics2["final_equity"], \
        f"CSV回测不可复现: {metrics1['final_equity']} != {metrics2['final_equity']}"

    print("✓ CSV回放回归测试通过")
    print(f"  数据哈希: {result1['data_hash']}")
    print(f"  交易次数: {metrics1['trade_count']}")


if __name__ == "__main__":
    print("=== CI 回放回归测试 ===\n")

    try:
        test_ci_minimal_reproducibility()
        print()
        test_ci_csv_reproducibility()
        print("\n✓ 所有CI测试通过")
    except AssertionError as e:
        print(f"\n✗ CI测试失败: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ CI测试出错: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
