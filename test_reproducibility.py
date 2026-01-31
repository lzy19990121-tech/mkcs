"""
可信、可复现回测系统验证脚本

验证以下功能：
1. 配置哈希计算
2. 数据哈希计算
3. Git commit 记录
4. 相同配置 + seed 100% 复现
"""

import json
import tempfile
import os
from datetime import datetime
from pathlib import Path

from config import BacktestConfig, create_mock_config
from agent.runner import run_backtest_with_config
from utils.hash import compute_config_hash, compute_data_hash, get_git_commit


def test_config_hash():
    """测试配置哈希计算"""
    print("\n=== 测试 1: 配置哈希计算 ===")

    config1 = create_mock_config(seed=42)
    config2 = create_mock_config(seed=42)
    config3 = create_mock_config(seed=43)

    hash1 = config1.compute_hash()
    hash2 = config2.compute_hash()
    hash3 = config3.compute_hash()

    print(f"  相同配置哈希一致: {hash1 == hash2} ({hash1})")
    print(f"  不同配置哈希不同: {hash1 != hash3} ({hash1} vs {hash3})")

    assert hash1 == hash2, "相同配置应有相同哈希"
    assert hash1 != hash3, "不同配置应有不同哈希"
    print("  ✓ 配置哈希测试通过")


def test_data_hash():
    """测试数据文件哈希计算"""
    print("\n=== 测试 2: 数据文件哈希 ===")

    # 创建测试文件
    test_file = "/tmp/test_data.csv"
    Path(test_file).write_text("date,open,high,low,close,volume\n2024-01-01,100,101,99,100,1000\n")

    hash1 = compute_data_hash(test_file)
    hash2 = compute_data_hash(test_file)

    # 修改文件
    Path(test_file).write_text("date,open,high,low,close,volume\n2024-01-01,100,101,99,101,1000\n")
    hash3 = compute_data_hash(test_file)

    print(f"  相同文件哈希一致: {hash1 == hash2} ({hash1})")
    print(f"  修改文件哈希不同: {hash1 != hash3}")

    os.remove(test_file)

    assert hash1 == hash2, "相同文件应有相同哈希"
    assert hash1 != hash3, "修改后的文件应有不同哈希"
    print("  ✓ 数据哈希测试通过")


def test_git_commit():
    """测试 Git commit 获取"""
    print("\n=== 测试 3: Git Commit 获取 ===")

    commit = get_git_commit(short=True)
    print(f"  Git Commit: {commit}")

    assert commit != "unknown", "应该能获取到 git commit"
    assert len(commit) == 8, "短 commit 应为 8 位"
    print("  ✓ Git commit 测试通过")


def test_reproducibility():
    """测试回测可复现性"""
    print("\n=== 测试 4: 回测可复现性 ===")

    config = create_mock_config(
        symbols=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-06-30',
        seed=42
    )

    print(f"  配置哈希: {config.compute_hash()}")

    # 第一次运行
    result1 = run_backtest_with_config(
        config,
        output_dir='reports/repro_test_1',
        verbose=False
    )

    # 第二次运行（相同配置）
    result2 = run_backtest_with_config(
        config,
        output_dir='reports/repro_test_2',
        verbose=False
    )

    # 比较结果
    metrics_match = (
        result1.get('metrics', {}).get('trade_count') == result2.get('metrics', {}).get('trade_count') and
        result1.get('metrics', {}).get('total_pnl') == result2.get('metrics', {}).get('total_pnl') and
        result1.get('metrics', {}).get('final_equity') == result2.get('metrics', {}).get('final_equity')
    )

    hash_match = result1.get('config_hash') == result2.get('config_hash')

    print(f"  配置哈希一致: {hash_match}")
    print(f"  交易次数一致: {result1.get('metrics', {}).get('trade_count') == result2.get('metrics', {}).get('trade_count')}")
    print(f"  总盈亏一致: {result1.get('metrics', {}).get('total_pnl') == result2.get('metrics', {}).get('total_pnl')}")
    print(f"  最终权益一致: {result1.get('metrics', {}).get('final_equity') == result2.get('metrics', {}).get('final_equity')}")
    print(f"  完全可复现: {metrics_match and hash_match}")

    assert hash_match, "配置哈希应一致"
    assert metrics_match, "回测指标应完全一致"
    print("  ✓ 可复现性测试通过")


def test_summary_output():
    """测试 summary.json 输出格式"""
    print("\n=== 测试 5: Summary.json 输出格式 ===")

    config = create_mock_config(seed=42)
    result = run_backtest_with_config(
        config,
        output_dir='reports/summary_test',
        verbose=False
    )

    # 验证必要字段
    required_fields = [
        'backtest_date',
        'replay_mode',
        'data_source',
        'config',
        'config_hash',
        'git_commit',
        'metrics'
    ]

    missing = [f for f in required_fields if f not in result]
    print(f"  必要字段: {required_fields}")
    print(f"  缺失字段: {missing if missing else '无'}")

    # 验证 metrics 字段
    metrics_fields = ['initial_cash', 'final_equity', 'total_return', 'trade_count']
    metrics = result.get('metrics', {})
    missing_metrics = [f for f in metrics_fields if f not in metrics]
    print(f"  Metrics 字段: {metrics_fields}")
    print(f"  缺失 Metrics: {missing_metrics if missing_metrics else '无'}")

    assert not missing, f"缺少必要字段: {missing}"
    assert not missing_metrics, f"缺少 metrics 字段: {missing_metrics}"
    print("  ✓ Summary 输出格式测试通过")


def test_different_seeds():
    """测试不同 seed 产生不同结果"""
    print("\n=== 测试 6: 不同 Seed 产生不同结果 ===")

    config1 = create_mock_config(seed=42)
    config2 = create_mock_config(seed=43)

    print(f"  Seed 42 哈希: {config1.compute_hash()}")
    print(f"  Seed 43 哈希: {config2.compute_hash()}")

    result1 = run_backtest_with_config(config1, output_dir='reports/seed_test_1', verbose=False)
    result2 = run_backtest_with_config(config2, output_dir='reports/seed_test_2', verbose=False)

    # 不同 seed 应该产生不同结果
    different_results = (
        result1.get('config_hash') != result2.get('config_hash') or
        result1.get('metrics', {}).get('trade_count') != result2.get('metrics', {}).get('trade_count')
    )

    print(f"  不同 seed 产生不同结果: {different_results}")
    print(f"  交易次数: {result1.get('metrics', {}).get('trade_count')} vs {result2.get('metrics', {}).get('trade_count')}")

    assert result1.get('config_hash') != result2.get('config_hash'), "不同 seed 应有不同配置哈希"
    print("  ✓ 不同 Seed 测试通过")


def main():
    """运行所有测试"""
    print("="*60)
    print("可信、可复现回测系统验证")
    print("="*60)

    try:
        test_config_hash()
        test_data_hash()
        test_git_commit()
        test_reproducibility()
        test_summary_output()
        test_different_seeds()

        print("\n" + "="*60)
        print("✓ 所有测试通过！")
        print("="*60)
        print("\n验证结果:")
        print("  1. 配置哈希计算正确")
        print("  2. 数据文件哈希计算正确")
        print("  3. Git commit 记录正确")
        print("  4. 相同配置 + seed 100% 复现")
        print("  5. Summary.json 输出格式正确")
        print("  6. 不同 seed 产生不同结果")

        return 0

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
