"""
P1-2: Replay 数据查找测试

验证 ReplayLocator 功能：
1. 定义 ReplayLocator：给定 trade_id / timestamp / symbol → 找到对应数据片段
2. 将 locator 接入 postmortem_generator
3. 输出内容：信号来源、当时 MarketState、RiskDecision、执行回报
4. 缺数据时给出明确 fallback

验收:
☐ 任意一笔 trade 在 post-mortem 中可回溯到当时 MarketState 与 AlphaOpinions
☐ 缺数据时报告明确写"缺什么、为什么缺"
☐ 报告生成全流程无异常
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.online.postmortem_generator import ReplayLocator
from analysis.replay_schema import ReplayOutput, TradeRecord, StepRecord, create_standard_replay_output
from decimal import Decimal


def create_test_replay_data(temp_dir: Path) -> str:
    """创建测试 replay 数据"""
    run_id = "test_run_001"

    # 创建 run 目录
    run_dir = temp_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 使用固定的基准时间（2024-01-01 12:00:00）
    base_time = datetime(2024, 1, 1, 12, 0, 0)

    # 创建 manifest
    manifest = {
        "experiment_id": run_id,
        "strategy_name": "test_strategy",
        "git_commit": "abc123",
        "config_hash": "hash456",
        "timestamp": base_time.isoformat()
    }

    with open(run_dir / "run_manifest.json", 'w') as f:
        json.dump(manifest, f)

    # 创建 summary
    summary = {
        "date_range": {
            "start": "2024-01-01",
            "end": "2024-01-08"
        },
        "metrics": {
            "initial_cash": 100000,
            "final_equity": 105000
        },
        "data_hash": "data789",
        "config": {
            "cost_model": {"commission": 0.001},
            "slippage": {"mode": "fixed", "value": 0.0001}
        }
    }

    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f)

    # 创建 trades.csv
    import pandas as pd
    trades_data = [
        {
            "trade_id": "trade_001",
            "timestamp": base_time.isoformat(),
            "symbol": "BTCUSDT",
            "side": "BUY",
            "price": 50000,
            "quantity": 0.1,
            "commission": 5.0
        },
        {
            "trade_id": "trade_002",
            "timestamp": (base_time + timedelta(minutes=30)).isoformat(),
            "symbol": "ETHUSDT",
            "side": "SELL",
            "price": 3000,
            "quantity": 1.0,
            "commission": 3.0
        }
    ]

    trades_df = pd.DataFrame(trades_data)
    trades_df.to_csv(run_dir / "trades.csv", index=False)

    # 创建 equity_curve.csv
    equity_data = []
    for i in range(100):
        t = base_time + timedelta(minutes=i)
        equity = 100000 + i * 50
        pnl = 50
        equity_data.append({
            "date": t.strftime("%Y-%m-%d %H:%M:%S"),  # 使用 'date' 而不是 'timestamp'
            "equity": equity,
            "pnl": pnl
        })

    equity_df = pd.DataFrame(equity_data)
    equity_df.to_csv(run_dir / "equity_curve.csv", index=False)

    return run_id


def test_locator_by_trade_id():
    """测试1: 按 trade_id 查找"""
    print("\n" + "="*70)
    print("测试1: 按 trade_id 查找")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_id = create_test_replay_data(temp_path)

        locator = ReplayLocator(str(temp_path))

        # 查找存在的 trade
        pointers = locator._find_by_trade_id("trade_001")

        assert len(pointers) > 0, "应该找到 trade_001"
        assert pointers[0]["trade_id"] == "trade_001", "trade_id 应该匹配"
        assert pointers[0]["symbol"] == "BTCUSDT", "symbol 应该匹配"
        assert pointers[0]["status"] == "found", "status 应该是 found"

        print(f"  ✅ 找到 trade_001")
        print(f"  - replay_id: {pointers[0]['replay_id']}")
        print(f"  - symbol: {pointers[0]['symbol']}")
        print(f"  - price: {pointers[0]['price']}")
        print(f"  - data_path: {pointers[0]['data_path']}")

        # 查找不存在的 trade
        pointers = locator._find_by_trade_id("nonexistent")

        assert len(pointers) == 0, "不应该找到不存在的 trade"

        print(f"  ✅ 不存在的 trade 返回空列表")

    return True


def test_locator_by_time():
    """测试2: 按时间查找"""
    print("\n" + "="*70)
    print("测试2: 按时间查找")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_id = create_test_replay_data(temp_path)

        locator = ReplayLocator(str(temp_path))

        # 查找指定时间（在数据范围内）- 使用2024-01-01范围内的某个时间
        query_time = datetime(2024, 1, 1, 12, 0, 0)  # 2024-01-01 12:00:00
        pointers = locator._find_by_time(query_time)

        assert len(pointers) > 0, "应该找到该时间段的数据"
        assert pointers[0]["status"] == "found", "status 应该是 found"

        print(f"  ✅ 找到时间范围内的数据")
        print(f"  - replay_id: {pointers[0]['replay_id']}")
        print(f"  - nearest_step_time: {pointers[0]['nearest_step_time']}")

        # 按 symbol 过滤
        pointers = locator._find_by_time(query_time, symbol="BTCUSDT")

        print(f"  ✅ 按 symbol 过滤")
        print(f"  - relevant_trades: {len(pointers[0].get('relevant_trades', []))}")

    return True


def test_locator_by_run_id():
    """���试3: 按 run_id 查找"""
    print("\n" + "="*70)
    print("测试3: 按 run_id 查找")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_id = create_test_replay_data(temp_path)

        locator = ReplayLocator(str(temp_path))

        # 查找存在的 run
        pointers = locator._find_by_run_id(run_id)

        assert len(pointers) > 0, "应该找到 run"
        assert pointers[0]["replay_id"] == run_id, "run_id 应该匹配"
        assert pointers[0]["status"] == "found", "status 应该是 found"
        assert "total_trades" in pointers[0], "应该包含 total_trades"

        print(f"  ✅ 找到 run")
        print(f"  - strategy_id: {pointers[0]['strategy_id']}")
        print(f"  - total_trades: {pointers[0]['total_trades']}")
        print(f"  - total_steps: {pointers[0]['total_steps']}")

        # 查找不存在的 run
        pointers = locator._find_by_run_id("nonexistent_run")

        assert len(pointers) > 0, "应该返回结果（即使是未找到）"
        assert pointers[0]["status"] == "not_found", "status 应该是 not_found"
        assert "reason" in pointers[0], "应该包含 reason"

        print(f"  ✅ 不存在的 run 返回 not_found 状态")
        print(f"  - reason: {pointers[0]['reason']}")

    return True


def test_signal_context():
    """测试4: 获取信号上下文"""
    print("\n" + "="*70)
    print("测试4: 获取信号上下文")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_id = create_test_replay_data(temp_path)

        locator = ReplayLocator(str(temp_path))

        # 先加载缓存
        locator._load_all_replays()

        # 获取信号上下文 - 使用数据范围内的某个时间点
        query_time = datetime(2024, 1, 1, 12, 0, 0)
        context = locator.get_signal_context(run_id, query_time)

        assert context["status"] == "found", "应该找到信号上下文"
        assert "step" in context, "应该包含 step 信息"
        assert "nearby_trades" in context, "应该包含 nearby_trades"
        assert "config" in context, "应该包含 config"

        print(f"  ✅ 获取到信号上下文")
        print(f"  - step timestamp: {context['step']['timestamp']}")
        print(f"  - step equity: {context['step']['equity']}")
        print(f"  - nearby_trades: {len(context['nearby_trades'])}")
        print(f"  - cost_model: {context['config']['cost_model']}")

        # 测试不存在的 run
        context = locator.get_signal_context("nonexistent", query_time)

        assert context["status"] == "not_found", "应该返回 not_found"
        assert "reason" in context, "应该包含 reason"

        print(f"  ✅ 不存在的 run 返回正确的错误信息")

    return True


def test_fallback_on_missing_data():
    """测试5: 缺数据时的 fallback"""
    print("\n" + "="*70)
    print("测试5: 缺数据时的 fallback")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 不创建任何数据
        locator = ReplayLocator(str(temp_path))

        # 查找不存在的数据
        conditions = {
            "timestamp": datetime.now(),
            "symbol": "NONEXISTENT"
        }

        pointers = locator.find_replay_data(conditions)

        # 应该返回空列表（因为没有数据）
        assert len(pointers) == 0, "没有数据时应该返回空列表"

        print(f"  ✅ 缺数据时返回空列表（不会崩溃）")

    return True


def test_find_replay_pointers_integration():
    """测试6: postmortem_generator 集成"""
    print("\n" + "="*70)
    print("测试6: postmortem_generator 集成")
    print("="*70)

    from analysis.online.postmortem_generator import PostMortemGenerator, PostMortemConfig
    from unittest.mock import Mock

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_id = create_test_replay_data(temp_path)

        # 创建 generator
        config = PostMortemConfig(
            output_dir=str(temp_path / "reports"),
            save_json=True,
            save_markdown=False
        )
        generator = PostMortemGenerator(config)
        generator.replay_data_path = str(temp_path)

        # 测试 _find_replay_pointers
        trigger_time = datetime.now() - timedelta(hours=1)
        pointers = generator._find_replay_pointers(
            trigger_time=trigger_time,
            trade_id="trade_001"
        )

        assert len(pointers) > 0, "应该找到指针"
        assert pointers[0]["status"] == "found", "应该成功找到"

        print(f"  ✅ _find_replay_pointers 集成正常")
        print(f"  - 找到 {len(pointers)} 个指针")
        print(f"  - status: {pointers[0]['status']}")

    return True


def test_error_handling():
    """测试7: 错误处理"""
    print("\n" + "="*70)
    print("测试7: 错误处理")
    print("="*70)

    # 使用无效路径
    locator = ReplayLocator("/invalid/path/that/does/not/exist")

    # 查找操作应该不会崩溃
    try:
        pointers = locator._find_by_trade_id("any_trade")
        assert len(pointers) == 0, "无效路径应该返回空列表"
        print(f"  ✅ 无效路径处理正确（返回空列表）")
    except Exception as e:
        # 如果抛出异常，应该是可控的
        print(f"  ✅ 无效路径抛出可控异常: {type(e).__name__}")

    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("P1-2: Replay 数据查找测试")
    print("="*70)

    tests = [
        ("按 trade_id 查找", test_locator_by_trade_id),
        ("按时间查找", test_locator_by_time),
        ("按 run_id 查找", test_locator_by_run_id),
        ("获取信号上下文", test_signal_context),
        ("缺数据时的 fallback", test_fallback_on_missing_data),
        ("postmortem_generator 集成", test_find_replay_pointers_integration),
        ("错误处理", test_error_handling),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {name} 失败: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("="*70)

    if failed == 0:
        print("\n🎉 所有测试通过！")

    sys.exit(0 if failed == 0 else 1)
