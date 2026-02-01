"""
生成长期回测数据（简化版）

使用 config.py 的回测配置生成足��长的回测数据
"""

import logging
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_long_backtest_with_config():
    """使用 BacktestConfig 运行长期回测"""
    from config import BacktestConfig, create_mock_config
    from agent.runner import run_backtest_with_config
    import json

    print("=" * 70)
    print("生成长期回测数据（简化版）")
    print("=" * 70)

    # 配置多个长期回测
    backtest_configs = []

    # 配置1: MA策略，120天
    end_date = date.today()
    start_date_120 = end_date - timedelta(days=120)

    backtest_configs.append({
        "name": "ma_5_20_120d",
        "config": create_mock_config(
            symbols=["AAPL"],
            start_date=start_date_120.isoformat(),
            end_date=end_date.isoformat(),
            seed=42
        )
    })

    # 配置2: MA策略，180天
    start_date_180 = end_date - timedelta(days=180)

    backtest_configs.append({
        "name": "ma_10_30_180d",
        "config": create_mock_config(
            symbols=["MSFT"],
            start_date=start_date_180.isoformat(),
            end_date=end_date.isoformat(),
            seed=43
        )
    })

    # 配置3: Breakout策略，150天
    start_date_150 = end_date - timedelta(days=150)

    backtest_configs.append({
        "name": "breakout_20_150d",
        "config": create_mock_config(
            symbols=["AAPL"],
            start_date=start_date_150.isoformat(),
            end_date=end_date.isoformat(),
            seed=44
        )
    })

    # 运行所有回测
    results = []

    for i, bt_config in enumerate(backtest_configs, 1):
        name = bt_config["name"]
        config = bt_config["config"]

        print(f"\n{'='*70}")
        print(f"回测 {i}/{len(backtest_configs)}: {name}")
        print('='*70)

        print(f"开始日期: {config.start_date}")
        print(f"结束日期: {config.end_date}")
        # 计算天数
        start = date.fromisoformat(config.start_date)
        end = date.fromisoformat(config.end_date)
        print(f"天数: {(end - start).days}")
        print(f"标的: {config.symbols}")

        try:
            # 运行回测
            result = run_backtest_with_config(config, output_dir="runs", verbose=False)

            # 获取结果指标
            total_return = result.get('metrics', {}).get('total_return', 0)
            final_equity = result.get('metrics', {}).get('final_equity', 0)
            trade_count = result.get('metrics', {}).get('trade_count', 0)

            print(f"\n✅ 回测完成！")
            print(f"   总收益: {total_return*100:.2f}%")
            print(f"   最终权益: ${final_equity:,.2f}")
            print(f"   交易次数: {trade_count}")

            # 检查窗口数量
            from analysis.replay_schema import load_replay_outputs
            from analysis.window_scanner import WindowScanner

            replays = load_replay_outputs("runs")
            latest_replay = replays[-1]  # 获取最新运行的
            num_steps = len(latest_replay.steps)

            print(f"\n🔍 窗口检查:")
            scanner = WindowScanner()

            for window_len in ["5d", "20d", "60d"]:
                windows = scanner.scan_replay(latest_replay, window_len)
                print(f"   {window_len}: {len(windows)} 个窗口")

            results.append({
                "name": name,
                "exp_id": latest_replay.run_id,
                "status": "success",
                "total_return": total_return,
                "num_steps": num_steps
            })

        except Exception as e:
            print(f"\n❌ 回测失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": name,
                "status": "failed",
                "error": str(e)
            })

    # 总结
    print("\n\n" + "=" * 70)
    print("生成总结")
    print("=" * 70)

    success_count = sum(1 for r in results if r['status'] == 'success')

    print(f"\n总回测数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(results) - success_count}")

    if success_count > 0:
        print(f"\n✅ 成功的回测:")
        for r in results:
            if r['status'] == 'success':
                print(f"   - {r['name']}: {r['num_steps']}步 "
                      f"收益={r['total_return']*100:.2f}%")

        print(f"\n💡 下一步:")
        print(f"   1. 重新生成基线数据:")
        print(f"      PYTHONPATH=/home/neal/mkcs python -c '")
        print(f"        from analysis.baseline_manager import BaselineManager;")
        print(f"        mgr = BaselineManager();")
        print(f"        mgr.freeze_baselines(\"runs\", \"baselines/risk\")'")
        print(f"      '")
        print(f"\n   2. 运行回归测试:")
        print(f"      PYTHONPATH=/home/neal/mkcs python tests/risk_regression/run_risk_regression.py")

    print("=" * 70)


if __name__ == "__main__":
    run_long_backtest_with_config()
