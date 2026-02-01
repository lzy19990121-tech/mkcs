"""
生成长期回测数据（至少90天）

为 SPL-4c 测试生成足够的数据，确保所有窗口长度都能工作。
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

def run_long_backtest(
    symbol: str = "AAPL",
    start_date: date = None,
    end_date: date = None,
    days: int = 120,
    strategy_name: str = "ma",
    strategy_params: dict = None,
    initial_cash: float = 100000.0
):
    """运行长期回测

    Args:
        symbol: 交易标的
        start_date: 开始日期（可选，默认为 end_date - days）
        end_date: 结束日期（可选，默认为今天）
        days: 回测天数（默认120天，约4个月）
        strategy_name: 策略名称
        strategy_params: 策略参数
        initial_cash: 初始资金

    Returns:
        回测结果
    """
    from skills.market_data.yahoo_source import YahooFinanceSource
    from skills.market_data.mock_source import MockMarketSource
    from skills.strategy.moving_average import MAStrategy
    from skills.strategy.breakout import BreakoutStrategy
    from broker.paper import PaperBroker
    from agent.replay_engine import ReplayEngine
    from utils.hash import compute_config_hash
    from reports.metrics import MetricsCalculator
    import json

    print("=" * 70)
    print(f"生成长期回测数据: {symbol}")
    print("=" * 70)

    # 设置日期
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=days)

    print(f"\n📅 回测区间: {start_date} ~ {end_date} ({days}天)")
    print(f"💰 初始资金: ${initial_cash:,.2f}")
    print(f"📈 交易标的: {symbol}")
    print(f"🎯 策略: {strategy_name}")

    # 设置策略参数
    if strategy_params is None:
        if strategy_name == "ma":
            strategy_params = {"fast_period": 5, "slow_period": 20}
        elif strategy_name == "breakout":
            strategy_params = {"period": 20, "threshold": 0.01}
        else:
            strategy_params = {}

    print(f"   参数: {strategy_params}")

    # 创建数据源
    print("\n🔌 初始化数据源...")
    use_yahoo = False
    try:
        yahoo_source = YahooFinanceSource(enable_cache=True)
        # 测试连接
        test_bars = yahoo_source.get_bars(symbol, start_date, end_date, "1d")
        if len(test_bars) > 0:
            print(f"   ✓ Yahoo Finance 连接成功 (获取 {len(test_bars)} 条数据)")
            data_source = yahoo_source
            use_yahoo = True
        else:
            raise Exception("Yahoo Finance 返回空数据")
    except Exception as e:
        print(f"   ⚠ Yahoo Finance 连接失败: {e}")
        print("   使用 Mock 数据源（种子=42）")
        data_source = MockMarketSource(
            seed=42,
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date
        )

    # 创建策略
    print("\n⚙️ 创建策略...")
    if strategy_name == "ma":
        strategy = MAStrategy(**strategy_params)
    elif strategy_name == "breakout":
        strategy = BreakoutStrategy(**strategy_params)
    else:
        raise ValueError(f"未知策略: {strategy_name}")

    # 创建经纪商
    broker = PaperBroker(initial_cash=Decimal(str(initial_cash)))

    # 创建回测引擎
    print("\n🚀 运行回测...")
    engine = ReplayEngine(
        strategy=strategy,
        broker=broker,
        data_source=data_source
    )

    # 运行回测
    result = engine.run(
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date
    )

    # 计算指标
    print("\n📊 计算指标...")
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(result)

    # 保存结果
    from agent.runner import save_replay_output
    from utils.hash import compute_config_hash
    import hashlib

    # 生成 experiment_id
    config_for_hash = {
        "strategy": strategy_name,
        "params": strategy_params,
        "symbol": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }
    config_hash = compute_config_hash(config_for_hash)
    exp_id = f"exp_{config_hash.split(':')[1][:8]}"

    # 保存到 runs/ 目录
    output_dir = Path("runs") / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)

    save_replay_output(result, metrics, output_dir)

    print(f"\n✅ 回测完成！")
    print(f"   实验ID: {exp_id}")
    print(f"   输出目录: {output_dir}")
    print(f"\n📈 性能指标:")
    print(f"   总收益: {metrics['total_return']*100:.2f}%")
    print(f"   最终权益: ${metrics['final_equity']:,.2f}")
    print(f"   交易次数: {metrics['trade_count']}")
    print(f"   数据点数: {len(result['steps'])}")

    # 检查窗口数量
    print(f"\n🔍 窗口检查:")
    from analysis.window_scanner import WindowScanner
    scanner = WindowScanner()

    for window_len in ["5d", "20d", "60d"]:
        windows = scanner.scan_replay_replay(result, window_len)
        print(f"   {window_len}: {len(windows)} 个窗口")

    return exp_id, result, metrics


def main():
    """主函数：生成多个长期回测"""
    print("\n" + "=" * 70)
    print("批量生成长期回测数据")
    print("=" * 70)

    # 配置多个回测
    backtests = [
        {
            "symbol": "AAPL",
            "days": 120,
            "strategy_name": "ma",
            "strategy_params": {"fast_period": 5, "slow_period": 20}
        },
        {
            "symbol": "MSFT",
            "days": 120,
            "strategy_name": "ma",
            "strategy_params": {"fast_period": 10, "slow_period": 30}
        },
        {
            "symbol": "AAPL",
            "days": 180,
            "strategy_name": "breakout",
            "strategy_params": {"period": 20, "threshold": 0.01}
        },
        {
            "symbol": "MSFT",
            "days": 180,
            "strategy_name": "breakout",
            "strategy_params": {"period": 20, "threshold": 0.01}
        }
    ]

    results = []

    for i, config in enumerate(backtests, 1):
        print(f"\n\n{'='*70}")
        print(f"回测 {i}/{len(backtests)}")
        print('='*70)

        try:
            exp_id, result, metrics = run_long_backtest(**config)
            results.append({
                "exp_id": exp_id,
                "config": config,
                "status": "success",
                "total_return": metrics['total_return'],
                "num_steps": len(result['steps'])
            })
        except Exception as e:
            print(f"\n❌ 回测失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "config": config,
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
                print(f"   - {r['exp_id']}: {r['config']['strategy_name']} "
                      f"{r['config']['symbol']} {r['num_steps']}步 "
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
    main()
