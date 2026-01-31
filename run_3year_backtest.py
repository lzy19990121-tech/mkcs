#!/usr/bin/env python3
"""
3年历史数据回测

使用 Yahoo Finance 数据源，回测2022-2024年数据
"""

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
import json
import csv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_3year_backtest():
    """运行3年历史数据回测"""

    print("=" * 60)
    print("3年历史数据回测 (2022-2024)")
    print("=" * 60)

    # 导入组件
    from skills.market_data.yahoo_source import YahooFinanceSource
    from skills.market_data.mock_source import MockMarketSource
    from skills.strategy.moving_average import MAStrategy
    from skills.risk.basic_risk import BasicRiskManager
    from broker.paper import PaperBroker
    from agent.runner import TradingAgent, create_default_agent
    from agent.replay_engine import ReplayEngine
    from reports.metrics import MetricsCalculator, MetricsReport

    # 设置回测参数
    end_date = date(2024, 12, 31)
    start_date = date(2022, 1, 1)
    initial_cash = Decimal("100000")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    print(f"\n📅 回测区间: {start_date} ~ {end_date}")
    print(f"💰 初始资金: ${initial_cash:,.2f}")
    print(f"📈 交易标的: {', '.join(symbols)}")

    # 创建数据源
    print("\n🔌 初始化数据源...")
    use_yahoo = False
    try:
        yahoo_source = YahooFinanceSource(enable_cache=True)
        # 测试连接
        test_end = datetime(2024, 12, 1)
        test_start = test_end - timedelta(days=10)
        test_bars = yahoo_source.get_bars("AAPL", test_start, test_end, "1d")
        if len(test_bars) > 0:
            print(f"   ✓ Yahoo Finance 连接成功 (测试获取 {len(test_bars)} 条数据)")
            data_source = yahoo_source
            use_yahoo = True
        else:
            print("   ⚠ Yahoo Finance 无数据，使用 Mock 数据源")
            data_source = MockMarketSource(seed=42)
    except Exception as e:
        print(f"   ⚠ Yahoo Finance 连接失败: {e}")
        print("   使用 Mock 数据源")
        data_source = MockMarketSource(seed=42)

    # 创建其他组件
    print("\n⚙️ 初始化组件...")
    strategy = MAStrategy(fast_period=5, slow_period=20)
    risk_manager = BasicRiskManager()
    broker = PaperBroker(initial_cash=initial_cash)
    print("   ✓ 策略: MA交叉 (5日/20日)")
    print("   ✓ 风控: 基础风控规则")
    print("   ✓ 经纪商: PaperBroker")

    # 创建 Agent
    agent = TradingAgent(
        data_source=data_source,
        strategy=strategy,
        risk_manager=risk_manager,
        broker=broker,
        db=None
    )
    print("   ✓ TradingAgent 初始化完成")

    # 创建回放引擎
    replay = ReplayEngine(start=start_date, end=end_date, interval="1d", market="US")

    # 运行回测
    print("\n🚀 开始回测...")
    print("-" * 60)

    output_dir = Path("reports/3year_backtest")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 权益曲线
    equity_curve = []
    all_risk_rejects = []

    total_days = 0
    total_signals = 0
    total_orders = 0
    total_fills = 0

    try:
        for point in replay.iter_days():
            ctx = point.ctx
            total_days += 1

            # 执行 tick
            result = agent.tick(ctx, symbols)

            total_signals += result['signals_generated']
            total_orders += result['orders_submitted']
            total_fills += result['orders_filled']
            all_risk_rejects.extend(result['risk_rejects'])

            # 记录权益
            portfolio_value = broker.get_total_equity()
            cash = broker.get_cash_balance()
            equity_curve.append({
                'date': ctx.trading_date.isoformat(),
                'equity': float(portfolio_value),
                'cash': float(cash)
            })

            if total_days % 100 == 0:
                print(f"   已处理 {total_days} 个交易日，当前权益: ${portfolio_value:,.2f}")

        print("\n" + "=" * 60)
        print("回测完成!")
        print("=" * 60)

        # 获取最终结果
        final_equity = broker.get_total_equity()
        total_return = (final_equity - initial_cash) / initial_cash
        trades = broker.get_trades()

        print(f"\n📊 回测结果:")
        print(f"   回测天数: {total_days}")
        print(f"   初始资金: ${initial_cash:,.2f}")
        print(f"   最终权益: ${final_equity:,.2f}")
        print(f"   总收益率: {total_return*100:.2f}%")
        print(f"   总交易次数: {len(trades)}")
        print(f"   生成信号: {total_signals}")
        print(f"   提交订单: {total_orders}")
        print(f"   成交订单: {total_fills}")
        print(f"   风控拒绝: {len(all_risk_rejects)}")

        # 持仓情况
        positions = broker.get_positions()
        if positions:
            print(f"\n📈 持仓情况:")
            for symbol, pos in positions.items():
                print(f"   {symbol}: {pos.quantity} 股 @ ${pos.avg_price:.2f}")
        else:
            print(f"\n📈 持仓情况: 无持仓")

        # 保存权益曲线
        equity_path = output_dir / "equity_curve.csv"
        with open(equity_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "equity", "cash"])
            writer.writeheader()
            writer.writerows(equity_curve)
        print(f"\n📄 权益曲线已保存: {equity_path}")

        # 保存交易记录
        if trades:
            trades_path = output_dir / "trades.csv"
            with open(trades_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "symbol", "side", "price", "quantity", "commission", "pnl"])
                for trade in trades:
                    writer.writerow([
                        trade.timestamp.isoformat(),
                        trade.symbol,
                        trade.side,
                        float(trade.price),
                        trade.quantity,
                        float(trade.commission),
                        float(trade.realized_pnl) if hasattr(trade, 'realized_pnl') else 0
                    ])
            print(f"📄 交易记录已保存: {trades_path}")

        # 保存风控拒绝记录
        if all_risk_rejects:
            rejects_path = output_dir / "risk_rejects.csv"
            with open(rejects_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "symbol", "action", "reason", "confidence"])
                for reject in all_risk_rejects:
                    writer.writerow([
                        reject.get('timestamp', ''),
                        reject.get('symbol', ''),
                        reject.get('action', ''),
                        reject.get('reason', ''),
                        reject.get('confidence', '')
                    ])
            print(f"📄 风控拒绝记录已保存: {rejects_path}")

        # 计算性能指标
        print("\n📊 计算性能指标...")

        if len(equity_curve) > 1:
            # 转换权益曲线格式
            equity_curve_tuples = [
                (datetime.fromisoformat(e['date']), Decimal(str(e['equity'])))
                for e in equity_curve
            ]

            metrics = MetricsCalculator.calculate(
                equity_curve=equity_curve_tuples,
                trades=trades,
                initial_cash=initial_cash
            )

            print("\n📈 性能指标:")
            print(f"   年化收益率: {metrics.annualized_return*100:.2f}%")
            print(f"   年化波动率: {metrics.volatility*100:.2f}%")
            print(f"   夏普比率: {metrics.sharpe_ratio:.2f}")
            print(f"   索提诺比率: {metrics.sortino_ratio:.2f}")
            print(f"   最大回撤: {metrics.max_drawdown*100:.2f}%")
            print(f"   最大回撤持续时间: {metrics.max_drawdown_duration} 天")
            print(f"   胜率: {metrics.win_rate*100:.1f}%")
            print(f"   盈亏比: {metrics.profit_factor:.2f}")
            print(f"   平均盈利: ${metrics.average_profit:.2f}")
            print(f"   平均亏损: ${metrics.average_loss:.2f}")

            # 生成详细报告
            report = MetricsReport.generate_report(metrics, "3年历史回测报告 (2022-2024)")
            report_path = output_dir / "metrics_report.md"
            with open(report_path, "w") as f:
                f.write(report)
            print(f"\n📄 详细报告已保存: {report_path}")

            # 保存指标JSON
            metrics_dict = metrics.to_dict()
            metrics_json_path = output_dir / "metrics.json"
            with open(metrics_json_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            print(f"📄 指标JSON已保存: {metrics_json_path}")

        # 保存结果摘要
        summary = {
            "backtest_date": datetime.now().isoformat(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "symbols": symbols,
            "initial_cash": float(initial_cash),
            "final_equity": float(final_equity),
            "total_return": float(total_return),
            "total_trades": len(trades),
            "total_signals": total_signals,
            "total_orders": total_orders,
            "total_fills": total_fills,
            "risk_rejects": len(all_risk_rejects),
            "trading_days": total_days,
            "data_source": "YahooFinance" if use_yahoo else "Mock",
            "strategy": "MA_Crossover_5_20",
            "risk_manager": "BasicRiskManager"
        }

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📄 结果摘要已保存: {summary_path}")

        print("\n" + "=" * 60)
        print("✅ 3年回测全部完成!")
        print("=" * 60)

        return summary

    except Exception as e:
        logger.exception("回测失败")
        print(f"\n❌ 回测失败: {e}")
        raise


if __name__ == "__main__":
    run_3year_backtest()
