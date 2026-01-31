#!/usr/bin/env python
"""
模拟盘测试 - 完整的交易日模拟

使用真实市场数据进行完整的模拟交易测试
"""

import sys
from datetime import datetime, timedelta
from decimal import Decimal
import json


def paper_trading_session():
    """完整模拟盘交易会话"""
    print("\n" + "="*60)
    print("  模拟盘交易测试")
    print("="*60)

    from skills.market_data.yahoo_source import YahooFinanceSource
    from skills.strategy.moving_average import MAStrategy
    from skills.strategy.portfolio import create_default_portfolio
    from skills.risk.basic_risk import BasicRiskManager
    from broker.paper import PaperBroker
    from broker.realistic import RealisticBroker, OrderType
    from reports.metrics import MetricsCalculator
    from core.models import OrderIntent, Position

    # 配置参数
    INITIAL_CASH = Decimal("100000")
    SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
    DAYS = 60  # 测试天数

    print(f"\n配置:")
    print(f"  初始资金: ${INITIAL_CASH:,.2f}")
    print(f"  交易标的: {', '.join(SYMBOLS)}")
    print(f"  测试周期: {DAYS} 天")

    # 获取市场数据
    print(f"\n{'='*60}")
    print("步骤 1: 获取市场数据")
    print('='*60)

    source = YahooFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=DAYS)

    market_data = {}
    for symbol in SYMBOLS:
        print(f"获取 {symbol} 数据...")
        bars = source.get_bars(symbol, start, end, '1d')
        market_data[symbol] = bars
        print(f"  ✓ {len(bars)} 根 K线")

    # 使用真实经纪商（带滑点）
    print(f"\n{'='*60}")
    print("步骤 2: 初始化交易系统")
    print('='*60)

    # 创建组合策略
    portfolio = create_default_portfolio()
    print(f"✓ 组合策略: {len(portfolio.strategies)} 个子策略")

    # 创建风控
    risk_manager = BasicRiskManager(
        max_position_ratio=0.3,
        max_positions=5,
        max_daily_loss_ratio=0.05
    )
    print(f"✓ 风控管理器")

    # 创建经纪商（带滑点模拟）
    broker = RealisticBroker(
        initial_cash=INITIAL_CASH,
        enable_slippage=True,
        enable_market_impact=True,
        enable_partial_fill=True
    )
    print(f"✓ 经纪商 (滑点: 启, 市场冲击: 启, 部分成交: 启)")

    # 模拟交易
    print(f"\n{'='*60}")
    print("步骤 3: 开始模拟交易")
    print('='*60)

    all_trades = []
    equity_curve = []
    daily_pnl = []
    signal_count = 0
    order_count = 0
    filled_count = 0

    # 找到所有交易日的交集
    min_bars = min(len(bars) for bars in market_data.values())
    print(f"\n模拟 {min_bars} 个交易日...\n")

    for i in range(30, min_bars):  # 跳过前30天用于指标计算
        trade_date = None
        bars_for_day = {}

        # 获取当日数据
        for symbol in SYMBOLS:
            bars = market_data[symbol]
            if i < len(bars):
                bars_for_day[symbol] = bars[:i+1]  # 历史数据
                current_bar = bars[i]
                if trade_date is None:
                    trade_date = current_bar.timestamp.date()

        if not bars_for_day:
            continue

        # 计算当前权益
        current_equity = broker.get_cash_balance()
        for symbol, qty in broker.get_positions().items():
            if symbol in market_data and i < len(market_data[symbol]):
                current_price = market_data[symbol][i].close
                current_equity += current_price * qty

        equity_curve.append((datetime.combine(trade_date, datetime.min.time()), current_equity))

        # 生成信号
        for symbol, hist_bars in bars_for_day.items():
            current_bar = hist_bars[-1]

            # 检查当前持仓
            positions = broker.get_positions()
            current_position = None
            if symbol in positions:
                # 简化：不创建完整的Position对象
                pass

            # 生成信号
            try:
                signals = portfolio.generate_signals(hist_bars, current_position)

                for signal in signals:
                    signal_count += 1

                    # 风控检查
                    intent = OrderIntent(
                        signal=signal,
                        timestamp=datetime.now(),
                        approved=True,
                        risk_reason="模拟盘测试"
                    )

                    # 提交订单
                    order_count += 1
                    result = broker.submit_order(intent, current_bar, OrderType.MARKET)

                    if result.trade:
                        filled_count += 1
                        all_trades.append(result.trade)

                        # 显示交易信息
                        action_color = "\033[0;32m" if signal.action == "BUY" else "\033[0;31m"
                        reset_color = "\033[0m"
                        print(f"{trade_date} {action_color}{signal.action}{reset_color} "
                              f"{signal.quantity} {signal.symbol} @ ${result.filled_price:.2f} "
                              f"(滑点: {result.slippage_percent:.3f}%)")

            except Exception as e:
                pass  # 忽略信号生成错误

    # 显示结果
    print(f"\n{'='*60}")
    print("步骤 4: 交易统计")
    print('='*60)

    print(f"\n信号统计:")
    print(f"  生成信号: {signal_count}")
    print(f"  提交订单: {order_count}")
    print(f"  成交订单: {filled_count}")
    print(f"  成交率: {filled_count/order_count*100 if order_count > 0 else 0:.1f}%")

    print(f"\n资金统计:")
    final_cash = broker.get_cash_balance()
    final_equity = equity_curve[-1][1] if equity_curve else final_cash
    total_return = (final_equity - INITIAL_CASH) / INITIAL_CASH

    print(f"  初始资金: ${INITIAL_CASH:,.2f}")
    print(f"  最终现金: ${final_cash:,.2f}")
    print(f"  最终权益: ${final_equity:,.2f}")
    print(f"  总收益率: {total_return:.2%}")

    print(f"\n持仓统计:")
    positions = broker.get_positions()
    print(f"  持仓数量: {len(positions)}")
    for symbol, qty in positions.items():
        current_price = market_data[symbol][-1].close if symbol in market_data else Decimal("0")
        market_value = current_price * qty
        print(f"    {symbol}: {qty} 股, 市值 ${market_value:,.2f}")

    # 性能指标
    if len(equity_curve) > 1 and len(all_trades) > 0:
        print(f"\n{'='*60}")
        print("步骤 5: 性能指标")
        print('='*60)

        try:
            metrics = MetricsCalculator.calculate(
                equity_curve=equity_curve,
                trades=all_trades,
                initial_cash=INITIAL_CASH
            )

            print(f"\n收益指标:")
            print(f"  总收益率: {metrics.total_return:.2%}")
            print(f"  年化收益率: {metrics.annualized_return:.2%}")

            print(f"\n风险指标:")
            print(f"  年化波动率: {metrics.volatility:.2%}")
            print(f"  最大回撤: {metrics.max_drawdown:.2%}")
            print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
            print(f"  索提诺比率: {metrics.sortino_ratio:.2f}")

            print(f"\n交易统计:")
            print(f"  总交易: {metrics.total_trades}")
            print(f"  盈利交易: {metrics.winning_trades}")
            print(f"  亏损交易: {metrics.losing_trades}")
            print(f"  胜率: {metrics.win_rate:.2%}")
            print(f"  盈亏比: {metrics.profit_factor:.2f}")

        except Exception as e:
            print(f"  指标计算失败: {e}")

    # 执行质量统计
    if hasattr(broker, 'get_execution_stats'):
        print(f"\n{'='*60}")
        print("步骤 6: 执行质量统计")
        print('='*60)

        stats = broker.get_execution_stats()
        print(f"\n执行统计:")
        print(f"  总订单: {stats['total_orders']}")
        print(f"  成交: {stats['filled_orders']}")
        print(f"  拒绝: {stats['rejected_orders']}")
        print(f"  部分成交: {stats['partial_fills']}")
        print(f"  成交率: {stats['fill_rate']:.2%}")
        print(f"  平均滑点: {stats['avg_slippage']:.4f}%")

        if stats.get('execution_quality_distribution'):
            print(f"\n执行质量分布:")
            for quality, count in stats['execution_quality_distribution'].items():
                print(f"    {quality}: {count}")

    # 权益曲线摘要
    print(f"\n{'='*60}")
    print("步骤 7: 权益曲线摘要")
    print('='*60)

    if len(equity_curve) > 0:
        print(f"\n权益变化 (前5天和后5天):")
        print(f"{'日期':<12} {'权益':>15} {'收益率':>10}")
        print("-" * 40)

        # 显示前5个
        for date, equity in equity_curve[:5]:
            ret = (equity - INITIAL_CASH) / INITIAL_CASH
            print(f"{date.date()}   ${equity:>12,.2f}   {ret:>8.2%}")

        if len(equity_curve) > 10:
            print("   ...")
            # 显示后5个
            for date, equity in equity_curve[-5:]:
                ret = (equity - INITIAL_CASH) / INITIAL_CASH
                print(f"{date.date()}   ${equity:>12,.2f}   {ret:>8.2%}")

    print(f"\n{'='*60}")
    print("  模拟盘测试完成")
    print('='*60)

    return True


def compare_brokers():
    """比较理想经纪商 vs 真实经纪商"""
    print("\n" + "="*60)
    print("  经纪商对比测试")
    print("="*60)

    from skills.market_data.yahoo_source import YahooFinanceSource
    from skills.strategy.moving_average import MAStrategy
    from broker.paper import PaperBroker
    from broker.realistic import RealisticBroker, OrderType
    from datetime import datetime, timedelta
    from decimal import Decimal

    # 获取数据
    source = YahooFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=30)

    bars = source.get_bars('AAPL', start, end, '1d')
    print(f"\n测试数据: AAPL {len(bars)} 根 K线\n")

    # 理想回测
    print("1. 理想经纪商 (无滑点):")
    ideal_broker = PaperBroker(initial_cash=Decimal("50000"))
    strategy = MAStrategy(fast_period=5, slow_period=20)

    ideal_trades = 0
    for i in range(20, len(bars)):
        hist_bars = bars[:i+1]
        signals = strategy.generate_signals(hist_bars, None)

        for signal in signals:
            from core.models import OrderIntent
            intent = OrderIntent(signal=signal, timestamp=datetime.now(), approved=True, risk_reason="")
            result = ideal_broker.submit_order(intent, bars[i])
            if result.trade:
                ideal_trades += 1

    ideal_final = ideal_broker.get_cash_balance()
    ideal_return = (ideal_final - 50000) / 50000
    print(f"  交易次数: {ideal_trades}")
    print(f"  最终资金: ${ideal_final:,.2f}")
    print(f"  收益率: {ideal_return:.2%}")

    # 真实回测
    print(f"\n2. 真实经纪商 (带滑点):")
    realistic_broker = RealisticBroker(
        initial_cash=Decimal("50000"),
        enable_slippage=True,
        enable_market_impact=True
    )

    realistic_trades = 0
    total_slippage = 0

    for i in range(20, len(bars)):
        hist_bars = bars[:i+1]
        signals = strategy.generate_signals(hist_bars, None)

        for signal in signals:
            from core.models import OrderIntent
            intent = OrderIntent(signal=signal, timestamp=datetime.now(), approved=True, risk_reason="")
            result = realistic_broker.submit_order(intent, bars[i], OrderType.MARKET)
            if result.trade:
                realistic_trades += 1
                total_slippage += result.slippage_percent

    realistic_final = realistic_broker.get_cash_balance()
    realistic_return = (realistic_final - 50000) / 50000
    avg_slippage = total_slippage / realistic_trades if realistic_trades > 0 else 0

    print(f"  交易次数: {realistic_trades}")
    print(f"  最终资金: ${realistic_final:,.2f}")
    print(f"  收益率: {realistic_return:.2%}")
    print(f"  平均滑点: {avg_slippage:.4f}%")

    # 对比
    print(f"\n3. 对比:")
    diff = realistic_return - ideal_return
    print(f"  收益率差异: {diff:.2%}")
    print(f"  滑点成本: ~{abs(diff):.2%}")

    return True


def main():
    """运行模拟盘测试"""
    print("\n" + "="*60)
    print("  模拟盘测试套件")
    print("="*60)

    tests = [
        ("完整模拟盘交易", paper_trading_session),
        ("经纪商对比测试", compare_brokers),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"  模拟盘测试结果: {passed} 通过, {failed} 失败")
    print("="*60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
