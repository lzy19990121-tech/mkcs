#!/usr/bin/env python
"""
活跃的模拟盘演示 - 展示实际的交易活动
"""

from skills.market_data.yahoo_source import YahooFinanceSource
from skills.strategy.moving_average import MAStrategy
from broker.realistic import RealisticBroker, OrderType
from core.models import Signal, OrderIntent
from decimal import Decimal
from datetime import datetime, timedelta

print("\n" + "="*60)
print("  模拟盘演示 - 真实交易场景")
print("="*60)

# 获取更多数据
print("\n步骤 1: 获取市场数据")
source = YahooFinanceSource()
end = datetime.now()
start = end - timedelta(days=180)  # 6个月

bars = source.get_bars('AAPL', start, end, '1d')
print(f"✓ 获取 {len(bars)} 根 AAPL K线 (6个月)")
print(f"  时间范围: {bars[0].timestamp.date()} 到 {bars[-1].timestamp.date()}")
print(f"  价格范围: ${min(b.close for b in bars):.2f} - ${max(b.close for b in bars):.2f}")

# 使用多种策略参数
print("\n步骤 2: 测试不同策略参数")

strategy_configs = [
    ("短期", 5, 10),
    ("中期", 10, 20),
    ("长期", 20, 50),
]

all_signals = []

for name, fast, slow in strategy_configs:
    strategy = MAStrategy(fast_period=fast, slow_period=slow)
    signals_count = 0

    for i in range(max(fast, slow), len(bars)):
        hist_bars = bars[:i+1]
        signals = strategy.generate_signals(hist_bars, None)
        if signals:
            signals_count += len(signals)
            all_signals.extend([(name, bars[i].timestamp.date(), s) for s in signals])

    print(f"  {name}策略 (MA{fast}/{slow}): {signals_count} 个信号")

print(f"\n✓ 总共生成 {len(all_signals)} 个信号")

# 创建经纪商并模拟交易
print("\n步骤 3: 模拟交易执行")
broker = RealisticBroker(
    initial_cash=Decimal("100000"),
    enable_slippage=True,
    enable_market_impact=True
)

# 模拟最后60天的交易
print(f"\n模拟最后60个交易日的执行:")
print("-" * 60)

trades_executed = 0
total_slippage = 0

for i in range(max(20, len(bars) - 60), len(bars)):
    current_bar = bars[i]
    date_str = current_bar.timestamp.date()

    # 获取历史K线
    hist_bars = bars[:i+1]

    # 使用短期策略生成信号
    strategy = MAStrategy(fast_period=5, slow_period=15)
    signals = strategy.generate_signals(hist_bars, None)

    for signal in signals:
        # 检查是否可以执行
        if signal.action == "SELL":
            positions = broker.get_positions()
            if signal.symbol not in positions or positions[signal.symbol] < signal.quantity:
                continue  # 没有足够持仓

        # 创建订单意图
        intent = OrderIntent(
            signal=signal,
            timestamp=datetime.now(),
            approved=True,
            risk_reason="演示"
        )

        # 执行订单
        result = broker.submit_order(intent, current_bar, OrderType.MARKET)

        if result.trade:
            trades_executed += 1
            total_slippage += result.slippage_percent

            action_color = "\033[0;32mBUY \033[0m" if signal.action == "BUY" else "\033[0;31mSELL\033[0m"
            print(f"{date_str} {action_color} {signal.quantity:3d} {signal.symbol} @ "
                  f"${result.filled_price:7.2f} (滑点: {result.slippage_percent:6.3f}%, "
                  f"质量: {result.execution_quality.value})")

print(f"\n✓ 执行了 {trades_executed} 笔交易")
if trades_executed > 0:
    print(f"  平均滑点: {total_slippage / trades_executed:.4f}%")

# 显示最终状态
print("\n步骤 4: 最终状态")
print("-" * 60)

final_cash = broker.get_cash_balance()
positions = broker.get_positions()

print(f"初始资金: $100,000.00")
print(f"最终现金: ${final_cash:,.2f}")

if positions:
    print(f"\n持仓:")
    total_market_value = Decimal("0")
    for symbol, qty in positions.items():
        # 获取最新价格
        latest_price = bars[-1].close
        market_value = latest_price * qty
        total_market_value += market_value
        print(f"  {symbol}: {qty} 股 @ ${latest_price:.2f} = ${market_value:,.2f}")

    final_equity = final_cash + total_market_value
    total_return = (final_equity - 100000) / 100000

    print(f"\n最终权益: ${final_equity:,.2f}")
    print(f"总收益率: {total_return:.2%}")
else:
    print("无持仓")
    total_return = (final_cash - 100000) / 100000
    print(f"总收益率: {total_return:.2%}")

# 显示执行统计
print("\n步骤 5: 执行质量统计")
print("-" * 60)

stats = broker.get_execution_stats()
print(f"总订单: {stats['total_orders']}")
print(f"成交: {stats['filled_orders']}")
print(f"拒绝: {stats['rejected_orders']}")
print(f"部分成交: {stats['partial_fills']}")
print(f"成交率: {stats['fill_rate']:.2%}")
print(f"平均滑点: {stats['avg_slippage']:.4f}%")

if stats.get('execution_quality_distribution'):
    print(f"\n执行质量分布:")
    for quality, count in sorted(stats['execution_quality_distribution'].items()):
        print(f"  {quality:12s}: {count}")

print("\n" + "="*60)
print("  模拟盘演示完成")
print("="*60)
