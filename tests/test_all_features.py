#!/usr/bin/env python
"""
测试所有新功能
"""

import sys
from datetime import datetime, timedelta
from decimal import Decimal


def test_yahoo_finance():
    """测试 Yahoo Finance 数据源"""
    print("\n" + "="*50)
    print("测试 1: Yahoo Finance 数据源")
    print("="*50)

    from skills.market_data.yahoo_source import YahooFinanceSource

    source = YahooFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=7)

    bars = source.get_bars('AAPL', start, end, '1d')
    print(f"✓ 获取 {len(bars)} 根 AAPL K线 (最近7天)")

    if bars:
        latest = bars[-1]
        print(f"✓ 最新价格: ${latest.close}, 成交量: {latest.volume}")

    quote = source.get_quote('AAPL')
    print(f"✓ 实时报价: ${quote.bid_price} / ${quote.ask_price}")

    return True


def test_technical_indicators():
    """测试技术指标库"""
    print("\n" + "="*50)
    print("测试 2: 技术指标库")
    print("="*50)

    from skills.indicators.technical import TechnicalIndicators

    prices = [130.0, 131.5, 129.8, 132.0, 133.5, 134.2, 130.5, 131.0, 132.5, 133.0]

    sma5 = TechnicalIndicators.sma(prices, 5)
    print(f"✓ SMA(5): {sma5[-1]:.2f}")

    ema12 = TechnicalIndicators.ema(prices, 12)
    print(f"✓ EMA(12): {ema12[-1]:.2f}")

    rsi = TechnicalIndicators.rsi(prices, 14)
    print(f"✓ RSI(14): {rsi[-1]:.2f}")

    macd, signal, hist = TechnicalIndicators.macd(prices)
    print(f"✓ MACD: {macd[-1]:.4f}")

    boll = TechnicalIndicators.bollinger_bands(prices, 20, 2)
    print(f"✓ 布林带: 上轨 {boll[0][-1]:.2f}, 中轨 {boll[1][-1]:.2f}")

    return True


def test_portfolio_strategy():
    """测试多策略组合"""
    print("\n" + "="*50)
    print("测试 3: 多策略组合")
    print("="*50)

    from skills.strategy.portfolio import create_default_portfolio
    from skills.market_data.mock_source import MockMarketSource

    source = MockMarketSource(seed=42)
    bars = source.get_bars('AAPL', datetime.now() - timedelta(days=100), datetime.now(), '1d')
    print(f"✓ 生成 {len(bars)} 根 K线")

    portfolio = create_default_portfolio()

    status = portfolio.get_strategy_status()
    print(f"✓ 策略组合包含 {len(status)} 个策略")
    for name, info in status.items():
        print(f"  - {name}: 权重={info['weight']}")

    signals = portfolio.generate_signals(bars, position=None)
    print(f"✓ 生成 {len(signals)} 个组合信号")
    for sig in signals:
        print(f"  {sig.symbol}: {sig.action} @ ${sig.price} (置信度: {sig.confidence:.2%})")

    return True


def test_slippage_simulation():
    """测试滑点模拟"""
    print("\n" + "="*50)
    print("测试 4: 滑点和市场冲击模拟")
    print("="*50)

    from broker.realistic import RealisticBroker, OrderType
    from core.models import Signal, OrderIntent
    from skills.market_data.mock_source import MockMarketSource

    source = MockMarketSource(seed=42)
    bars = source.get_bars('AAPL', datetime.now() - timedelta(days=100), datetime.now(), '1d')
    test_bar = bars[-1]
    print(f"✓ 当前价格: ${test_bar.close}, 成交量: {test_bar.volume}")

    broker = RealisticBroker(
        initial_cash=Decimal('100000'),
        enable_slippage=True,
        enable_market_impact=True
    )

    signal = Signal(
        symbol='AAPL',
        timestamp=test_bar.timestamp,
        action='BUY',
        price=test_bar.close,
        quantity=100,
        confidence=0.8,
        reason='测试'
    )
    intent = OrderIntent(signal=signal, timestamp=datetime.now(), approved=True, risk_reason='测试')

    result = broker.submit_order(intent, test_bar, OrderType.MARKET)
    print(f"✓ 小单执行 (100股):")
    print(f"  成交: {result.filled_quantity} @ ${result.filled_price}")
    print(f"  滑点: {result.slippage_percent:.4f}%")
    print(f"  质量: {result.execution_quality.value}")

    return True


def test_alert_system():
    """测试告警系统"""
    print("\n" + "="*50)
    print("测试 5: 监控告警系统")
    print("="*50)

    from monitoring.alerts import AlertManager, ConsoleAlertChannel

    alert_manager = AlertManager(channels=[ConsoleAlertChannel()])
    alert_manager.config['daily_loss_limit'] = Decimal('0.05')  # 5%
    alert_manager.config['position_loss_threshold'] = Decimal('0.05')  # 5%

    print("1. 测试单日亏损限制:")
    alert_manager.check_daily_loss_limit(
        daily_pnl=Decimal('-6000'),
        initial_equity=Decimal('100000')
    )

    print("\n2. 测试模型性能告警:")
    alert_manager.check_model_performance(0.45)  # 低于默认阈值 0.5

    print("\n3. 测试回撤告警:")
    alert_manager.check_max_drawdown(Decimal('0.18'), Decimal('100000'))  # 18% > 15%

    return True


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("  新功能全面测试")
    print("="*60)

    tests = [
        ("Yahoo Finance 数据源", test_yahoo_finance),
        ("技术指标库", test_technical_indicators),
        ("多策略组合", test_portfolio_strategy),
        ("滑点模拟", test_slippage_simulation),
        ("告警系统", test_alert_system),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"  测试结果: {passed} 通过, {failed} 失败")
    print("="*60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
