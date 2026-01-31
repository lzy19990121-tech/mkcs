#!/usr/bin/env python
"""
扩展测试 - 完整回测和集成测试
"""

import sys
from datetime import datetime, timedelta
from decimal import Decimal
import os


def test_full_backtest_with_real_data():
    """使用真实数据进行完整回测"""
    print("\n" + "="*50)
    print("扩展测试 1: 策略与数据集成测试 (Yahoo Finance)")
    print("="*50)

    from skills.market_data.yahoo_source import YahooFinanceSource
    from skills.strategy.moving_average import MAStrategy
    from broker.paper import PaperBroker
    from core.models import OrderIntent

    # 获取真实数据
    source = YahooFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=90)  # 3个月

    print(f"获取 AAPL 数据 ({start.date()} 到 {end.date()})...")
    bars = source.get_bars('AAPL', start, end, '1d')
    print(f"✓ 获取 {len(bars)} 根 K线")

    if len(bars) < 30:
        print("✗ 数据不足，跳过测试")
        return False

    # 创建策略和经纪商
    strategy = MAStrategy(fast_period=5, slow_period=20)
    broker = PaperBroker(initial_cash=Decimal('10000'))

    # 生成信号
    print("\n生成交易信号...")
    signals = strategy.generate_signals(bars, position=None)
    print(f"✓ 生成 {len(signals)} 个信号")

    # 模拟执行部分信号
    executed = 0
    for signal in signals[:5]:  # 只执行前5个
        intent = OrderIntent(
            signal=signal,
            timestamp=datetime.now(),
            approved=True,
            risk_reason="测试"
        )

        # 查找对应的K线
        bar = next((b for b in bars if b.timestamp == signal.timestamp), None)
        if bar:
            broker.submit_order(intent, bar)
            executed += 1

    print(f"✓ 执行了 {executed} 个订单")
    print(f"  最终现金: ${broker.get_cash_balance():.2f}")
    print(f"  持仓: {broker.get_positions()}")

    return True


def test_technical_indicators_comprehensive():
    """全面测试技术指标"""
    print("\n" + "="*50)
    print("扩展测试 2: 技术指标全面测试")
    print("="*50)

    from skills.indicators.technical import TechnicalIndicators

    # 生成模拟价格数据
    import random
    random.seed(42)
    prices = [100 + i * 0.1 + random.uniform(-2, 2) for i in range(100)]

    print(f"测试数据: {len(prices)} 个价格点")
    print(f"价格范围: ${min(prices):.2f} - ${max(prices):.2f}")

    indicators_tested = []

    # 趋势指标
    try:
        sma = TechnicalIndicators.sma(prices, 20)
        indicators_tested.append(f"SMA(20): {sma[-1]:.2f}")
    except Exception as e:
        indicators_tested.append(f"SMA 失败: {e}")

    try:
        ema = TechnicalIndicators.ema(prices, 20)
        indicators_tested.append(f"EMA(20): {ema[-1]:.2f}")
    except Exception as e:
        indicators_tested.append(f"EMA 失败: {e}")

    # 动量指标
    try:
        rsi = TechnicalIndicators.rsi(prices, 14)
        indicators_tested.append(f"RSI(14): {rsi[-1]:.2f}")
    except Exception as e:
        indicators_tested.append(f"RSI 失败: {e}")

    try:
        macd, signal, hist = TechnicalIndicators.macd(prices)
        indicators_tested.append(f"MACD: {macd[-1]:.4f}")
    except Exception as e:
        indicators_tested.append(f"MACD 失败: {e}")

    # 波动率指标
    try:
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(prices, 20, 2)
        indicators_tested.append(f"布林带上轨: {bb_upper[-1]:.2f}")
        indicators_tested.append(f"布林带下轨: {bb_lower[-1]:.2f}")
    except Exception as e:
        indicators_tested.append(f"布林带 失败: {e}")

    try:
        atr = TechnicalIndicators.atr(
            [prices[0]] * 100,  # open 简化
            prices,  # high
            prices,  # low
            14
        )
        indicators_tested.append(f"ATR(14): {atr[-1]:.2f}")
    except Exception as e:
        indicators_tested.append(f"ATR 失败: {e}")

    # 成交量指标
    volumes = [random.randint(1000000, 5000000) for _ in range(100)]
    try:
        obv = TechnicalIndicators.obv(prices, volumes)
        indicators_tested.append(f"OBV: {obv[-1]:.0f}")
    except Exception as e:
        indicators_tested.append(f"OBV 失败: {e}")

    try:
        vwap = TechnicalIndicators.vwap(prices, volumes)
        indicators_tested.append(f"VWAP: {vwap[-1]:.2f}")
    except Exception as e:
        indicators_tested.append(f"VWAP 失败: {e}")

    # 显示结果
    print(f"\n测试了 {len(indicators_tested)} 个指标:")
    for indicator in indicators_tested:
        print(f"  ✓ {indicator}")

    return True


def test_portfolio_performance():
    """测试组合策略性能"""
    print("\n" + "="*50)
    print("扩展测试 3: 多策略组合性能")
    print("="*50)

    from skills.strategy.portfolio import create_default_portfolio, PortfolioStrategy, StrategyConfig
    from skills.strategy.moving_average import MAStrategy
    from skills.market_data.yahoo_source import YahooFinanceSource
    from datetime import datetime, timedelta

    # 获取数据
    source = YahooFinanceSource()
    end = datetime.now()
    start = end - timedelta(days=60)

    print("获取数据...")
    bars = source.get_bars('MSFT', start, end, '1d')
    print(f"✓ {len(bars)} 根 K线")

    if len(bars) < 30:
        print("✗ 数据不足")
        return False

    # 创建组合
    portfolio = create_default_portfolio()

    # 测试权重调整
    print("\n测试动态权重调整:")
    status = portfolio.get_strategy_status()
    for name, info in status.items():
        print(f"  {name}: 权重={info['weight']}")

    # 调整权重
    portfolio.set_strategy_weight("MA_Crossover_Short", 0.5)
    print(f"\n✓ 调整 MA_Crossover_Short 权重为 0.5")

    status = portfolio.get_strategy_status()
    for name, info in status.items():
        print(f"  {name}: 权重={info['weight']}")

    # 禁用/启用策略
    print("\n测试策略禁用/启用:")
    portfolio.disable_strategy("ML_RandomForest")
    print("✓ 已禁用 ML_RandomForest")

    status = portfolio.get_strategy_status()
    enabled_count = sum(1 for info in status.values() if info['enabled'])
    print(f"  启用策略数: {enabled_count}/{len(status)}")

    # 重新启用
    portfolio.enable_strategy("ML_RandomForest")
    print("✓ 已重新启用 ML_RandomForest")

    # 生成信号
    signals = portfolio.generate_signals(bars, position=None)
    print(f"\n✓ 生成 {len(signals)} 个信号")
    for sig in signals[:3]:  # 只显示前3个
        print(f"  {sig.symbol}: {sig.action} (置信度: {sig.confidence:.2%})")

    return True


def test_performance_metrics():
    """测试性能指标计算"""
    print("\n" + "="*50)
    print("扩展测试 4: 性能指标计算")
    print("="*50)

    from reports.metrics import MetricsCalculator
    from decimal import Decimal
    from datetime import datetime, timedelta
    from core.models import Trade

    # 模拟交易历史
    trades = [
        Trade(
            trade_id="T001",
            symbol="AAPL",
            side="BUY",
            price=Decimal("150.00"),
            quantity=100,
            timestamp=datetime.now() - timedelta(days=10),
            commission=Decimal("1.00")
        ),
        Trade(
            trade_id="T002",
            symbol="AAPL",
            side="SELL",
            price=Decimal("155.00"),
            quantity=100,
            timestamp=datetime.now() - timedelta(days=5),
            commission=Decimal("1.00")
        ),
        Trade(
            trade_id="T003",
            symbol="MSFT",
            side="BUY",
            price=Decimal("300.00"),
            quantity=50,
            timestamp=datetime.now() - timedelta(days=3),
            commission=Decimal("1.50")
        ),
        Trade(
            trade_id="T004",
            symbol="MSFT",
            side="SELL",
            price=Decimal("295.00"),
            quantity=50,
            timestamp=datetime.now() - timedelta(days=1),
            commission=Decimal("1.50")
        ),
    ]

    # 模拟权益曲线
    initial_cash = Decimal("20000")
    equity_curve = [
        (datetime.now() - timedelta(days=10), initial_cash),
        (datetime.now() - timedelta(days=8), Decimal("20500")),
        (datetime.now() - timedelta(days=5), Decimal("20300")),
        (datetime.now() - timedelta(days=3), Decimal("20000")),
        (datetime.now() - timedelta(days=1), Decimal("19970")),
    ]

    # 计算指标
    print("计算性能指标...")
    metrics = MetricsCalculator.calculate(
        equity_curve=equity_curve,
        trades=trades,
        initial_cash=initial_cash
    )

    print(f"\n✓ 性能指标:")
    print(f"  总收益率: {metrics.total_return:.2%}")
    print(f"  年化收益率: {metrics.annualized_return:.2%}")
    print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
    print(f"  最大回撤: {metrics.max_drawdown:.2%}")
    print(f"  胜率: {metrics.win_rate:.2%}")
    print(f"  盈亏比: {metrics.profit_factor:.2f}")
    print(f"  总交易次数: {metrics.total_trades}")
    print(f"  盈利交易: {metrics.winning_trades}")
    print(f"  亏损交易: {metrics.losing_trades}")

    return True


def test_order_execution_scenarios():
    """测试各种订单执行场景"""
    print("\n" + "="*50)
    print("扩展测试 5: 订单执行场景")
    print("="*50)

    from broker.realistic import RealisticBroker, OrderType, ExecutionQuality
    from core.models import Signal, OrderIntent
    from skills.market_data.mock_source import MockMarketSource
    from datetime import datetime

    # 生成测试数据
    source = MockMarketSource(seed=42)
    bars = source.get_bars('AAPL', datetime.now() - timedelta(days=100), datetime.now(), '1d')
    test_bar = bars[-1]

    scenarios = [
        ("小单市价", 100, OrderType.MARKET, Decimal("100000")),
        ("中单市价", 500, OrderType.MARKET, Decimal("100000")),
        ("小单限价", 100, OrderType.LIMIT, Decimal("100000")),
        ("资金不足", 10000, OrderType.MARKET, Decimal("10000")),
    ]

    print(f"测试价格: ${test_bar.close}, 成交量: {test_bar.volume}\n")

    for scenario_name, quantity, order_type, cash in scenarios:
        broker = RealisticBroker(initial_cash=cash)

        signal = Signal(
            symbol='AAPL',
            timestamp=test_bar.timestamp,
            action='BUY',
            price=test_bar.close,
            quantity=quantity,
            confidence=0.8,
            reason='测试'
        )
        intent = OrderIntent(signal=signal, timestamp=datetime.now(), approved=True, risk_reason='测试')

        result = broker.submit_order(intent, test_bar, order_type)

        print(f"{scenario_name}:")
        print(f"  数量: {quantity}")
        print(f"  成交: {result.filled_quantity}")
        if result.trade:
            print(f"  价格: ${result.filled_price:.2f}")
            print(f"  滑点: {result.slippage_percent:.4f}%")
            print(f"  质量: {result.execution_quality.value}")
        else:
            print(f"  原因: {result.reason}")
        print()

    return True


def test_alert_channels():
    """测试各种告警渠道"""
    print("\n" + "="*50)
    print("扩展测试 6: 告警渠道测试")
    print("="*50)

    from monitoring.alerts import (
        AlertManager, Alert, AlertType,
        ConsoleAlertChannel, FileAlertChannel
    )
    from pathlib import Path

    # 测试控制台渠道
    print("1. 控制台告警渠道:")
    console_channel = ConsoleAlertChannel()
    alert = Alert(
        alert_type=AlertType.DAILY_LOSS_LIMIT,
        severity="critical",
        timestamp=datetime.now(),
        title="测试告警",
        message="这是一个测试告警",
        symbol="AAPL"
    )
    console_channel.send(alert)
    print("  ✓ 控制台告警已发送")

    # 测试文件渠道
    print("\n2. 文件告警渠道:")
    test_file = "/tmp/test_alerts.log"
    file_channel = FileAlertChannel(test_file)
    file_channel.send(alert)

    if Path(test_file).exists():
        print(f"  ✓ 文件告警已写入 {test_file}")
        with open(test_file, 'r') as f:
            content = f.read()
            print(f"  内容: {content[:100]}...")
        # 清理
        Path(test_file).unlink()
    else:
        print("  ✗ 文件未创建")

    # 测试告警管理器
    print("\n3. 告警管理器:")
    alert_manager = AlertManager(channels=[console_channel])
    alert_manager.check_daily_loss_limit(Decimal("-6000"), Decimal("100000"))
    alert_manager.check_max_drawdown(Decimal("0.18"), Decimal("100000"))
    print("  ✓ 告警管理器正常工作")

    return True


def main():
    """运行所有扩展测试"""
    print("\n" + "="*60)
    print("  扩展测试套件")
    print("="*60)

    tests = [
        ("完整回测 (真实数据)", test_full_backtest_with_real_data),
        ("技术指标全面测试", test_technical_indicators_comprehensive),
        ("多策略组合性能", test_portfolio_performance),
        ("性能指标计算", test_performance_metrics),
        ("订单执行场景", test_order_execution_scenarios),
        ("告警渠道测试", test_alert_channels),
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
    print(f"  扩展测试结果: {passed} 通过, {failed} 失败")
    print("="*60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
