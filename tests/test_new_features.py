"""
新功能集成测试

测试：
1. 事件日志系统
2. 日计划生成器
3. 日复盘生成器
4. TUI界面
"""

import os
import sys
import shutil
from datetime import date, datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_event_logging():
    """测试事件日志系统"""
    from events.event_log import EventLogger, Event

    print("\n" + "="*60)
    print("测试 1: 事件日志系统")
    print("="*60)

    logger = EventLogger(log_dir="test_logs")

    # 记录测试事件
    events = [
        Event(
            ts=datetime.now(),
            symbol="AAPL",
            stage="data_fetch",
            payload={"bars": 21},
            reason="获取K线"
        ),
        Event(
            ts=datetime.now(),
            symbol="AAPL",
            stage="signal_gen",
            payload={"action": "BUY", "confidence": 0.85},
            reason="金叉买入"
        ),
    ]

    for event in events:
        logger.log(event)

    # 查询事件
    queried = logger.query_events(symbol="AAPL")
    print(f"✓ 记录 {len(events)} 个事件")
    print(f"✓ 查询到 {len(queried)} 个AAPL事件")

    logger.close()
    return True


def test_daily_planner():
    """测试日计划生成器"""
    from reports.planner import DailyPlanner
    from events.event_log import EventLogger, Event

    print("\n" + "="*60)
    print("测试 2: 日计划生成器")
    print("="*60)

    logger = EventLogger(log_dir="test_logs")

    # 添加历史事件
    yesterday = date.today() - __import__('datetime').timedelta(days=1)
    event = Event(
        ts=datetime.combine(yesterday, datetime.min.time()),
        symbol="AAPL",
        stage="signal_gen",
        payload={"action": "BUY", "confidence": 0.85},
        reason="金叉买入"
    )
    logger.log(event)

    # 生成计划
    planner = DailyPlanner(event_logger=logger)
    plan = planner.generate_plan(date.today(), ["AAPL", "GOOGL"])

    print("✓ 生成日计划成功")
    print(f"  计划长度: {len(plan)} 字符")

    logger.close()
    return True


def test_daily_reviewer():
    """测试日复盘生成器"""
    from reports.reviewer import DailyReviewer
    from events.event_log import EventLogger, Event
    from broker.paper import PaperBroker

    print("\n" + "="*60)
    print("测试 3: 日复盘生成器")
    print("="*60)

    logger = EventLogger(log_dir="test_logs")
    broker = PaperBroker(initial_cash=100000)

    # 添加测试事件
    events = [
        Event(
            ts=datetime.now(),
            symbol="AAPL",
            stage="signal_gen",
            payload={"action": "BUY", "confidence": 0.85},
            reason="金叉买入"
        ),
        Event(
            ts=datetime.now(),
            symbol="AAPL",
            stage="order_fill",
            payload={"side": "BUY", "price": 150.0, "quantity": 100},
            reason="订单执行"
        ),
    ]

    for event in events:
        logger.log(event)

    # 生成复盘
    reviewer = DailyReviewer(event_logger=logger)
    review = reviewer.generate_review(date.today(), broker)

    print("✓ 生成日复盘成功")
    print(f"  复盘长度: {len(review)} 字符")

    logger.close()
    return True


def test_tui():
    """测试TUI界面"""
    from tui.watchlist import WatchlistTUI
    from broker.paper import PaperBroker
    from decimal import Decimal

    print("\n" + "="*60)
    print("测试 4: TUI界面")
    print("="*60)

    tui = WatchlistTUI()
    broker = PaperBroker(initial_cash=100000)

    # 执行一笔交易
    from core.models import Signal, OrderIntent
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.85,
        reason="金叉买入"
    )

    intent = OrderIntent(
        signal=signal,
        timestamp=datetime.now(),
        approved=True,
        risk_reason="OK"
    )

    broker.submit_order(intent)
    from core.models import Bar
    bar = Bar(
        symbol="AAPL",
        timestamp=datetime.now() + timedelta(days=1),
        open=Decimal("150.00"),
        high=Decimal("151.00"),
        low=Decimal("149.00"),
        close=Decimal("150.50"),
        volume=100000,
        interval="1d"
    )
    broker.on_bar(bar)

    # 使用 StateCallback 替代直接 broker 访问
    from tui.watchlist import SimpleStateCallback
    state_callback = SimpleStateCallback(initial_cash=100000)
    state_callback.positions['AAPL'] = {
        'symbol': 'AAPL',
        'quantity': 100,
        'avg_price': 150.0,
        'market_value': 15050.0,
        'unrealized_pnl': 50.0
    }
    state_callback.cash = 84950.0
    tui.set_state_callback(state_callback)

    # 渲染TUI
    layout = tui.render()

    print("✓ TUI渲染成功")
    print(f"  观察列表: {len(tui.watchlist)} 个标的")
    print(f"  选中标的: {tui.selected_symbol}")

    return True


def test_agent_integration():
    """测试Agent集成"""
    from agent.runner import create_default_agent

    print("\n" + "="*60)
    print("测试 5: Agent完整集成")
    print("="*60)

    # 创建Agent
    agent = create_default_agent(initial_cash=100000, log_dir="test_logs")

    # 运行回测
    results = agent.run_replay_backtest(
        start=date(2024, 1, 8),
        end=date(2024, 1, 8),
        symbols=["AAPL", "GOOGL"],
        interval="1d",
        verbose=False
    )

    print(f"✓ 回测完成")
    print(f"  处理日期: {len(results)} 天")
    print(f"  最终权益: ${agent.broker.get_total_equity():,.2f}")

    # 检查事件日志
    from events.event_log import EventLogger
    logger = EventLogger(log_dir="test_logs")
    events = logger.query_events()
    print(f"  事件数量: {len(events)}")

    logger.close()
    return True


def main():
    """运行所有测试"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║          新功能集成测试 - 完整测试套件                    ║")
    print("╚════════════════════════════════════════════════════════════╝")

    tests = [
        ("事件日志系统", test_event_logging),
        ("日计划生成器", test_daily_planner),
        ("日复盘生成器", test_daily_reviewer),
        ("TUI界面", test_tui),
        ("Agent集成", test_agent_integration),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ 通过" if success else "❌ 失败"))
        except Exception as e:
            results.append((name, f"❌ 错误: {e}"))
            import traceback
            traceback.print_exc()

    # 清理测试文件
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")
        print("\n✓ 测试文件已清理")

    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    for name, result in results:
        print(f"{result} {name}")

    passed = sum(1 for _, r in results if "✅" in r)
    total = len(results)

    print("\n" + "="*60)
    if passed == total:
        print(f"✅ 所有测试通过 ({passed}/{total})")
        return 0
    else:
        print(f"❌ 部分测试失败 ({passed}/{total})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
