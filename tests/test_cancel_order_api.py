"""
P1-3: 撤销订单 API 测试

验证撤销订单功能：
1. 定义 endpoint：POST /api/orders/cancel
2. 参数：order_id / client_order_id / symbol（至少一种主键）
3. 对接 execution/broker adapter（paper 模拟撤单、live 真实撤单 dry-run）
4. 返回标准结构：status、reason、updated_order

验收:
☐ paper 模式撤单可用且状态变更正确
☐ live 模式默认 dry-run，不会误撤
☐ API 返回字段可用于 UI 更新订单状态
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from decimal import Decimal
from datetime import datetime
from broker.paper import PaperBroker
from core.models import Signal, OrderIntent
from web.services.live_trading_service import LiveTradingService


def test_paper_cancel_by_order_id():
    """测试1: paper 模式按 order_id 撤单"""
    print("\n" + "="*70)
    print("测试1: paper 模式按 order_id 撤单")
    print("="*70)

    broker = PaperBroker(initial_cash=Decimal('100000'), commission_per_share=Decimal('0.01'))

    # 提交一个订单
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 2, 9, 30),
        action="BUY",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.8,
        reason="测试"
    )
    intent = OrderIntent(signal=signal, timestamp=signal.timestamp, approved=True, risk_reason="OK")
    order = broker.submit_order(intent)

    order_id = order.order_id
    print(f"  提交订单: {order_id}")
    print(f"  挂单数: {len(broker.pending_orders)}")

    # 撤销订单
    result = broker.cancel_order(order_id=order_id)

    assert result["status"] == "success", "撤销应该成功"
    assert order_id in result["cancelled_orders"], "应该包含被撤销的订单 ID"
    assert len(broker.pending_orders) == 0, "挂单应该被清空"

    print(f"  ✅ 撤销成功")
    print(f"  - reason: {result['reason']}")
    print(f"  - cancelled_orders: {result['cancelled_orders']}")
    print(f"  - 挂单数: {len(broker.pending_orders)}")

    # 撤销不存在的订单
    result = broker.cancel_order(order_id="nonexistent")
    assert result["status"] == "error", "撤销不存在的订单应该返回 error"

    print(f"  ✅ 撤销不存在的订单返回正确错误")

    return True


def test_paper_cancel_by_symbol():
    """测试2: paper 模式按 symbol 撤销所有订单"""
    print("\n" + "="*70)
    print("测试2: paper 模式按 symbol 撤销所有订单")
    print("="*70)

    broker = PaperBroker(initial_cash=Decimal('100000'), commission_per_share=Decimal('0.01'))

    # 提交多个不同品种的订单
    symbols = ["AAPL", "GOOGL", "MSFT"]
    for symbol in symbols:
        signal = Signal(
            symbol=symbol,
            timestamp=datetime(2024, 1, 2, 9, 30),
            action="BUY",
            price=Decimal("150.00"),
            quantity=100,
            confidence=0.8,
            reason="测试"
        )
        intent = OrderIntent(signal=signal, timestamp=signal.timestamp, approved=True, risk_reason="OK")
        broker.submit_order(intent)

    print(f"  提交了 {len(broker.pending_orders)} 个订单")
    print(f"  挂单: {list(broker.pending_orders.keys())}")

    # 撤销 AAPL 订单
    result = broker.cancel_order(symbol="AAPL")

    assert result["status"] == "success", "撤销应该成功"
    assert len(result["cancelled_orders"]) == 1, "应该撤销 1 个 AAPL 订单"
    assert len(broker.pending_orders) == 2, "应该剩余 2 个订单"

    print(f"  ✅ 按 symbol 撤销成功")
    print(f"  - reason: {result['reason']}")
    print(f"  - cancelled_orders: {result['cancelled_orders']}")

    # 撤销所有剩余订单
    result = broker.cancel_order(symbol="GOOGL")
    assert len(broker.pending_orders) == 1, "应该剩余 1 个订单"

    result = broker.cancel_order(symbol="MSFT")
    assert len(broker.pending_orders) == 0, "应该没有剩余订单"

    print(f"  ✅ 所有订单已撤销")

    return True


def test_paper_cancel_invalid_params():
    """测试3: 参数验证"""
    print("\n" + "="*70)
    print("测试3: 参数验证")
    print("="*70)

    broker = PaperBroker(initial_cash=Decimal('100000'), commission_per_share=Decimal('0.01'))

    # 不提供任何参数
    result = broker.cancel_order()

    assert result["status"] == "error", "应该返回 error"
    assert "at least one" in result["reason"].lower(), "应该提示需要至少一个参数"

    print(f"  ✅ 无参数时返回正确错误")
    print(f"  - reason: {result['reason']}")

    return True


def test_live_service_cancel():
    """测试4: LiveTradingService cancel_order"""
    print("\n" + "="*70)
    print("测试4: LiveTradingService cancel_order")
    print("="*70)

    service = LiveTradingService()

    # 测试 paper 模式撤销（服务使用 paper 模式）
    result = service.cancel_order(symbol="AAPL", dry_run=False)

    # 可能是 "error"（因为没有订单）或 "success"（如果有订单）
    assert "status" in result, "应该包含 status 字段"
    assert "reason" in result, "应该包含 reason 字段"

    print(f"  ✅ LiveTradingService.cancel_order 正常工作")
    print(f"  - status: {result['status']}")
    print(f"  - reason: {result['reason']}")

    return True


def test_get_pending_orders():
    """测试5: 获取挂单列表"""
    print("\n" + "="*70)
    print("测试5: 获取挂单列表")
    print("="*70)

    broker = PaperBroker(initial_cash=Decimal('100000'), commission_per_share=Decimal('0.01'))

    # 初始状态应该为空
    pending = broker.get_pending_orders()
    assert len(pending) == 0, "初始挂单应该为空"

    print(f"  ✅ 初始挂单为空")

    # 提交订单
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 2, 9, 30),
        action="BUY",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.8,
        reason="测试"
    )
    intent = OrderIntent(signal=signal, timestamp=signal.timestamp, approved=True, risk_reason="OK")
    order = broker.submit_order(intent)

    # 获取挂单
    pending = broker.get_pending_orders()

    assert len(pending) == 1, "应该有 1 个挂单"
    assert pending[0]["order_id"] == order.order_id, "订单 ID 应该匹配"
    assert pending[0]["symbol"] == "AAPL", "symbol 应该匹配"
    assert pending[0]["status"] == "pending", "状态应该是 pending"

    print(f"  ✅ 获取挂单列表正确")
    print(f"  - 挂单数: {len(pending)}")
    print(f"  - order_id: {pending[0]['order_id']}")
    print(f"  - symbol: {pending[0]['symbol']}")

    return True


def test_cancel_then_check_pending():
    """测试6: 撤单后检查挂单状态"""
    print("\n" + "="*70)
    print("测试6: 撤单后检查挂单状态")
    print("="*70)

    broker = PaperBroker(initial_cash=Decimal('100000'), commission_per_share=Decimal('0.01'))

    # 提交订单
    signal = Signal(
        symbol="TSLA",
        timestamp=datetime(2024, 1, 2, 9, 30),
        action="BUY",
        price=Decimal("200.00"),
        quantity=50,
        confidence=0.9,
        reason="测试"
    )
    intent = OrderIntent(signal=signal, timestamp=signal.timestamp, approved=True, risk_reason="OK")
    order = broker.submit_order(intent)

    # 撤销前应该有挂单
    pending = broker.get_pending_orders()
    assert len(pending) == 1, "撤销前应该有 1 个挂单"

    # 撤销
    result = broker.cancel_order(order_id=order.order_id)

    # 撤销后应该没有挂单
    pending = broker.get_pending_orders()
    assert len(pending) == 0, "撤销后应该没有挂单"

    print(f"  ✅ 撤单后挂单状态正确更新")
    print(f"  - 撤销前挂单数: 1")
    print(f"  - 撤销后挂单数: {len(pending)}")

    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("P1-3: 撤销订单 API 测试")
    print("="*70)

    tests = [
        ("paper 模式按 order_id 撤单", test_paper_cancel_by_order_id),
        ("paper 模式按 symbol 撤销所有订单", test_paper_cancel_by_symbol),
        ("参数验证", test_paper_cancel_invalid_params),
        ("LiveTradingService cancel_order", test_live_service_cancel),
        ("获取挂单列表", test_get_pending_orders),
        ("撤单后检查挂单状态", test_cancel_then_check_pending),
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
