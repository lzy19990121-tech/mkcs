from datetime import datetime, timedelta
from decimal import Decimal

from broker.paper import PaperBroker
from core.models import Signal, OrderIntent, Bar


def test_order_not_filled_same_bar():
    broker = PaperBroker(initial_cash=100000, commission_per_share=0.01)

    t0 = datetime(2024, 1, 2, 9, 30)
    signal = Signal(
        symbol="AAPL",
        timestamp=t0,
        action="BUY",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.8,
        reason="test"
    )
    intent = OrderIntent(signal=signal, timestamp=t0, approved=True, risk_reason="OK")
    broker.submit_order(intent)

    bar_t0 = Bar(
        symbol="AAPL",
        timestamp=t0,
        open=Decimal("150.00"),
        high=Decimal("151.00"),
        low=Decimal("149.00"),
        close=Decimal("150.50"),
        volume=100000,
        interval="1d"
    )
    fills, rejects = broker.on_bar(bar_t0)
    assert len(fills) == 0

    bar_t1 = Bar(
        symbol="AAPL",
        timestamp=t0 + timedelta(days=1),
        open=Decimal("151.00"),
        high=Decimal("152.00"),
        low=Decimal("150.00"),
        close=Decimal("151.50"),
        volume=100000,
        interval="1d"
    )
    fills, rejects = broker.on_bar(bar_t1)
    assert len(fills) == 1
