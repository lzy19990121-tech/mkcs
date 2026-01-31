"""
Replay engine for deterministic backtests.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterator, List

from core.context import RunContext
from skills.session.trading_session import TradingSession


@dataclass(frozen=True)
class ReplayPoint:
    ctx: RunContext
    sessions: List[tuple[datetime, datetime]]


class ReplayEngine:
    def __init__(self, start: date, end: date, interval: str, market: str):
        self.start = start
        self.end = end
        self.interval = interval
        self.market = market

    def iter_days(self) -> Iterator[ReplayPoint]:
        trading_days = TradingSession.get_trading_days(self.start, self.end, self.market)
        for trading_date in trading_days:
            sessions = TradingSession.get_trading_sessions(trading_date, self.market)
            if not sessions:
                continue
            ctx = RunContext(
                now=sessions[0][0],
                trading_date=trading_date,
                mode="replay",
                bar_interval=self.interval
            )
            yield ReplayPoint(ctx=ctx, sessions=sessions)
