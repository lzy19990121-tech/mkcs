"""
Run context definitions.
"""

from dataclasses import dataclass
from datetime import datetime, date


@dataclass(frozen=True)
class RunContext:
    now: datetime
    trading_date: date
    mode: str
    bar_interval: str
