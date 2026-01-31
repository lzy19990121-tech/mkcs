"""
Replay engine for deterministic backtests.
"""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Iterator, List, Optional, Dict, Any
from pathlib import Path

from core.context import RunContext
from skills.session.trading_session import TradingSession


class ReplayMode(Enum):
    """回放模式"""
    MOCK = "replay_mock"      # 工程闭环验证
    REAL = "replay_real"      # 策略/风控验证 (CSV)


@dataclass(frozen=True)
class ReplayPoint:
    ctx: RunContext
    sessions: List[tuple[datetime, datetime]]


class ReplayEngine:
    """回放引擎

    支持两种模式:
    - MOCK: 使用模拟数据，用于工程验证
    - REAL: 使用 CSV 数据，用于策略/风控验证
    """

    def __init__(
        self,
        start: date,
        end: date,
        interval: str,
        market: str,
        mode: ReplayMode = ReplayMode.MOCK,
        config: Optional[Any] = None
    ):
        self.start = start
        self.end = end
        self.interval = interval
        self.market = market
        self.mode = mode
        self.config = config

        # 计算配置哈希
        self.config_hash = self._compute_config_hash()

    def _compute_config_hash(self) -> str:
        """计算配置哈希"""
        try:
            from utils.hash import compute_config_hash
            if self.config:
                return compute_config_hash(self.config.to_dict())
        except Exception:
            pass
        return "unknown"

    def iter_days(self) -> Iterator[ReplayPoint]:
        trading_days = TradingSession.get_trading_days(self.start, self.end, self.market)
        for trading_date in trading_days:
            sessions = TradingSession.get_trading_sessions(trading_date, self.market)
            if not sessions:
                continue
            ctx = RunContext(
                now=sessions[0][0],
                trading_date=trading_date,
                mode=self.mode.value,
                bar_interval=self.interval
            )
            yield ReplayPoint(ctx=ctx, sessions=sessions)

    def get_metadata(self) -> Dict[str, Any]:
        """获取回放元数据"""
        from utils.hash import get_git_commit, get_repo_status

        return {
            "replay_mode": self.mode.value,
            "config_hash": self.config_hash,
            "git_commit": get_git_commit(short=True),
            "git_branch": get_repo_status().get("branch", "unknown"),
            "repo_dirty": get_repo_status().get("dirty", False),
            "start_date": self.start.isoformat(),
            "end_date": self.end.isoformat(),
            "interval": self.interval,
            "market": self.market
        }
