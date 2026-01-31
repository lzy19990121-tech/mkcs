"""
市场数据源抽象基类

定义获取市场数据的接口契约
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from core.models import Bar, Quote


class MarketDataSource(ABC):
    """市场数据源抽象基类"""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取K线数据

        Args:
            symbol: 标的代码
            start: 开始时间
            end: 结束时间
            interval: 时间周期（1m, 5m, 1h, 1d）

        Returns:
            K线数据列表，按时间升序排列
        """
        pass

    @abstractmethod
    def get_bars_until(
        self,
        symbol: str,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取截至某个时间点的K线数据

        Args:
            symbol: 标的代码
            end: 截止时间（包含该时间点）
            interval: 时间周期（1m, 5m, 1h, 1d）

        Returns:
            K线数据列表，按时间升序排列
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """获取实时报价

        Args:
            symbol: 标的代码

        Returns:
            最新报价
        """
        pass
