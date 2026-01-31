"""
策略抽象基类

定义交易策略的接口契约
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import Bar, Signal, Position


class Strategy(ABC):
    """交易策略抽象基类"""

    @abstractmethod
    def generate_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None
    ) -> List[Signal]:
        """生成交易信号

        Args:
            bars: 历史K线数据（按时间升序）
            position: 当前持仓（可选）

        Returns:
            信号列表
        """
        pass

    @abstractmethod
    def get_min_bars_required(self) -> int:
        """获取策略所需的最小K线数量

        Returns:
            最小K线数量
        """
        pass
