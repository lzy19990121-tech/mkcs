"""
风控抽象基类

定义风险管理的接口契约
"""

from abc import ABC, abstractmethod
from typing import Dict
from typing import Optional

from core.models import Signal, OrderIntent, Position


class RiskManager(ABC):
    """风控管理器抽象基类"""

    @abstractmethod
    def check(
        self,
        signal: Signal,
        positions: Dict[str, Position],
        cash_balance: float
    ) -> OrderIntent:
        """风控检查

        Args:
            signal: 交易信号
            positions: 当前持仓字典
            cash_balance: 可用资金

        Returns:
            订单意图（包含风控结果）
        """
        pass

    @abstractmethod
    def set_capital(self, initial_capital: float):
        """设置初始资金"""
        pass

    @abstractmethod
    def update_daily_pnl(self, pnl):
        """更新当日盈亏"""
        pass
