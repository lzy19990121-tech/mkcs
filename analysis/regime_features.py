"""
SPL-5a-A: 自适应输入特征定义

定义市场状态特征（Regime Features），用于自适应风控决策。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque
import numpy as np


class RegimeState(Enum):
    """市场状态枚举"""
    VOLATILITY = "volatility"
    TREND = "trend"
    LIQUIDITY = "liquidity"


@dataclass
class RegimeFeatures:
    """市场状态特征

    包含用于自适应风控的三个维度：
    1. 波动状态（realized_vol）
    2. 趋势/震荡状态（ADX）
    3. 流动性/成本状态（spread_proxy）
    """

    # 波动状态
    realized_vol: float              # 过去20日收益波动率
    vol_bucket: str                    # "low", "med", "high"

    # 趋势状态
    adx: float                           # 平均趋向指数
    trend_bucket: str                   # "weak", "strong"

    # 流动性状态
    spread_proxy: float                 # 价差代理
    cost_bucket: str                    # "low", "high"

    # 元数据
    calculated_at: datetime
    window_length: int = 20             # 计算窗口长度

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "realized_vol": self.realized_vol,
            "vol_bucket": self.vol_bucket,
            "adx": self.adx,
            "trend_bucket": self.trend_bucket,
            "spread_proxy": self.spread_proxy,
            "cost_bucket": self.cost_bucket,
            "calculated_at": self.calculated_at.isoformat(),
            "window_length": self.window_length
        }


# 分桶阈值配置
VOLATILITY_BUCKETS = {
    "low": (0.0, 0.01),      # < 1%
    "med": (0.01, 0.02),     # 1% - 2%
    "high": (0.02, float("inf"))  # > 2%
}

TREND_BUCKETS = {
    "weak": (0, 25),          # ADX < 25
    "strong": (25, float("inf")) # ADX >= 25
}

COST_BUCKETS = {
    "low": (0.0, 0.001),     # < 0.1%
    "high": (0.001, float("inf")) # >= 0.1%
}


def bucket_value(value: float, buckets: Dict[str, Tuple]) -> str:
    """将数值分桶"""
    for bucket_name, (min_val, max_val) in buckets.items():
        if min_val <= value < max_val:
            return bucket_name
    # 如果都不匹配，返回最后一个桶
    return list(buckets.keys())[-1]


class RegimeFeatureCalculator:
    """计算市场状态特征

    从运行时数据中实时计算波动、趋势、流动性等特征。
    """

    def __init__(
        self,
        window_length: int = 20,
        max_history: int = 100
    ):
        """初始化特征计算器

        Args:
            window_length: 计算窗口长度（天数）
            max_history: 最大历史数据点数
        """
        self.window_length = window_length
        self.max_history = max_history

        # 历史数据缓冲区
        self.price_history: deque = deque(maxlen=max_history)
        self.return_history: deque = deque(maxlen=max_history)
        self.high_history: deque = deque(maxlen=max_history)
        self.low_history: deque = deque(maxlen=max_history)

        # 状态
        self.initialized = False
        self.peak_price = None
        self.current_price = None

    def update(self, price: float, timestamp: datetime = None) -> None:
        """更新价格数据

        Args:
            price: 当前价格
            timestamp: 时间戳
        """
        self.current_price = price

        # 初始化峰值
        if self.peak_price is None:
            self.peak_price = price

        # 更新峰值
        if price > self.peak_price:
            self.peak_price = price

        # 添加到历史（即使是第一个）
        self.price_history.append(price)

        self.initialized = True

    def calculate_return(self, price: float) -> float:
        """计算单期收益（对数收益率）"""
        if len(self.price_history) < 2:
            return 0.0
        prev_price = self.price_history[-2] if len(self.price_history) >= 2 else price
        if prev_price > 0:
            return np.log(price / prev_price)
        return 0.0

    def calculate(self) -> RegimeFeatures:
        """计算当前的市场状态特征

        Returns:
            RegimeFeatures 对象
        """
        if not self.initialized or len(self.price_history) < 2:
            # 数据不足，返回默认值
            return RegimeFeatures(
                realized_vol=0.0,
                vol_bucket="low",
                adx=25.0,  # 默认中等趋势
                trend_bucket="strong",
                spread_proxy=0.001,
                cost_bucket="low",
                calculated_at=datetime.now(),
                window_length=self.window_length
            )

        # 1. 计算波动率
        # 使用最近 window_length 个收益
        if len(self.return_history) < self.window_length:
            # 如果数据不足，用现有数据
            window_data = list(self.price_history)
        else:
            window_data = list(self.price_history)[-self.window_length:]

        # 计算收益序列
        returns = []
        for i in range(1, len(window_data)):
            prev = window_data[i-1]
            curr = window_data[i]
            if prev > 0:
                ret = np.log(curr / prev)
                returns.append(ret)
                self.return_history.append(ret)

        # realized_vol = 收益标准差 * sqrt(252) 年化
        if len(returns) > 1:
            realized_vol = np.std(returns) * np.sqrt(252)
        else:
            realized_vol = 0.01

        # 分桶
        vol_bucket = bucket_value(realized_vol, VOLATILITY_BUCKETS)

        # 2. 计算ADX（趋势强度）
        adx = self._calculate_adx()
        trend_bucket = bucket_value(adx, TREND_BUCKETS)

        # 3. 计算价差代理
        spread_proxy = self._estimate_spread()
        cost_bucket = bucket_value(spread_proxy, COST_BUCKETS)

        return RegimeFeatures(
            realized_vol=realized_vol,
            vol_bucket=vol_bucket,
            adx=adx,
            trend_bucket=trend_bucket,
            spread_proxy=spread_proxy,
            cost_bucket=cost_bucket,
            calculated_at=datetime.now(),
            window_length=self.window_length
        )

    def _calculate_adx(self) -> float:
        """计算平均趋向指数（简化版）

        使用简化的方法计算趋势强度：
        - 基于价格方向的一致性
        - 返回 0-100 的值（模仿 ADX）

        Returns:
            ADX 值 (0-100)
        """
        if len(self.price_history) < 14:
            return 25.0  # 默认中等趋势

        prices = np.array(list(self.price_history)[-14:])

        # 计算方向变化
        deltas = np.diff(prices)
        up_moves = np.sum(deltas > 0)
        down_moves = np.sum(deltas < 0)

        total_moves = up_moves + down_moves
        if total_moves == 0:
            return 25.0

        # 方向一致性 = (净方向数 / 总变动数) * 100
        direction_strength = abs(up_moves - down_moves) / total_moves * 100

        # 映射到 0-100 范围
        adx = min(100.0, direction_strength * 1.5 + 10)  # 调整使得范围更广

        return adx

    def _estimate_spread(self) -> float:
        """估算价差（基于价格波动）

        使用价格范围作为价差代理：
        spread_proxy = (high - low) / close

        Returns:
            价差代理值
        """
        if len(self.price_history) < 2:
            return 0.001

        # 使用最近20天的价格范围
        window_size = min(20, len(self.price_history))
        recent_prices = list(self.price_history)[-window_size:]

        price_range = max(recent_prices) - min(recent_prices)
        avg_price = np.mean(recent_prices)

        if avg_price > 0:
            return price_range / avg_price / 10  # 缩放到合理范围
        return 0.001

    def get_risk_adjustment_factor(self, regime: RegimeFeatures) -> float:
        """获取风险调整因子

        根据市场状态返回风险乘数：
        - 高波动 → 高风险 → factor > 1
        - 低波动 → 低风险 → factor ≈ 1
        - 震荡市场 → 风险提升

        Args:
            regime: 市场状态特征

        Returns:
            风险调整因子（0.8 - 1.5）
        """
        factor = 1.0

        # 波动调整
        if regime.vol_bucket == "high":
            factor *= 1.3
        elif regime.vol_bucket == "low":
            factor *= 0.9

        # 趋势调整
        if regime.trend_bucket == "weak":
            factor *= 1.2  # 震荡市场风险提升

        # 流动性调整
        if regime.cost_bucket == "high":
            factor *= 1.1

        return factor


# 全局计算器实例（按策略ID管理）
_calculators: Dict[str, RegimeFeatureCalculator] = {}


def get_calculator(strategy_id: str, window_length: int = 20) -> RegimeFeatureCalculator:
    """获取或创建特征计算器

    Args:
        strategy_id: 策略ID
        window_length: 窗口长度

    Returns:
        RegimeFeatureCalculator 实例
    """
    if strategy_id not in _calculators:
        _calculators[strategy_id] = RegimeFeatureCalculator(window_length=window_length)
    return _calculators[strategy_id]


def calculate_regime_from_replay(
    replay,
    window_length: str = "20d"
) -> RegimeFeatures:
    """从回测数据计算市场状态

    Args:
        replay: 回测输出
        window_length: 窗口长度（如 "20d", "60d"）

    Returns:
        RegimeFeatures 对象
    """
    # 解析窗口长度
    if window_length.endswith("d"):
        window_days = int(window_length[:-1])
    else:
        window_days = 20

    # 创建计算器
    calculator = RegimeFeatureCalculator(window_length=window_days)

    # 使用回测数据填充
    from analysis.replay_schema import ReplayOutput
    if isinstance(replay, ReplayOutput):
        for step in replay.steps:
            # 从 equity 计算价格（简化：假设初始100000，使用 equity）
            price = float(step.equity) / 1000.0  # 简化价格
            calculator.update(price, step.timestamp)

    return calculator.calculate()


if __name__ == "__main__":
    """测试代码"""
    print("=== RegimeFeatures 测试 ===\n")

    # 测试1: 创建默认特征
    print("1. 默认特征（数据不足）:")
    default_features = RegimeFeatures(
        realized_vol=0.0,
        vol_bucket="low",
        adx=25.0,
        trend_bucket="strong",
        spread_proxy=0.001,
        cost_bucket="low",
        calculated_at=datetime.now()
    )
    print(f"   波动: {default_features.realized_vol:.4f} ({default_features.vol_bucket})")
    print(f"   趋势: {default_features.adx:.1f} ({default_features.trend_bucket})")
    print(f"   流动性: {default_features.spread_proxy:.4f} ({default_features.cost_bucket})")

    # 测试2: 分桶函数
    print("\n2. 分桶测试:")
    vol_values = [0.005, 0.015, 0.03]
    for v in vol_values:
        bucket = bucket_value(v, VOLATILITY_BUCKETS)
        print(f"   vol={v:.3f} → bucket={bucket}")

    print("\n✓ 测试通过")
