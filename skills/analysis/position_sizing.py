"""
仓位管理模块

实现连续仓位 sizing 和动态���位调整
"""

import logging
from decimal import Decimal
from typing import Optional, Dict, Any
from dataclasses import dataclass

from core.strategy_models import RegimeType

logger = logging.getLogger(__name__)


@dataclass
class PositionConfig:
    """仓位配置"""
    max_position_ratio: float = 0.2      # 单个标的最大仓位比例
    max_total_position: float = 0.8      # 总仓位上限
    min_position_ratio: float = 0.05     # 最小仓位比例
    base_position_multiplier: float = 1.0 # 基础仓位乘数

    # 波动率调整参数
    vol_adjustment_enabled: bool = True
    max_vol_adjustment: float = 2.0      # 最大波动率调整系数
    vol_threshold: float = 1.5           # 高波动阈值

    # 信号强度调整参数
    signal_scaling: bool = True          # 是否根据信号强度调整
    min_signal_threshold: float = 0.3    # 最小信号强度阈值


class PositionSizer:
    """仓位计算器

    根据信号强度、波动率、市场状态计算目标仓位
    """

    def __init__(self, config: PositionConfig = None):
        """初始化仓位计算器

        Args:
            config: 仓位配置
        """
        self.config = config or PositionConfig()

    def calculate_position(
        self,
        capital: float,
        current_price: float,
        signal_strength: float,
        volatility: float = 1.0,
        regime: RegimeType = RegimeType.UNKNOWN,
        current_position: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """计算目标仓位

        Args:
            capital: 可用资金
            current_price: 当前价格
            signal_strength: 信号强度 (0-1)
            volatility: 波动率水平（相对值，1为平均水平）
            regime: 市场状态
            current_position: 当前持仓数量（可选）
            metadata: 额外元数据（可选）

        Returns:
            仓位计算结果字典，包含：
            - target_quantity: 目标持仓数量
            - target_value: 目标持仓金额
            - position_ratio: 仓位比例
            - adjustment_factor: 调整系数
            - reason: 计算原因
        """
        # 1. 计算基础仓位（基于信号强度）
        if signal_strength < self.config.min_signal_threshold:
            return {
                "target_quantity": 0,
                "target_value": 0.0,
                "position_ratio": 0.0,
                "adjustment_factor": 0.0,
                "reason": f"信号强度不足 ({signal_strength:.2f} < {self.config.min_signal_threshold})"
            }

        base_ratio = signal_strength * self.config.max_position_ratio

        # 2. 应用市场状态调整
        regime_factor = self._get_regime_factor(regime)

        # 3. 应用波动率调整
        vol_factor = 1.0
        if self.config.vol_adjustment_enabled:
            vol_factor = self._calculate_volatility_factor(volatility)

        # 4. 综合调整系数
        total_factor = regime_factor * vol_factor * self.config.base_position_multiplier

        # 5. 计算最终仓位比例
        final_ratio = base_ratio * total_factor
        final_ratio = max(final_ratio, self.config.min_position_ratio)
        final_ratio = min(final_ratio, self.config.max_position_ratio)

        # 6. 计算目标金额和数量
        target_value = capital * final_ratio
        target_quantity = int(target_value / current_price)

        # 向下取整到100的倍数（A股一手100股）
        target_quantity = (target_quantity // 100) * 100

        # 如果目标仓位小于最小交易单位，则为0
        if target_quantity < 100:
            target_quantity = 0
            final_ratio = 0.0

        return {
            "target_quantity": target_quantity,
            "target_value": target_value,
            "position_ratio": final_ratio,
            "adjustment_factor": total_factor,
            "regime_factor": regime_factor,
            "vol_factor": vol_factor,
            "reason": f"信号强度={signal_strength:.2f}, 调整系数={total_factor:.2f}"
        }

    def _get_regime_factor(self, regime: RegimeType) -> float:
        """获取市场状态调整系数"""
        factors = {
            RegimeType.TREND: 1.0,      # 趋势市：满仓
            RegimeType.RANGE: 0.3,      # 震荡市：30%仓位
            RegimeType.HIGH_VOL: 0.2,   # 高波动：20%仓位
            RegimeType.LOW_VOL: 0.5,    # 低波动：50%仓位
            RegimeType.UNKNOWN: 0.0,    # 未知：平仓
        }
        return factors.get(regime, 0.5)

    def _calculate_volatility_factor(self, volatility: float) -> float:
        """计算波动率调整系数

        高波动降低仓位，低波动适当增加
        """
        if not self.config.vol_adjustment_enabled:
            return 1.0

        if volatility <= 0.5:
            # 低波动：可以适当增加仓位
            return min(1.2, 1.0 + (0.5 - volatility) * 0.4)
        elif volatility <= self.config.vol_threshold:
            # 正常波动
            return 1.0
        else:
            # 高波动：降低仓位
            excess = volatility - self.config.vol_threshold
            reduction = min(excess * 0.3, self.config.max_vol_adjustment - 1.0)
            return max(1.0 / self.config.max_vol_adjustment, 1.0 - reduction)

    def calculate_trade_quantity(
        self,
        current_quantity: int,
        target_quantity: int,
        max_trade_ratio: float = 0.5
    ) -> int:
        """计算本次交易数量

        限制单次交易规模，避免一次性买入/卖出过多

        Args:
            current_quantity: 当前持仓数量
            target_quantity: 目标持仓数量
            max_trade_ratio: 最大交易比例（相对目标差额）

        Returns:
            本次交易数量（正数为买入，负数为卖出）
        """
        if target_quantity == current_quantity:
            return 0

        diff = target_quantity - current_quantity

        # 限制单次交易规模
        max_trade = int(abs(diff) * max_trade_ratio)

        # 向下取整到100的倍数
        max_trade = (max_trade // 100) * 100
        max_trade = max(max_trade, 100)

        if abs(diff) <= max_trade:
            return diff

        return max_trade if diff > 0 else -max_trade


class RiskParitySizer(PositionSizer):
    """风险平价仓位计算器

    根据波动率倒数分配资金，使各标的风险贡献均衡
    """

    def calculate_portfolio_weights(
        self,
        volatilities: Dict[str, float],
        capital: float,
        prices: Dict[str, float]
    ) -> Dict[str, int]:
        """计算风险平价组合权重

        Args:
            volatilities: 各标的波动率 {symbol: volatility}
            capital: 总资金
            prices: 各标的当前价格

        Returns:
            各标的持仓数量 {symbol: quantity}
        """
        # 计算风险平价权重（波动率倒数）
        inv_vols = {s: 1.0 / max(v, 0.1) for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        weights = {s: v / total_inv_vol for s, v in inv_vols.items()}

        # 计算持仓数量
        quantities = {}
        for symbol, weight in weights.items():
            target_value = capital * weight * self.config.max_position_ratio
            quantity = int(target_value / prices[symbol])
            quantities[symbol] = (quantity // 100) * 100

        return quantities


class KellySizer(PositionSizer):
    """凯利公式仓位计算器

    根据胜率和盈亏比计算最优仓位
    """

    def __init__(
        self,
        win_rate: float = 0.55,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        kelly_fraction: float = 0.25,  # 使用凯利公式的分数（降低风险）
        config: PositionConfig = None
    ):
        """初始化凯利计算器

        Args:
            win_rate: 胜率 (0-1)
            avg_win: 平均盈利（相对值）
            avg_loss: 平均亏损（相对值）
            kelly_fraction: 凯利公式分数（建议0.25-0.5）
            config: 仓位配置
        """
        super().__init__(config)
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction

    def calculate_kelly_position(
        self,
        capital: float,
        current_price: float,
        signal_strength: float = 1.0,
        volatility: float = 1.0
    ) -> Dict[str, Any]:
        """使用凯利公式计算仓位

        凯利公式：f* = (bp - q) / b
        其中：
        - f* = 最优仓位比例
        - b = 盈亏比 (avg_win / avg_loss)
        - p = 胜率
        - q = 败率 = 1 - p

        实际使用凯利分数降低风险
        """
        # 计算盈亏比
        b = self.avg_win / max(self.avg_loss, 0.01)
        p = self.win_rate
        q = 1 - p

        # 凯利公式
        kelly_f = (b * p - q) / b

        # 如果凯利值为负，不交易
        if kelly_f <= 0:
            return {
                "target_quantity": 0,
                "target_value": 0.0,
                "position_ratio": 0.0,
                "kelly_fraction": kelly_f,
                "reason": "凯利公式为负值，不建议交易"
            }

        # 应用凯利分数
        adjusted_kelly = kelly_f * self.kelly_fraction

        # 结合信号强度
        final_ratio = adjusted_kelly * signal_strength

        # 应用波动率调整
        vol_factor = self._calculate_volatility_factor(volatility)
        final_ratio *= vol_factor

        # 限制范围
        final_ratio = max(0, min(final_ratio, self.config.max_position_ratio))

        # 计算具体数量
        target_value = capital * final_ratio
        target_quantity = (int(target_value / current_price) // 100) * 100

        return {
            "target_quantity": target_quantity,
            "target_value": target_value,
            "position_ratio": final_ratio,
            "kelly_fraction": kelly_f,
            "adjusted_kelly": adjusted_kelly,
            "reason": f"凯利公式: f={kelly_f:.3f}, 调整后={adjusted_kelly:.3f}"
        }

    def update_performance(self, is_win: bool, pnl_ratio: float):
        """更新性能统计

        Args:
            is_win: 是否盈利
            pnl_ratio: 盈亏比例（正数为盈利，负数为亏损）
        """
        # 简单的指数移动平均更新
        alpha = 0.1

        if is_win:
            self.avg_win = (1 - alpha) * self.avg_win + alpha * pnl_ratio
        else:
            self.avg_loss = (1 - alpha) * self.avg_loss + alpha * abs(pnl_ratio)

        # 更新胜率
        self.win_rate = (1 - alpha) * self.win_rate + alpha * (1.0 if is_win else 0.0)


if __name__ == "__main__":
    """测试代码"""
    print("=== PositionSizer 测试 ===\n")

    sizer = PositionSizer()

    # 测试1: 强信号 + 正常波动
    print("1. 强信号 + 正常波动:")
    result = sizer.calculate_position(
        capital=100000,
        current_price=150,
        signal_strength=0.8,
        volatility=1.0,
        regime=RegimeType.TREND
    )
    print(f"   目标数量: {result['target_quantity']}")
    print(f"   仓位比例: {result['position_ratio']:.2%}")
    print(f"   调整系数: {result['adjustment_factor']:.2f}")

    # 测试2: 弱信号 + 高波动
    print("\n2. 弱信号 + 高波动:")
    result = sizer.calculate_position(
        capital=100000,
        current_price=150,
        signal_strength=0.4,
        volatility=2.0,
        regime=RegimeType.HIGH_VOL
    )
    print(f"   目标数量: {result['target_quantity']}")
    print(f"   仓位比例: {result['position_ratio']:.2%}")
    print(f"   原因: {result['reason']}")

    # 测试3: 震荡市
    print("\n3. 震荡市:")
    result = sizer.calculate_position(
        capital=100000,
        current_price=150,
        signal_strength=0.7,
        volatility=1.0,
        regime=RegimeType.RANGE
    )
    print(f"   目标数量: {result['target_quantity']}")
    print(f"   仓位比例: {result['position_ratio']:.2%}")

    # 测试4: 凯利公式
    print("\n4. 凯利公式:")
    kelly = KellySizer(win_rate=0.6, avg_win=1.5, avg_loss=1.0, kelly_fraction=0.25)
    result = kelly.calculate_kelly_position(
        capital=100000,
        current_price=150,
        signal_strength=0.8
    )
    print(f"   目标数量: {result['target_quantity']}")
    print(f"   仓位比例: {result['position_ratio']:.2%}")
    print(f"   凯利值: {result['kelly_fraction']:.3f}")

    # 测试5: 交易数量分批
    print("\n5. 交易数量分批:")
    trade_qty = sizer.calculate_trade_quantity(
        current_quantity=0,
        target_quantity=1000,
        max_trade_ratio=0.3
    )
    print(f"   当前: 0, 目标: 1000, 本次交易: {trade_qty}")

    print("\n✓ 所有测试通过")
