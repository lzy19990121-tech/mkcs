"""
市场状态检测器

识别市场当前处于趋势市、震荡市、高波动等状态
为策略提供市场上下文，实现策略 gating
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

import numpy as np

from core.models import Bar
from core.strategy_models import RegimeType, RegimeInfo

logger = logging.getLogger(__name__)


class RegimeDetectorConfig:
    """市场状态检测配置"""

    def __init__(
        self,
        # ADX 相关参数
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,    # ADX > 25 表示有趋势
        adx_strong_trend: float = 40.0,       # ADX > 40 表示强趋势

        # 波动率相关参数
        atr_period: int = 14,
        volatility_lookback: int = 50,        # 波动率比较窗口
        high_vol_threshold: float = 1.5,      # 波动率超过中位数的倍数

        # 震荡检测参数
        range_period: int = 20,
        range_threshold: float = 0.02,        # 价格在区间内的波动阈值（2%）

        # RSI 参数
        rsi_period: int = 14,

        # 置信度计算权重
        trend_weight: float = 0.4,
        volatility_weight: float = 0.3,
        momentum_weight: float = 0.3
    ):
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_strong_trend = adx_strong_trend
        self.atr_period = atr_period
        self.volatility_lookback = volatility_lookback
        self.high_vol_threshold = high_vol_threshold
        self.range_period = range_period
        self.range_threshold = range_threshold
        self.rsi_period = rsi_period
        self.trend_weight = trend_weight
        self.volatility_weight = volatility_weight
        self.momentum_weight = momentum_weight


@dataclass
class RegimeState:
    """市场状态检测结果（带历史）"""
    current: RegimeInfo
    history: List[RegimeInfo] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def add_to_history(self, info: RegimeInfo):
        """添加到历史记录"""
        self.history.append(info)
        # 保留最近 100 条记录
        if len(self.history) > 100:
            self.history = self.history[-100:]

    @property
    def regime_duration(self) -> int:
        """当前状态持续天数"""
        if not self.history:
            return 0

        current_regime = self.current.regime
        count = 0
        for info in reversed(self.history):
            if info.regime == current_regime:
                count += 1
            else:
                break
        return count

    @property
    def regime_stability(self) -> float:
        """状态稳定性（最近20天中当前状态的比例）"""
        if not self.history:
            return 0.0

        recent = self.history[-20:] if len(self.history) >= 20 else self.history
        if not recent:
            return 0.0

        current_regime = self.current.regime
        same_count = sum(1 for info in recent if info.regime == current_regime)
        return same_count / len(recent)


class RegimeDetector:
    """市场状态检测器

    综合使用 ADX、ATR、价格波动等指标判断市场状态
    """

    def __init__(self, config: RegimeDetectorConfig = None):
        """初始化检测器

        Args:
            config: 检测配置
        """
        self.config = config or RegimeDetectorConfig()
        self._state: Optional[RegimeState] = None

    def detect(self, bars: List[Bar]) -> RegimeInfo:
        """检测当前市场状态

        Args:
            bars: K线数据，按时间升序排列

        Returns:
            市场状态信息
        """
        if len(bars) < max(self.config.adx_period, self.config.atr_period, self.config.range_period) + 10:
            # 数据不足，返回未知状态
            return RegimeInfo(regime=RegimeType.UNKNOWN, confidence=0.0)

        # 1. 计算 ADX（趋势强度）
        adx = self._calculate_adx(bars, self.config.adx_period)

        # 2. 计算波动率水平
        atr, volatility_level = self._calculate_volatility(bars)

        # 3. 计算价格是否在震荡区间
        is_range_bound, range_bound_strength = self._detect_range_bound(bars)

        # 4. 计算 RSI
        rsi = self._calculate_rsi(bars, self.config.rsi_period)

        # 5. 综合判断市场状态
        regime, confidence = self._classify_regime(
            adx, volatility_level, is_range_bound, range_bound_strength, rsi
        )

        # 6. 计算趋势强度（归一化到 0-1）
        trend_strength = min(adx / self.config.adx_strong_trend, 1.0)

        return RegimeInfo(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            adx=adx,
            rsi=rsi,
            atr=atr
        )

    def detect_with_state(self, bars: List[Bar]) -> RegimeState:
        """检测市场状态并更新历史记录

        Args:
            bars: K线数据

        Returns:
            市场状态（包含历史）
        """
        current_info = self.detect(bars)

        if self._state is None:
            self._state = RegimeState(current=current_info)
        else:
            self._state.current = current_info
            self._state.add_to_history(current_info)

        return self._state

    def _calculate_adx(self, bars: List[Bar], period: int = 14) -> float:
        """计算平均趋向指数 (ADX)

        ADX 衡量趋势强度：
        - ADX < 20: 无趋势/弱趋势
        - ADX 20-25: 有趋势迹象
        - ADX > 25: 明显趋势
        - ADX > 40: 强趋势
        """
        if len(bars) < period * 2:
            return 0.0

        # 转换价格数据
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]

        # 计算 +DM 和 -DM
        plus_dm = []
        minus_dm = []

        for i in range(1, len(closes)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

        # 计算 TR (True Range)
        tr = []
        for i in range(1, len(closes)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr.append(max(tr1, tr2, tr3))

        # 计算 +DI 和 -DI
        plus_di = []
        minus_di = []

        for i in range(period, len(tr)):
            # 平滑的 +DM 和 -DM
            smoothed_plus_dm = sum(plus_dm[i-period+1:i+1])
            smoothed_minus_dm = sum(minus_dm[i-period+1:i+1])
            smoothed_tr = sum(tr[i-period+1:i+1])

            if smoothed_tr > 0:
                plus_di.append(100 * smoothed_plus_dm / smoothed_tr)
                minus_di.append(100 * smoothed_minus_dm / smoothed_tr)
            else:
                plus_di.append(0)
                minus_di.append(0)

        # 计算 DX 和 ADX
        dx = []
        for i in range(len(plus_di)):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx.append(100 * abs(plus_di[i] - minus_di[i]) / di_sum)
            else:
                dx.append(0)

        # ADX 是 DX 的移动平均
        if len(dx) >= period:
            adx = sum(dx[-period:]) / period
        else:
            adx = sum(dx) / len(dx) if dx else 0

        return adx

    def _calculate_volatility(self, bars: List[Bar]) -> Tuple[float, float]:
        """计算波动率水平

        Returns:
            (ATR值, 相对波动率水平)
        """
        # 计算 ATR
        atr = self._calculate_atr(bars, self.config.atr_period)

        # 计算历史 ATR 中位数
        atrs = []
        for i in range(self.config.atr_period, len(bars)):
            window_bars = bars[max(0, i - self.config.atr_period):i+1]
            window_atr = self._calculate_atr(window_bars, self.config.atr_period)
            atrs.append(window_atr)

        if not atrs:
            return float(atr), 0.5

        median_atr = np.median(atrs)

        # 相对波动率（当前 ATR 与历史中位数的比值）
        if median_atr > 0:
            relative_volatility = float(atr) / float(median_atr)
        else:
            relative_volatility = 1.0

        return float(atr), relative_volatility

    def _calculate_atr(self, bars: List[Bar], period: int) -> Decimal:
        """计算 ATR"""
        if len(bars) < period + 1:
            # 数据不足，使用简单计算
            total_range = sum(float(b.high - b.low) for b in bars)
            return Decimal(str(total_range / len(bars))) if bars else Decimal("0")

        true_ranges = []
        for i in range(1, len(bars)):
            prev_close = float(bars[i-1].close)
            current_high = float(bars[i].high)
            current_low = float(bars[i].low)

            tr = max(
                current_high - current_low,
                abs(current_high - prev_close),
                abs(current_low - prev_close)
            )
            true_ranges.append(tr)

        recent_tr = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
        atr = sum(recent_tr) / len(recent_tr)

        return Decimal(str(atr))

    def _detect_range_bound(self, bars: List[Bar]) -> Tuple[bool, float]:
        """检测是否处于震荡区间

        Returns:
            (是否震荡, 震荡强度)
        """
        if len(bars) < self.config.range_period:
            return False, 0.0

        lookback_bars = bars[-self.config.range_period:]
        closes = [float(b.close) for b in lookback_bars]

        # 计算价格范围
        max_price = max(closes)
        min_price = min(closes)
        price_range = max_price - min_price
        avg_price = sum(closes) / len(closes)

        # 相对波动幅度
        relative_range = price_range / avg_price if avg_price > 0 else 0

        # 检查价格是否在区间内
        is_range_bound = relative_range < self.config.range_threshold

        # 震荡强度（1 - 相对波动幅度 / 阈值）
        if self.config.range_threshold > 0:
            range_strength = max(0, 1 - relative_range / self.config.range_threshold)
        else:
            range_strength = 0

        return is_range_bound, range_strength

    def _calculate_rsi(self, bars: List[Bar], period: int = 14) -> float:
        """计算 RSI"""
        if len(bars) < period + 1:
            return 50.0

        closes = [float(b.close) for b in bars]
        deltas = np.diff(closes)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _classify_regime(
        self,
        adx: float,
        volatility_level: float,
        is_range_bound: bool,
        range_strength: float,
        rsi: float
    ) -> Tuple[RegimeType, float]:
        """分类市场状态

        Args:
            adx: 趋势强度指标
            volatility_level: 相对波动率水平
            is_range_bound: 是否震荡
            range_strength: 震荡强度
            rsi: RSI 指标

        Returns:
            (市场状态类型, 置信度)
        """
        # 1. 首先判断是否有趋势
        has_trend = adx > self.config.adx_trend_threshold
        is_strong_trend = adx > self.config.adx_strong_trend

        # 2. 判断是否高波动
        is_high_vol = volatility_level > self.config.high_vol_threshold
        is_low_vol = volatility_level < (1.0 / self.config.high_vol_threshold)

        # 3. 综合判断
        if is_high_vol:
            # 高波动情况
            if has_trend:
                return RegimeType.HIGH_VOL, 0.8
            else:
                return RegimeType.HIGH_VOL, 0.7

        if is_strong_trend and not is_range_bound:
            # 强趋势
            confidence = min(0.5 + (adx - self.config.adx_trend_threshold) / 50, 0.95)
            return RegimeType.TREND, confidence

        if has_trend:
            # 有趋势但不够强
            if is_range_bound:
                # 有趋势但价格在区间内，可能是趋势早期或转折
                return RegimeType.RANGE, 0.6
            else:
                return RegimeType.TREND, 0.65

        if is_range_bound:
            # 震荡市
            confidence = min(0.5 + range_strength * 0.4, 0.9)
            return RegimeType.RANGE, confidence

        # 默认返回震荡市（低置信度）
        return RegimeType.RANGE, 0.4

    def reset_state(self):
        """重置状态"""
        self._state = None

    def get_state(self) -> Optional[RegimeState]:
        """获取当前状态"""
        return self._state


def is_regime_allowed(regime: RegimeType, strategy_type: str) -> bool:
    """判断策略是否允许在特定市场状态下运行

    Args:
        regime: 当前市场状态
        strategy_type: 策略类型 (ma, breakout, ml)

    Returns:
        是否允许运行
    """
    # 策略配置：定义每种策略允许的市场状态
    STRATEGY_REGIME_CONFIG = {
        "ma": {
            RegimeType.TREND: True,      # MA 策略在趋势市表现好
            RegimeType.RANGE: False,     # 震荡市禁用（避免频繁交易）
            RegimeType.HIGH_VOL: False,  # 高波动禁用
            RegimeType.LOW_VOL: True,    # 低波动可以运行
            RegimeType.UNKNOWN: False,   # 未知状态禁用
        },
        "breakout": {
            RegimeType.TREND: True,      # 突破策略在趋势市有效
            RegimeType.RANGE: False,     # 震荡市假突破多
            RegimeType.HIGH_VOL: True,   # 高波动可能有突破机会
            RegimeType.LOW_VOL: False,   # 低波动突破信号少
            RegimeType.UNKNOWN: False,
        },
        "ml": {
            RegimeType.TREND: True,
            RegimeType.RANGE: True,      # ML 可能学会适应震荡
            RegimeType.HIGH_VOL: False,  # 高波动预测困难
            RegimeType.LOW_VOL: True,
            RegimeType.UNKNOWN: True,    # ML 可以尝试预测
        }
    }

    return STRATEGY_REGIME_CONFIG.get(strategy_type, {}).get(regime, False)


def get_regime_position_modifier(regime: RegimeType, base_position: float) -> float:
    """根据市场状态调整目标仓位

    Args:
        regime: 当前市场状态
        base_position: 基础目标仓位

    Returns:
        调整后的仓位
    """
    # 仓位调整系数
    REGIME_POSITION_MODIFIER = {
        RegimeType.TREND: 1.0,          # 趋势市：满仓
        RegimeType.RANGE: 0.3,          # 震荡市：降权至 30%
        RegimeType.HIGH_VOL: 0.2,       # 高波动：降权至 20%
        RegimeType.LOW_VOL: 0.5,        # 低波动：降权至 50%
        RegimeType.UNKNOWN: 0.0,        # 未知：平仓
    }

    modifier = REGIME_POSITION_MODIFIER.get(regime, 0.0)
    return base_position * modifier


if __name__ == "__main__":
    """测试代码"""
    from datetime import timedelta

    print("=== RegimeDetector 测试 ===\n")

    # 创建测试数据
    base_price = 100.0
    bars = []
    base_date = datetime(2024, 1, 1)

    # 生成上涨趋势数据
    for i in range(50):
        date = base_date + timedelta(days=i)
        trend = i * 0.5  # 每天上涨 0.5
        noise = (i % 5 - 2) * 0.3  # 小幅波动

        open_price = base_price + trend + noise
        close_price = open_price + 0.2
        high_price = max(open_price, close_price) + 0.3
        low_price = min(open_price, close_price) - 0.2

        bars.append(Bar(
            symbol="TEST",
            timestamp=date,
            open=Decimal(str(open_price)),
            high=Decimal(str(high_price)),
            low=Decimal(str(low_price)),
            close=Decimal(str(close_price)),
            volume=1000000,
            interval="1d"
        ))

    # 创建检测器
    detector = RegimeDetector()

    # 检测市场状态
    regime_info = detector.detect(bars)

    print("1. 趋势市检测结果:")
    print(f"   市场状态: {regime_info.regime.value}")
    print(f"   置信度: {regime_info.confidence:.2f}")
    print(f"   ADX: {regime_info.adx:.2f}")
    print(f"   趋势强度: {regime_info.trend_strength:.2f}")
    print(f"   波动率水平: {regime_info.volatility_level:.2f}")
    print(f"   ATR: {regime_info.atr}")

    # 测试带状态的检测
    print("\n2. 带状态的检测:")
    for i in range(5):
        test_bars = bars[:30+i*5]
        regime_state = detector.detect_with_state(test_bars)
        print(f"   Day {30+i*5}: {regime_state.current.regime.value}, "
              f"持续 {regime_state.regime_duration} 天, "
              f"稳定性 {regime_state.regime_stability:.2f}")

    # 测试策略 gating
    print("\n3. 策略 Gating:")
    for regime in RegimeType:
        for strategy in ["ma", "breakout", "ml"]:
            allowed = is_regime_allowed(regime, strategy)
            print(f"   {regime.value:10s} -> {strategy:8s}: {'允许' if allowed else '禁止'}")

    # 测试仓位调整
    print("\n4. 仓位调整:")
    for regime in RegimeType:
        modified = get_regime_position_modifier(regime, 1.0)
        print(f"   {regime.value:10s}: 1.0 -> {modified:.2f}")

    # 生成震荡数据测试
    print("\n5. 震荡市检测结果:")
    range_bars = []
    for i in range(50):
        date = base_date + timedelta(days=i)
        # 价格在 98-102 之间震荡
        base = 100
        oscillation = 2 * (i % 10 / 5 - 1)  # -2 到 +2

        close_price = base + oscillation
        open_price = close_price + ((-1) ** i) * 0.2
        high_price = max(open_price, close_price) + 0.2
        low_price = min(open_price, close_price) - 0.2

        range_bars.append(Bar(
            symbol="TEST",
            timestamp=date,
            open=Decimal(str(open_price)),
            high=Decimal(str(high_price)),
            low=Decimal(str(low_price)),
            close=Decimal(str(close_price)),
            volume=1000000,
            interval="1d"
        ))

    detector.reset_state()
    range_regime = detector.detect(range_bars)
    print(f"   市场状态: {range_regime.regime.value}")
    print(f"   置信度: {range_regime.confidence:.2f}")
    print(f"   ADX: {range_regime.adx:.2f}")

    print("\n✓ 所有测试通过")
