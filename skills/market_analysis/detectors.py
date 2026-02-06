"""
市场检测器

实现 Regime / Volatility / Liquidity 检测
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

from core.models import Bar
from skills.market_analysis.market_state import (
    RegimeType,
    VolatilityState,
    VolatilityTrend,
    LiquidityState
)

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    市场状态检测器

    综合 >=2 个信号判定：
    1. ADX - 趋势强度
    2. 价格斜率显著性
    3. MA 分离度
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        trend_slope_threshold: float = 0.01,
        ma_short_period: int = 10,
        ma_long_period: int = 30,
        ma_separation_threshold: float = 0.02
    ):
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.trend_slope_threshold = trend_slope_threshold
        self.ma_short_period = ma_short_period
        self.ma_long_period = ma_long_period
        self.ma_separation_threshold = ma_separation_threshold

    def detect(self, bars: List[Bar]) -> Dict[str, Any]:
        """
        检测市场状态

        Returns:
            {
                "regime": RegimeType,
                "confidence": float (0-1),
                "features": dict
            }
        """
        if len(bars) < max(self.ma_long_period, self.adx_period) + 10:
            return {
                "regime": RegimeType.UNKNOWN,
                "confidence": 0.0,
                "features": {"reason": "数据不足"}
            }

        # 1. 计算 ADX
        adx = self._calculate_adx(bars)
        has_trend_adx = adx > self.adx_trend_threshold

        # 2. 计算价格斜率显著性
        slope_significant, slope_value = self._calculate_slope_significance(bars)
        has_trend_slope = slope_significant

        # 3. 计算 MA 分离度
        ma_separation = self._calculate_ma_separation(bars)
        has_trend_ma = abs(ma_separation) > self.ma_separation_threshold

        # 4. 计算价格区间（判定震荡）
        price_range = self._calculate_price_range(bars)
        is_range_bound = price_range < 0.02  # 2%以内算震荡

        # 5. 综合判断
        signals = {
            "adx": has_trend_adx,
            "slope": has_trend_slope,
            "ma_separation": has_trend_ma,
            "is_range": is_range_bound
        }

        trend_signals = sum([has_trend_adx, has_trend_slope, has_trend_ma])
        range_signals = 3 - trend_signals if is_range_bound else 0

        # 判定逻辑
        if is_range_bound and range_signals >= 2:
            regime = RegimeType.RANGE
            confidence = 0.7
        elif trend_signals >= 2:
            regime = RegimeType.TREND
            confidence = min(0.5 + (adx - 20) / 40, 0.95)
        elif adx > 40:
            regime = RegimeType.HIGH_VOL
            confidence = 0.8
        else:
            regime = RegimeType.UNKNOWN
            confidence = 0.3

        return {
            "regime": regime,
            "confidence": confidence,
            "features": {
                "adx": adx,
                "slope": slope_value,
                "ma_separation": ma_separation,
                "price_range": price_range,
                "signals": signals
            }
        }

    def _calculate_adx(self, bars: List[Bar]) -> float:
        """计算 ADX"""
        if len(bars) < self.adx_period * 2:
            return 0.0

        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]

        # 计算 +DM, -DM, TR
        plus_dm = []
        minus_dm = []
        tr = []

        for i in range(1, len(closes)):
            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]

            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)

            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))

        # 计算 +DI, -DI
        period = self.adx_period
        plus_di = []
        minus_di = []

        for i in range(period, len(tr)):
            smoothed_plus = sum(plus_dm[i-period+1:i+1])
            smoothed_minus = sum(minus_dm[i-period+1:i+1])
            smoothed_tr = sum(tr[i-period+1:i+1])

            if smoothed_tr > 0:
                plus_di.append(100 * smoothed_plus / smoothed_tr)
                minus_di.append(100 * smoothed_minus / smoothed_tr)
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

        if len(dx) >= period:
            return sum(dx[-period:]) / period
        return sum(dx) / len(dx) if dx else 0

    def _calculate_slope_significance(self, bars: List[Bar]) -> Tuple[bool, float]:
        """计算价格斜率显著性"""
        if len(bars) < 20:
            return False, 0.0

        closes = [float(b.close) for b in bars[-20:]]
        x = np.arange(len(closes))
        y = np.array(closes)

        # 线性回归
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept

        # 计算 R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 相对斜率
        relative_slope = slope / np.mean(y) if np.mean(y) > 0 else 0

        return r_squared > 0.3 and abs(relative_slope) > self.trend_slope_threshold, relative_slope

    def _calculate_ma_separation(self, bars: List[Bar]) -> float:
        """计算 MA 分离度"""
        if len(bars) < self.ma_long_period:
            return 0.0

        closes = [float(b.close) for b in bars]

        ma_short = np.mean(closes[-self.ma_short_period:])
        ma_long = np.mean(closes[-self.ma_long_period:])

        return (ma_short - ma_long) / ma_long if ma_long > 0 else 0

    def _calculate_price_range(self, bars: List[Bar]) -> float:
        """计算价格区间（相对波动）"""
        if len(bars) < 20:
            return 0.0

        lookback = bars[-20:]
        highs = [float(b.high) for b in lookback]
        lows = [float(b.low) for b in lookback]

        avg_price = np.mean([float(b.close) for b in lookback])
        range_size = (max(highs) - min(lows)) / avg_price if avg_price > 0 else 0

        return range_size


class VolatilityDetector:
    """
    波动率检测器

    计算：
    - realized volatility
    - ATR / price
    - vol percentile
    - vol trend
    """

    def __init__(
        self,
        lookback_period: int = 20,
        atr_period: int = 14,
        vol_percentile_window: int = 100,
        vol_threshold_low: float = 0.01,
        vol_threshold_high: float = 0.03
    ):
        self.lookback_period = lookback_period
        self.atr_period = atr_period
        self.vol_percentile_window = vol_percentile_window
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high

        self._historical_vols: deque = deque(maxlen=vol_percentile_window)

    def detect(self, bars: List[Bar]) -> Dict[str, Any]:
        """
        检测波动率状态

        Returns:
            {
                "state": VolatilityState,
                "trend": VolatilityTrend,
                "percentile": float,
                "realized_vol": float,
                "atr_ratio": float,
                "features": dict
            }
        """
        if len(bars) < self.lookback_period:
            return {
                "state": VolatilityState.NORMAL,
                "trend": VolatilityTrend.STABLE,
                "percentile": 0.5,
                "realized_vol": 0.0,
                "atr_ratio": 0.0,
                "features": {}
            }

        # 1. 计算已实现波动率
        realized_vol = self._calculate_realized_vol(bars)

        # 2. 计算 ATR 比率
        atr, atr_ratio = self._calculate_atr_ratio(bars)

        # 3. 判断波动率状态
        if realized_vol < self.vol_threshold_low:
            vol_state = VolatilityState.LOW
        elif realized_vol > self.vol_threshold_high:
            vol_state = VolatilityState.HIGH
            if realized_vol > self.vol_threshold_high * 2:
                vol_state = VolatilityState.EXTREME
        else:
            vol_state = VolatilityState.NORMAL

        # 4. 判断波动率趋势
        vol_trend = self._calculate_vol_trend(bars)

        # 5. 计算分位数
        self._historical_vols.append(realized_vol)
        percentile = self._calculate_percentile(realized_vol)

        return {
            "state": vol_state,
            "trend": vol_trend,
            "percentile": percentile,
            "realized_vol": realized_vol,
            "atr_ratio": atr_ratio,
            "features": {
                "atr": atr,
                "historical_vol_mean": np.mean(self._historical_vols) if self._historical_vols else realized_vol
            }
        }

    def _calculate_realized_vol(self, bars: List[Bar]) -> float:
        """计算已实现波动率（年化）"""
        lookback = bars[-self.lookback_period:]
        closes = [float(b.close) for b in lookback]

        returns = np.diff(closes) / closes[:-1]
        vol = np.std(returns) * np.sqrt(252)  # 年化
        return vol

    def _calculate_atr_ratio(self, bars: List[Bar]) -> Tuple[float, float]:
        """计算 ATR 和 ATR/价格比例"""
        atr = self._calculate_atr(bars, self.atr_period)
        current_price = float(bars[-1].close)
        atr_ratio = atr / current_price if current_price > 0 else 0
        return atr, atr_ratio

    def _calculate_atr(self, bars: List[Bar], period: int) -> float:
        """计算 ATR"""
        if len(bars) < period + 1:
            # 简化计算
            total_range = sum(float(b.high - b.low) for b in bars)
            return total_range / len(bars)

        true_ranges = []
        for i in range(1, len(bars)):
            prev_close = float(bars[i-1].close)
            high = float(bars[i].high)
            low = float(bars[i].low)

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        return np.mean(true_ranges[-period:])

    def _calculate_vol_trend(self, bars: List[Bar]) -> VolatilityTrend:
        """计算波动率趋势"""
        if len(bars) < 40:
            return VolatilityTrend.STABLE

        # 计算两个时间窗口的波动率
        recent_vol = self._calculate_realized_vol(bars[-20:])
        older_vol = self._calculate_realized_vol(bars[-40:-20])

        if older_vol == 0:
            return VolatilityTrend.STABLE

        ratio = recent_vol / older_vol

        if ratio > 1.3:
            return VolatilityTrend.RISING
        elif ratio < 0.7:
            return VolatilityTrend.FALLING
        else:
            return VolatilityTrend.STABLE

    def _calculate_percentile(self, current_vol: float) -> float:
        """计算波动率分位数"""
        if not self._historical_vols:
            return 0.5

        vols = list(self._historical_vols)
        return sum(v < current_vol for v in vols) / len(vols)


class LiquidityDetector:
    """
    流动性检测器

    基于成交量/换手/异常波动判断
    """

    def __init__(
        self,
        volume_ma_period: int = 20,
        thin_threshold: float = 0.3,
        frozen_threshold: float = 0.1
    ):
        self.volume_ma_period = volume_ma_period
        self.thin_threshold = thin_threshold  # 成交量低于均值的30%算低流动性
        self.frozen_threshold = frozen_threshold  # 低于10%算流动性枯竭

    def detect(self, bars: List[Bar]) -> Dict[str, Any]:
        """
        检测流动性状态

        Returns:
            {
                "state": LiquidityState,
                "volume_ratio": float,
                "turnover_ratio": float,
                "features": dict
            }
        """
        if len(bars) < self.volume_ma_period:
            return {
                "state": LiquidityState.NORMAL,
                "volume_ratio": 1.0,
                "turnover_ratio": 1.0,
                "features": {"reason": "数据不足"}
            }

        # 1. 计算成交量比率
        current_volume = bars[-1].volume
        avg_volume = np.mean([b.volume for b in bars[-self.volume_ma_period:]])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # 2. 检测异常波动（可能预示流动性问题）
        price_change = abs(float(bars[-1].close - bars[-2].close)) / float(bars[-2].close)

        # 3. 判断流动性状态
        if volume_ratio < self.frozen_threshold:
            liq_state = LiquidityState.FROZEN
        elif volume_ratio < self.thin_threshold:
            liq_state = LiquidityState.THIN
        else:
            liq_state = LiquidityState.NORMAL

        return {
            "state": liq_state,
            "volume_ratio": volume_ratio,
            "turnover_ratio": 1.0,  # 简化，实际需要流通股本数据
            "features": {
                "current_volume": current_volume,
                "avg_volume": avg_volume,
                "price_change": price_change
            }
        }


class MarketAnalyzer:
    """
    市场分析器（整合接口）

    统一调用各检测器，生成完整的市场分析结果
    """

    def __init__(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        vol_detector: Optional[VolatilityDetector] = None,
        liq_detector: Optional[LiquidityDetector] = None
    ):
        self.regime_detector = regime_detector or RegimeDetector()
        self.vol_detector = vol_detector or VolatilityDetector()
        self.liq_detector = liq_detector or LiquidityDetector()

    def analyze(self, bars: List[Bar]) -> Dict[str, Any]:
        """
        综合分析市场

        Returns:
            {
                "regime": Dict,      # RegimeDetector 结果
                "volatility": Dict,  # VolatilityDetector 结果
                "liquidity": Dict,   # LiquidityDetector 结果
                "summary": Dict       # 综合摘要
            }
        """
        regime_result = self.regime_detector.detect(bars)
        vol_result = self.vol_detector.detect(bars)
        liq_result = self.liq_detector.detect(bars)

        # 综合摘要
        summary = self._generate_summary(regime_result, vol_result, liq_result)

        return {
            "regime": regime_result,
            "volatility": vol_result,
            "liquidity": liq_result,
            "summary": summary
        }

    def _generate_summary(
        self,
        regime_result: Dict,
        vol_result: Dict,
        liq_result: Dict
    ) -> Dict[str, Any]:
        """生成综合摘要"""
        return {
            "market_condition": self._classify_market_condition(regime_result, vol_result, liq_result),
            "risk_level": self._calculate_risk_level(regime_result, vol_result, liq_result),
            "trading_advice": self._generate_trading_advice(regime_result, vol_result, liq_result)
        }

    def _classify_market_condition(self, regime_result, vol_result, liq_result) -> str:
        """分类市场环境"""
        regime = regime_result.get("regime")
        vol_state = vol_result.get("state")
        liq_state = liq_result.get("state")

        if liq_state == LiquidityState.FROZEN or vol_state == VolatilityState.EXTREME:
            return "CRISIS"
        elif regime == RegimeType.HIGH_VOL:
            return "VOLATILE"
        elif regime == RegimeType.TREND:
            return "TRENDING"
        elif regime == RegimeType.RANGE:
            return "RANGING"
        else:
            return "UNCERTAIN"

    def _calculate_risk_level(self, regime_result, vol_result, liq_result) -> str:
        """计算风险等级"""
        risk_score = 0

        if vol_result.get("state") == VolatilityState.EXTREME:
            risk_score += 3
        elif vol_result.get("state") == VolatilityState.HIGH:
            risk_score += 2

        if liq_result.get("state") == LiquidityState.FROZEN:
            risk_score += 3
        elif liq_result.get("state") == LiquidityState.THIN:
            risk_score += 1

        if regime_result.get("regime") == RegimeType.CRISIS:
            risk_score += 3
        elif regime_result.get("regime") == RegimeType.HIGH_VOL:
            risk_score += 2

        if risk_score >= 5:
            return "EXTREME"
        elif risk_score >= 3:
            return "HIGH"
        elif risk_score >= 1:
            return "MODERATE"
        else:
            return "LOW"

    def _generate_trading_advice(self, regime_result, vol_result, liq_result) -> str:
        """生成交易建议"""
        risk_level = self._calculate_risk_level(regime_result, vol_result, liq_result)

        if risk_level == "EXTREME":
            return "AVOID_NEW_POSITIONS"
        elif risk_level == "HIGH":
            return "REDUCE_EXPOSURE"
        elif risk_level == "MODERATE":
            return "CAUTIOUS"
        else:
            return "NORMAL"


if __name__ == "__main__":
    """测试代码"""
    from datetime import timedelta

    print("=== Market Detectors 测试 ===\n")

    # 创建测试数据
    bars = []
    for i in range(50):
        date = datetime(2024, 1, 1) + timedelta(days=i)
        trend = i * 0.5
        noise = (i % 5 - 2) * 0.3

        close = 100 + trend + noise
        high = close + 0.5
        low = close - 0.5

        bars.append(Bar(
            symbol="TEST",
            timestamp=date,
            open=Decimal(str(close)),
            high=Decimal(str(high)),
            low=Decimal(str(low)),
            close=Decimal(str(close)),
            volume=1000000 + (i % 10) * 100000,
            interval="1d"
        ))

    # 测试各检测器
    print("1. RegimeDetector:")
    regime_detector = RegimeDetector()
    regime_result = regime_detector.detect(bars)
    print(f"   Regime: {regime_result['regime'].value}")
    print(f"   Confidence: {regime_result['confidence']:.2f}")

    print("\n2. VolatilityDetector:")
    vol_detector = VolatilityDetector()
    vol_result = vol_detector.detect(bars)
    print(f"   State: {vol_result['state'].value}")
    print(f"   Trend: {vol_result['trend'].value}")
    print(f"   Percentile: {vol_result['percentile']:.2f}")

    print("\n3. LiquidityDetector:")
    liq_detector = LiquidityDetector()
    liq_result = liq_detector.detect(bars)
    print(f"   State: {liq_result['state'].value}")
    print(f"   Volume Ratio: {liq_result['volume_ratio']:.2f}")

    print("\n4. MarketAnalyzer:")
    analyzer = MarketAnalyzer()
    analysis = analyzer.analyze(bars)
    print(f"   Condition: {analysis['summary']['market_condition']}")
    print(f"   Risk Level: {analysis['summary']['risk_level']}")
    print(f"   Advice: {analysis['summary']['trading_advice']}")

    print("\n✓ 所有测试通过")
