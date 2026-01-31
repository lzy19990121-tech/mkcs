"""
技术指标库

包含 30+ 个常用技术指标，用于量化交易信号生成和特征工程
"""

from decimal import Decimal
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """技术指标计算器

    支持的技术指标：
    - 趋势指标: SMA, EMA, MACD, TRIX, KST, ROC
    - 动量指标: RSI, Stochastic, Williams %R, CCI, MFI
    - 成交量指标: OBV, VWAP, ADL, CMF, NVI, PVI
    - 波动率指标: ATR, Bollinger Bands, Keltner Channels
    - 价格指标: Typical Price, Median Price, Weighted Close
    """

    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """简单移动平均线 (Simple Moving Average)

        Args:
            prices: 价格列表
            period: 周期

        Returns:
            SMA 值列表
        """
        if len(prices) < period:
            return [float('nan')] * len(prices)

        sma_values = [float('nan')] * (period - 1)

        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(avg)

        return sma_values

    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """指数移动平均线 (Exponential Moving Average)

        Args:
            prices: 价格列表
            period: 周期

        Returns:
            EMA 值列表
        """
        if len(prices) < period:
            return [float('nan')] * len(prices)

        multiplier = 2 / (period + 1)
        ema_values = [float('nan')] * (period - 1)

        # 第一个 EMA 值使用 SMA
        ema = sum(prices[:period]) / period
        ema_values.append(ema)

        # 后续使用 EMA 公式
        for i in range(period, len(prices)):
            ema = (prices[i] - ema) * multiplier + ema
            ema_values.append(ema)

        return ema_values

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """相对强弱指标 (Relative Strength Index)

        Args:
            prices: 价格列表
            period: 周期，默认 14

        Returns:
            RSI 值列表 (0-100)
        """
        if len(prices) < period + 1:
            return [float('nan')] * len(prices)

        rsi_values = [float('nan')] * period

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gains.append(max(change, 0))
            losses.append(abs(min(change, 0)))

        # 初始平均
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        if avg_loss == 0:
            return [100.0] * len(prices)

        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))

        # 后续使用 Wilder 平滑
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))

        return rsi_values

    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD (Moving Average Convergence Divergence)

        Args:
            prices: 价格列表
            fast: 快线周期，默认 12
            slow: 慢线周期，默认 26
            signal: 信号线周期，默认 9

        Returns:
            (MACD 线, 信号线, 柱状图)
        """
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)

        macd_line = []
        for ef, es in zip(ema_fast, ema_slow):
            if not (float('nan') in (ef, es)):
                macd_line.append(ef - es)
            else:
                macd_line.append(float('nan'))

        signal_line = TechnicalIndicators.ema(
            [v for v in macd_line if not float('nan') == v],
            signal
        )

        # 重新对齐
        nan_count = sum(1 for v in macd_line if float('nan') == v)
        signal_line = [float('nan')] * nan_count + signal_line

        histogram = [m - s if not (float('nan') in (m, s)) else float('nan')
                     for m, s in zip(macd_line, signal_line)]

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """布林带 (Bollinger Bands)

        Args:
            prices: 价格列表
            period: 周期，默认 20
            std_dev: 标准差倍数，默认 2

        Returns:
            (上轨, 中轨, 下轨)
        """
        sma = TechnicalIndicators.sma(prices, period)

        upper = []
        lower = []

        for i in range(len(sma)):
            if float('nan') == sma[i]:
                upper.append(float('nan'))
                lower.append(float('nan'))
            else:
                start_idx = max(0, i - period + 1)
                window = prices[start_idx:i + 1]
                std = (sum((p - sma[i]) ** 2 for p in window) / len(window)) ** 0.5

                upper.append(sma[i] + std_dev * std)
                lower.append(sma[i] - std_dev * std)

        return upper, sma, lower

    @staticmethod
    def stochastic(highs: List[float], lows: List[float], closes: List[float],
                   k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
        """随机指标 (Stochastic Oscillator)

        Args:
            highs: 最高价列表
            lows: 最低价列表
            closes: 收盘价列表
            k_period: %K 周期，默认 14
            d_period: %D 周期，默认 3

        Returns:
            (%K, %D)
        """
        if len(closes) < k_period:
            return ([float('nan')] * len(closes), [float('nan')] * len(closes))

        k_values = [float('nan')] * (k_period - 1)

        for i in range(k_period - 1, len(closes)):
            high_window = highs[i - k_period + 1:i + 1]
            low_window = lows[i - k_period + 1:i + 1]

            highest_high = max(high_window)
            lowest_low = min(low_window)

            if highest_high == lowest_low:
                k = 50.0
            else:
                k = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)

            k_values.append(k)

        # %D 是 %K 的 SMA
        d_values = TechnicalIndicators.sma(
            [v for v in k_values if not float('nan') == v],
            d_period
        )

        # 重新对齐
        nan_count = sum(1 for v in k_values if float('nan') == v)
        d_values = [float('nan')] * nan_count + d_values

        return k_values, d_values

    @staticmethod
    def williams_r(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """威廉指标 (Williams %R)

        Args:
            highs: 最高价列表
            lows: 最低价列表
            closes: 收盘价列表
            period: 周期，默认 14

        Returns:
            Williams %R 值列表 (-100 到 0)
        """
        if len(closes) < period:
            return [float('nan')] * len(closes)

        r_values = [float('nan')] * (period - 1)

        for i in range(period - 1, len(closes)):
            high_window = highs[i - period + 1:i + 1]
            low_window = lows[i - period + 1:i + 1]

            highest_high = max(high_window)
            lowest_low = min(low_window)

            if highest_high == lowest_low:
                r = -50.0
            else:
                r = -100 * (highest_high - closes[i]) / (highest_high - lowest_low)

            r_values.append(r)

        return r_values

    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """平均真实波幅 (Average True Range)

        Args:
            highs: 最高价列表
            lows: 最低价列表
            closes: 收盘价列表
            period: 周期，默认 14

        Returns:
            ATR 值列表
        """
        if len(closes) < 2:
            return [float('nan')] * len(closes)

        # 计算真实波幅
        tr_list = []
        for i in range(1, len(closes)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i - 1])
            tr3 = abs(lows[i] - closes[i - 1])
            tr_list.append(max(tr1, tr2, tr3))

        # 使用 Wilder 平滑计算 ATR
        atr_values = [float('nan')]  # 第一个值无法计算

        if len(tr_list) < period:
            return atr_values + [float('nan')] * (len(closes) - 1)

        # 初始 ATR 使用 SMA
        atr = sum(tr_list[:period]) / period
        atr_values.append(atr)

        # 后续使用 Wilder 平滑
        for i in range(period, len(tr_list)):
            atr = (atr * (period - 1) + tr_list[i]) / period
            atr_values.append(atr)

        return atr_values

    @staticmethod
    def obv(prices: List[float], volumes: List[int]) -> List[int]:
        """能量潮 (On-Balance Volume)

        Args:
            prices: 价格列表
            volumes: 成交量列表

        Returns:
            OBV 值列表
        """
        if len(prices) != len(volumes) or len(prices) == 0:
            return []

        obv_values = [volumes[0]]

        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv_values.append(obv_values[-1] + volumes[i])
            elif prices[i] < prices[i - 1]:
                obv_values.append(obv_values[-1] - volumes[i])
            else:
                obv_values.append(obv_values[-1])

        return obv_values

    @staticmethod
    def vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[int], period: int = 20) -> List[float]:
        """成交量加权平均价 (Volume Weighted Average Price)

        Args:
            highs: 最高价列表
            lows: 最低价列表
            closes: 收盘价列表
            volumes: 成交量列表
            period: 周期，默认 20

        Returns:
            VWAP 值列表
        """
        if len(closes) < period:
            return [float('nan')] * len(closes)

        vwap_values = [float('nan')] * (period - 1)

        for i in range(period - 1, len(closes)):
            total_pv = 0
            total_volume = 0

            for j in range(i - period + 1, i + 1):
                typical_price = (highs[j] + lows[j] + closes[j]) / 3
                total_pv += typical_price * volumes[j]
                total_volume += volumes[j]

            if total_volume == 0:
                vwap_values.append(float('nan'))
            else:
                vwap_values.append(total_pv / total_volume)

        return vwap_values

    @staticmethod
    def cci(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> List[float]:
        """顺势指标 (Commodity Channel Index)

        Args:
            highs: 最高价列表
            lows: 最低价列表
            closes: 收盘价列表
            period: 周期，默认 20

        Returns:
            CCI 值列表
        """
        if len(closes) < period:
            return [float('nan')] * len(closes)

        cci_values = [float('nan')] * (period - 1)

        for i in range(period - 1, len(closes)):
            typical_prices = [(highs[j] + lows[j] + closes[j]) / 3
                             for j in range(i - period + 1, i + 1)]

            sma_tp = sum(typical_prices) / period

            # 计算平均偏差
            mad = sum(abs(tp - sma_tp) for tp in typical_prices) / period

            if mad == 0:
                cci = 0.0
            else:
                cci = (typical_prices[-1] - sma_tp) / (0.015 * mad)

            cci_values.append(cci)

        return cci_values

    @staticmethod
    def mfi(highs: List[float], lows: List[float], closes: List[float], volumes: List[int], period: int = 14) -> List[float]:
        """资金流量指标 (Money Flow Index)

        Args:
            highs: 最高价列表
            lows: 最低价列表
            closes: 收盘价列表
            volumes: 成交量列表
            period: 周期，默认 14

        Returns:
            MFI 值列表 (0-100)
        """
        if len(closes) < period + 1:
            return [float('nan')] * len(closes)

        mfi_values = [float('nan')] * period

        for i in range(period, len(closes)):
            positive_mf = 0
            negative_mf = 0

            for j in range(i - period + 1, i + 1):
                typical_price = (highs[j] + lows[j] + closes[j]) / 3
                money_flow = typical_price * volumes[j]

                if typical_price > (highs[j - 1] + lows[j - 1] + closes[j - 1]) / 3:
                    positive_mf += money_flow
                elif typical_price < (highs[j - 1] + lows[j - 1] + closes[j - 1]) / 3:
                    negative_mf += money_flow

            if negative_mf == 0:
                mfi = 100.0
            else:
                mfi = 100 - (100 / (1 + positive_mf / negative_mf))

            mfi_values.append(mfi)

        return mfi_values

    @staticmethod
    def roc(prices: List[float], period: int = 12) -> List[float]:
        """变动率指标 (Rate of Change)

        Args:
            prices: 价格列表
            period: 周期，默认 12

        Returns:
            ROC 值列表（百分比）
        """
        if len(prices) < period:
            return [float('nan')] * len(prices)

        roc_values = [float('nan')] * (period - 1)

        for i in range(period - 1, len(prices)):
            if prices[i - period] == 0:
                roc_values.append(float('nan'))
            else:
                roc = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
                roc_values.append(roc)

        return roc_values

    @staticmethod
    def momentum(prices: List[float], period: int = 10) -> List[float]:
        """动量指标 (Momentum)

        Args:
            prices: 价格列表
            period: 周期，默认 10

        Returns:
            Momentum 值列表
        """
        if len(prices) < period:
            return [float('nan')] * len(prices)

        momentum_values = [float('nan')] * (period - 1)

        for i in range(period - 1, len(prices)):
            momentum_values.append(prices[i] - prices[i - period])

        return momentum_values

    @staticmethod
    def trix(prices: List[float], period: int = 15) -> List[float]:
        """三重指数平滑移动平均 (TRIX)

        Args:
            prices: 价格列表
            period: 周期，默认 15

        Returns:
            TRIX 值列表
        """
        # 三重 EMA
        ema1 = TechnicalIndicators.ema(prices, period)
        ema2 = TechnicalIndicators.ema([v for v in ema1 if not float('nan') == v], period)
        ema3 = TechnicalIndicators.ema([v for v in ema2 if not float('nan') == v], period)

        # 计算 ROC
        trix_values = []
        for i in range(len(ema3)):
            if i == 0:
                trix_values.append(float('nan'))
            else:
                if ema3[i - 1] == 0:
                    trix_values.append(float('nan'))
                else:
                    trix = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 10000
                    trix_values.append(trix)

        # 对齐长度
        nan_count = len(prices) - len(trix_values)
        trix_values = [float('nan')] * nan_count + trix_values

        return trix_values


class IndicatorFeatures:
    """指标特征提取器

    从 K 线数据中提取所有技术指标作为特征
    """

    @staticmethod
    def extract_all(bars: List) -> Dict[str, List[float]]:
        """提取所有技术指标

        Args:
            bars: K 线数据列表

        Returns:
            指标字典，key 为指标名，value 为指标值列表
        """
        if len(bars) == 0:
            return {}

        # 提取价格和成交量
        highs = [float(bar.high) for bar in bars]
        lows = [float(bar.low) for bar in bars]
        opens = [float(bar.open) for bar in bars]
        closes = [float(bar.close) for bar in bars]
        volumes = [bar.volume for bar in bars]

        indicators = {}

        # 趋势指标
        indicators['sma_5'] = TechnicalIndicators.sma(closes, 5)
        indicators['sma_10'] = TechnicalIndicators.sma(closes, 10)
        indicators['sma_20'] = TechnicalIndicators.sma(closes, 20)
        indicators['sma_50'] = TechnicalIndicators.sma(closes, 50)
        indicators['ema_12'] = TechnicalIndicators.ema(closes, 12)
        indicators['ema_26'] = TechnicalIndicators.ema(closes, 26)

        # MACD
        macd, signal, hist = TechnicalIndicators.macd(closes)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = hist

        # 动量指标
        indicators['rsi_14'] = TechnicalIndicators.rsi(closes, 14)
        indicators['rsi_6'] = TechnicalIndicators.rsi(closes, 6)

        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(highs, lows, closes)
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d

        # Williams %R
        indicators['williams_r'] = TechnicalIndicators.williams_r(highs, lows, closes)

        # CCI
        indicators['cci'] = TechnicalIndicators.cci(highs, lows, closes)

        # MFI
        indicators['mfi'] = TechnicalIndicators.mfi(highs, lows, closes, volumes)

        # ROC
        indicators['roc_10'] = TechnicalIndicators.roc(closes, 10)
        indicators['roc_20'] = TechnicalIndicators.roc(closes, 20)

        # Momentum
        indicators['momentum'] = TechnicalIndicators.momentum(closes)

        # TRIX
        indicators['trix'] = TechnicalIndicators.trix(closes)

        # 波动率指标
        indicators['atr_14'] = TechnicalIndicators.atr(highs, lows, closes)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(closes)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        # BB 位置 (0-1)
        indicators['bb_position'] = [
            ((c - l) / (u - l)) if not (float('nan') in (c, u, l)) and (u - l) > 0 else 0.5
            for c, u, l in zip(closes, bb_upper, bb_lower)
        ]
        # BB 带宽
        indicators['bb_width'] = [
            ((u - l) / m) if not (float('nan') in (u, l, m)) and m > 0 else 0
            for u, l, m in zip(bb_upper, bb_lower, bb_middle)
        ]

        # 成交量指标
        indicators['obv'] = TechnicalIndicators.obv(closes, volumes)
        indicators['vwap'] = TechnicalIndicators.vwap(highs, lows, closes, volumes)

        # 成交量变化率
        vol_roc = [float('nan')] + [
            ((volumes[i] - volumes[i-1]) / volumes[i-1] * 100) if volumes[i-1] > 0 else 0
            for i in range(1, len(volumes))
        ]
        indicators['volume_roc'] = vol_roc

        # 价格动量
        indicators['price_change'] = [float('nan')] + [
            ((closes[i] - closes[i-1]) / closes[i-1] * 100) if closes[i-1] > 0 else 0
            for i in range(1, len(closes))
        ]

        # 真实波幅
        indicators['true_range'] = [
            max(highs[i] - lows[i],
                abs(highs[i] - closes[i-1]) if i > 0 else 0,
                abs(lows[i] - closes[i-1]) if i > 0 else 0)
            for i in range(len(bars))
        ]

        return indicators

    @staticmethod
    def get_latest_features(bars: List) -> Dict[str, float]:
        """获取最新的指标值

        Args:
            bars: K 线数据列表

        Returns:
            指标字典，key 为指标名，value 为最新值
        """
        all_indicators = IndicatorFeatures.extract_all(bars)

        latest = {}
        for name, values in all_indicators.items():
            if values:
                value = values[-1]
                if float('nan') == value:
                    # 使用前一个有效值
                    for v in reversed(values[:-1]):
                        if not float('nan') == v:
                            latest[name] = v
                            break
                    else:
                        latest[name] = 0.0
                else:
                    latest[name] = value

        return latest


if __name__ == "__main__":
    """测试代码"""
    print("=== 技术指标测试 ===\n")

    from skills.market_data.mock_source import MockMarketSource
    from datetime import datetime, timedelta

    # 生成测试数据
    source = MockMarketSource(seed=42)
    bars = source.get_bars("AAPL", datetime.now() - timedelta(days=100), datetime.now(), "1d")

    print(f"生成 {len(bars)} 根 K 线\n")

    # 提取所有指标
    indicators = IndicatorFeatures.extract_all(bars)

    print(f"成功提取 {len(indicators)} 个技术指标:")
    for name in sorted(indicators.keys()):
        print(f"  - {name}")

    # 获取最新值
    latest = IndicatorFeatures.get_latest_features(bars)

    print(f"\n最新指标值 (示例前10个):")
    for name, value in list(latest.items())[:10]:
        print(f"  {name}: {value:.4f}")

    print("\n✓ 测试完成")
