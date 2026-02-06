"""
增强的突破策略

支持：
- 波动率归一化突破阈值
- 突破确认条件（成交量、波动扩张、收盘位置）
- 失败突破识别
- 市场状态感知
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
import logging

from core.models import Bar, Position
from core.strategy_models import (
    StrategySignal,
    RegimeType,
    RegimeInfo,
    RiskHints
)
from skills.strategy.enhanced_base import EnhancedStrategy
from skills.analysis.regime_detector import RegimeDetector, RegimeDetectorConfig
from skills.analysis.dynamic_stops import DynamicStopManager, ExitReason

logger = logging.getLogger(__name__)


class BreakoutConfig:
    """突破策略配置"""

    def __init__(
        self,
        period: int = 20,
        # 波动率归一化参数
        use_atr_threshold: bool = True,
        atr_multiplier: float = 1.5,
        price_threshold_pct: float = 0.01,  # 固定百分比阈值（备用）

        # 确认条件
        require_volume_confirm: bool = True,
        volume_ratio_threshold: float = 1.5,  # 成交量倍数

        require_volatility_expand: bool = True,
        volatility_expand_threshold: float = 1.2,

        require_close_position: bool = True,
        close_position_threshold: float = 0.7,  # 收盘在上方70%的位置

        # 失败突破识别
        enable_failure_detection: bool = True,
        failure_pullback_threshold: float = 0.5,  # 回落超过突破幅度的50%

        # 仓位管理
        max_position_ratio: float = 0.2,
        enable_regime_gating: bool = True
    ):
        self.period = period
        self.use_atr_threshold = use_atr_threshold
        self.atr_multiplier = atr_multiplier
        self.price_threshold_pct = price_threshold_pct
        self.require_volume_confirm = require_volume_confirm
        self.volume_ratio_threshold = volume_ratio_threshold
        self.require_volatility_expand = require_volatility_expand
        self.volatility_expand_threshold = volatility_expand_threshold
        self.require_close_position = require_close_position
        self.close_position_threshold = close_position_threshold
        self.enable_failure_detection = enable_failure_detection
        self.failure_pullback_threshold = failure_pullback_threshold
        self.max_position_ratio = max_position_ratio
        self.enable_regime_gating = enable_regime_gating


class EnhancedBreakoutStrategy(EnhancedStrategy):
    """增强版突破策略

    特点：
    1. 波动率归一化：阈值 = k * ATR / price
    2. 多重确认：成交量、波动扩张、收盘位置
    3. 失败识别：快速回落时退出
    4. 市场状态感知
    """

    def __init__(
        self,
        config: BreakoutConfig = None,
        detector_config: RegimeDetectorConfig = None
    ):
        """初始化策略

        Args:
            config: 突破策略配置
            detector_config: 市场状态检测器配置
        """
        self.config = config or BreakoutConfig()

        super().__init__(
            name="breakout",
            regime_detector=RegimeDetector(detector_config) if detector_config else None,
            enable_regime_gating=self.config.enable_regime_gating
        )

        # 止损管理器
        self.stop_manager = DynamicStopManager()

        # 突破跟踪（用于失败识别）
        self._breakout_tracking: Dict[str, Dict[str, Any]] = {}

    def get_min_bars_required(self) -> int:
        """获取所需最小K线数量"""
        return max(
            self.config.period,
            self.config.period * 2,  # 用于成交量比较
            20  # 用于波动率计算
        ) + 10

    def _generate_strategy_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None,
        regime_info: Optional[RegimeInfo] = None
    ) -> List[StrategySignal]:
        """生成策略信号

        Args:
            bars: K线数据
            position: 当前持仓
            regime_info: 市场状态信息

        Returns:
            StrategySignal 列表
        """
        if len(bars) < self.get_min_bars_required():
            return []

        current_bar = bars[-1]
        symbol = current_bar.symbol

        # 1. 检查是否有失败的突破（需要快速退出）
        if position and position.quantity > 0:
            failure_signal = self._check_failed_breakout(symbol, current_bar.close, position)
            if failure_signal:
                return [failure_signal]

        # 2. 计算突破区间
        lookback_bars = bars[-(self.config.period + 1):-1]
        high_n = max(b.high for b in lookback_bars)
        low_n = min(b.low for b in lookback_bars)

        # 3. 计算ATR
        atr = self._calculate_atr(bars, 14)

        # 4. 计算突破阈值（波动率归一化）
        if self.config.use_atr_threshold:
            threshold = (atr * Decimal(str(self.config.atr_multiplier))) / current_bar.close
        else:
            threshold = Decimal(str(self.config.price_threshold_pct))

        # 5. 检测向上突破
        close = current_bar.close

        if close > high_n * (Decimal("1") + threshold):
            # 检查确认条件
            confirm_result = self._check_confirmations(bars, high_n, close, atr)
            if not confirm_result["confirmed"]:
                return []

            # 计算信号强度（基于确认条件数量）
            confidence = self._calculate_breakout_confidence(confirm_result, regime_info)

            # 计算目标仓位
            target_weight = self._calculate_position_weight(confidence, regime_info)

            current_qty = position.quantity if position else 0
            max_qty = 1000
            target_position = int(target_weight * max_qty)

            # 风险提示
            risk_hints = RiskHints(
                stop_loss=low_n,  # 区间下沿作为止损
                take_profit=close + (atr * Decimal("2")),
                trailing_stop=Decimal("0.015"),
                max_holding_days=10,
                position_limit=self.config.max_position_ratio * target_weight
            )

            signal = StrategySignal(
                symbol=symbol,
                timestamp=current_bar.timestamp,
                target_position=target_position,
                target_weight=target_weight,
                confidence=confidence,
                reason=f"向上突破{self.config.period}日高点 {high_n}",
                regime=regime_info.regime if regime_info else RegimeType.UNKNOWN,
                risk_hints=risk_hints,
                raw_action="BUY",
                metadata={
                    "breakout_high": float(high_n),
                    "breakout_close": float(close),
                    "atr": float(atr),
                    "current_price": float(close),
                    "confirmations": confirm_result
                }
            )

            # 记录突破信息（用于失败检测）
            self._breakout_tracking[symbol] = {
                "entry_price": close,
                "breakout_level": high_n,
                "entry_time": current_bar.timestamp,
                "highest_since_entry": close
            }

            return [signal]

        # 6. 检测向下突破（做空）
        elif close < low_n * (Decimal("1") - threshold):
            # 简化版：暂不实现做空
            pass

        return []

    def _check_confirmations(
        self,
        bars: List[Bar],
        break_level: Decimal,
        close_price: Decimal,
        atr: Decimal
    ) -> Dict[str, Any]:
        """检查突破确认条件

        Returns:
            确认结果字典
        """
        results = {
            "confirmed": True,
            "volume_confirm": False,
            "volatility_expand": False,
            "close_position": False,
            "score": 0
        }

        current_bar = bars[-1]
        lookback_bars = bars[-(self.config.period * 2):-1]

        # 1. 成交量确认
        if self.config.require_volume_confirm:
            avg_volume = sum(b.volume for b in lookback_bars) / len(lookback_bars)
            volume_ratio = current_bar.volume / avg_volume if avg_volume > 0 else 1.0

            results["volume_confirm"] = volume_ratio >= self.config.volume_ratio_threshold
            results["volume_ratio"] = float(volume_ratio)
            if results["volume_confirm"]:
                results["score"] += 1

        # 2. 波动扩张确认
        if self.config.require_volatility_expand:
            recent_atr = float(self._calculate_atr(bars[-20:], 14))
            older_atr = float(self._calculate_atr(bars[-40:-20], 14)) if len(bars) >= 40 else recent_atr

            vol_expand = recent_atr / older_atr if older_atr > 0 else 1.0
            results["volatility_expand"] = vol_expand >= self.config.volatility_expand_threshold
            results["volatility_expand_ratio"] = vol_expand
            if results["volatility_expand"]:
                results["score"] += 1

        # 3. 收盘位置确认
        if self.config.require_close_position:
            day_range = current_bar.high - current_bar.low
            close_position = (close_price - current_bar.low) / day_range if day_range > 0 else 0.5

            results["close_position"] = float(close_position) >= self.config.close_position_threshold
            results["close_position_value"] = float(close_position)
            if results["close_position"]:
                results["score"] += 1

        # 综合判断
        total_checks = (
            (1 if self.config.require_volume_confirm else 0) +
            (1 if self.config.require_volatility_expand else 0) +
            (1 if self.config.require_close_position else 0)
        )

        if total_checks > 0:
            results["confirmed"] = results["score"] >= (total_checks + 1) // 2  # 至少一半通过

        return results

    def _calculate_breakout_confidence(
        self,
        confirm_result: Dict[str, Any],
        regime_info: Optional[RegimeInfo]
    ) -> float:
        """计算突破信号置信度"""
        base_confidence = 0.6

        # 根据确认条件数量增加置信度
        score_bonus = confirm_result.get("score", 0) * 0.1

        # 根据市场状态调整
        regime_bonus = 0
        if regime_info:
            if regime_info.regime == RegimeType.HIGH_VOL:
                regime_bonus = 0.1  # 高波动突破更可靠
            elif regime_info.regime == RegimeType.TREND:
                regime_bonus = 0.15  # 趋势市突破可靠
            elif regime_info.regime == RegimeType.RANGE:
                regime_bonus = -0.1  # 震荡市可能假突破

        confidence = base_confidence + score_bonus + regime_bonus
        return max(0.3, min(0.95, confidence))

    def _calculate_position_weight(
        self,
        confidence: float,
        regime_info: Optional[RegimeInfo]
    ) -> float:
        """计算目标仓位权重"""
        base_weight = confidence * self.config.max_position_ratio

        # 市场状态调整
        if regime_info:
            vol = regime_info.volatility_level
            vol_adjustment = 1.0 / max(vol, 0.5)
            base_weight *= vol_adjustment

        return max(0, min(base_weight, self.config.max_position_ratio))

    def _check_failed_breakout(
        self,
        symbol: str,
        current_price: Decimal,
        position: Position
    ) -> Optional[StrategySignal]:
        """检查是否是失败的突破

        失败突破特征：
        - 突破后快速回落
        - 跌破突破水平
        """
        if not self.config.enable_failure_detection:
            return None

        tracking = self._breakout_tracking.get(symbol)
        if not tracking:
            return None

        entry_price = tracking["entry_price"]
        breakout_level = tracking["breakout_level"]

        # 更新最高价
        if current_price > tracking["highest_since_entry"]:
            tracking["highest_since_entry"] = current_price

        # 计算突破幅度
        breakout_range = tracking["highest_since_entry"] - breakout_level

        # 计算当前回落幅度
        pullback = tracking["highest_since_entry"] - current_price

        # 如果回落超过突破幅度的50%，认为是失败突破
        if breakout_range > 0:
            pullback_ratio = float(pullback / breakout_range)

            if pullback_ratio >= self.config.failure_pullback_threshold:
                # 快速退出
                self._breakout_tracking.pop(symbol, None)

                return StrategySignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    target_position=0,
                    target_weight=0,
                    confidence=0.8,
                    reason=f"失败突破: 回落{pullback_ratio:.1%}超过阈值",
                    regime=RegimeType.UNKNOWN,
                    risk_hints=None,
                    raw_action="SELL",
                    metadata={
                        "entry_price": float(entry_price),
                        "highest_price": float(tracking["highest_since_entry"]),
                        "pullback_ratio": pullback_ratio
                    }
                )

        # 跌破突破水平也视为失败
        if current_price < breakout_level:
            self._breakout_tracking.pop(symbol, None)

            return StrategySignal(
                symbol=symbol,
                timestamp=datetime.now(),
                target_position=0,
                target_weight=0,
                confidence=0.7,
                reason=f"失败突破: 跌破突破水平 {breakout_level}",
                regime=RegimeType.UNKNOWN,
                risk_hints=None,
                raw_action="SELL",
                metadata={}
            )

        return None

    def _calculate_atr(self, bars: List[Bar], period: int = 14) -> Decimal:
        """计算ATR"""
        if len(bars) < period + 1:
            total_range = sum((bar.high - bar.low) for bar in bars)
            return total_range / Decimal(len(bars)) if bars else Decimal('1')

        true_ranges = []
        for i in range(1, len(bars)):
            prev_close = bars[i - 1].close
            current_high = bars[i].high
            current_low = bars[i].low

            tr = max(
                current_high - current_low,
                abs(current_high - prev_close),
                abs(current_low - prev_close)
            )
            true_ranges.append(tr)

        recent_tr = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
        atr = sum(recent_tr) / Decimal(len(recent_tr))

        return atr

    def reset_state(self) -> None:
        """重置策略状态"""
        super().reset_state()
        self._breakout_tracking.clear()


if __name__ == "__main__":
    """测试代码"""
    from skills.market_data.mock_source import MockMarketSource

    print("=== EnhancedBreakoutStrategy 测试 ===\n")

    # 创建测试数据（包含突破）
    base_price = Decimal("100.0")
    bars = []
    base_date = datetime(2024, 1, 1)

    # 前30天：震荡
    for i in range(30):
        date = base_date + timedelta(days=i)
        oscillation = Decimal(str(2 * (i % 10 / 5 - 1)))  # -2 到 +2

        close_price = base_price + oscillation
        open_price = close_price + Decimal(str(((-1) ** i) * 0.2))
        high_price = max(open_price, close_price) + Decimal("0.5")
        low_price = min(open_price, close_price) - Decimal("0.5")

        bars.append(Bar(
            symbol="TEST",
            timestamp=date,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000000,
            interval="1d"
        ))

    # 后10天：突破（价格上涨）
    for i in range(30, 40):
        date = base_date + timedelta(days=i)
        trend = Decimal(str((i - 30) * 1.5))  # 每天上涨1.5

        close_price = base_price + Decimal("3") + trend
        open_price = close_price - Decimal("0.5")
        high_price = close_price + Decimal("1.0")
        low_price = close_price - Decimal("1.0")

        bars.append(Bar(
            symbol="TEST",
            timestamp=date,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1500000,  # 成交量放大
            interval="1d"
        ))

    # 创建策略
    config = BreakoutConfig(
        period=20,
        use_atr_threshold=True,
        atr_multiplier=1.0,
        require_volume_confirm=True,
        require_volatility_expand=True,
        require_close_position=True,
        enable_failure_detection=True
    )

    strategy = EnhancedBreakoutStrategy(config=config)

    # 测试突破检测
    signals = strategy.generate_strategy_signals(bars)
    print(f"生成信号数: {len(signals)}")
    for sig in signals:
        print(f"  {sig.raw_action}: 目标仓位={sig.target_position}, 权重={sig.target_weight:.2f}")
        print(f"  原因: {sig.reason}")
        if sig.metadata.get("confirmations"):
            conf = sig.metadata["confirmations"]
            print(f"  确认: 成交量={conf.get('volume_confirm')}, "
                  f"波动扩张={conf.get('volatility_expand')}, "
                  f"收盘位置={conf.get('close_position')}, "
                  f"得分={conf.get('score', 0)}")

    print("\n✓ 测试通过")


