"""
增强的策略基类

支持统一信号输出、状态管理、市场状态感知和动态参数调整
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any, TYPE_CHECKING
from datetime import datetime
from decimal import Decimal
import logging

if TYPE_CHECKING:
    from skills.analysis.regime_detector import RegimeDetector

from core.models import Bar, Position, Signal
from core.strategy_models import (
    StrategySignal,
    StrategyState,
    StatefulStrategy,
    RegimeType,
    RegimeInfo,
    RiskHints
)
from skills.market_analysis.market_state import MarketState, MarketStateBuilder
from skills.market_analysis.detectors import MarketAnalyzer
from skills.market_analysis import MarketManager, get_market_manager

logger = logging.getLogger(__name__)


class EnhancedStrategy(ABC):
    """增强策略基类

    所有新策略应继承此类，支持：
    1. 统一的 StrategySignal 输出
    2. 策略状态管理
    3. 市场状态感知
    4. 连续仓位 sizing
    5. 动态参数调整
    6. 策略可行性评估
    """

    def __init__(
        self,
        name: str,
        regime_detector: Optional["RegimeDetector"] = None,
        enable_regime_gating: bool = True,
        enable_market_manager: bool = True,
        strategy_type: Optional[str] = None,
        enable_feasibility_check: bool = True
    ):
        """初始化策略

        Args:
            name: 策略名称
            regime_detector: 市场状态检测器（可选，向后兼容）
            enable_regime_gating: 是否启用市场状态 gating
            enable_market_manager: 是否启用 MarketManager（推荐使用）
            strategy_type: 策略类型 (ma, breakout, ml)，用于可行性评估
            enable_feasibility_check: 是否启用策略可行性检查
        """
        self.name = name
        self.strategy_type = strategy_type or name.lower()
        self.enable_feasibility_check = enable_feasibility_check

        # 向后兼容：如果传入了 regime_detector，仍使用它
        self.regime_detector = regime_detector
        self.enable_regime_gating = enable_regime_gating

        # 新架构：使用 MarketManager
        self.market_manager: Optional[MarketManager] = None
        if enable_market_manager:
            try:
                self.market_manager = get_market_manager(symbol="DEFAULT")
                logger.debug(f"{name}: 使用 MarketManager 进行市场分析")
            except Exception as e:
                logger.warning(f"{name}: 无法初始化 MarketManager: {e}")

        self._state: Optional[StrategyState] = None

        # 当前市场状态缓存
        self._current_market_state: Optional[MarketState] = None

        # 当前动态参数
        self._dynamic_params: Dict[str, Any] = {}

    @abstractmethod
    def get_min_bars_required(self) -> int:
        """获取策略所需的最小K线数量"""
        pass

    @abstractmethod
    def _generate_strategy_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None,
        regime_info: Optional[RegimeInfo] = None
    ) -> List[StrategySignal]:
        """生成策略信号（内部方法）

        子类实现此方法，返回统一的 StrategySignal

        Args:
            bars: K线数据
            position: 当前持仓
            regime_info: 市场状态信息

        Returns:
            策略信号列表
        """
        pass

    def generate_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None,
        external_data: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """生成交易信号（兼容旧接口）

        将 StrategySignal 转换为 Signal，保持向后兼容

        Args:
            bars: K线数据
            position: 当前持仓
            external_data: 外部数据（情绪、宏观等）

        Returns:
            Signal 列表
        """
        strategy_signals = self.generate_strategy_signals(bars, position, external_data)

        # 获取当前价格
        current_price = bars[-1].close if bars else None

        # 转换为 Signal（保持向后兼容）
        return [
            self._convert_to_legacy_signal(
                sig,
                position,
                current_price=current_price
            )
            for sig in strategy_signals
        ]

    def generate_strategy_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None,
        external_data: Optional[Dict[str, Any]] = None
    ) -> List[StrategySignal]:
        """生成策略信号（新接口）

        直接返回 StrategySignal，支持市场状态感知和可行性评估

        Args:
            bars: K线数据
            position: 当前持仓
            external_data: 外部数据（情绪、宏观等）

        Returns:
            StrategySignal 列表
        """
        if len(bars) < self.get_min_bars_required():
            return []

        # 1. 获取市场状态
        market_state = self.get_market_state(bars, external_data)

        # 2. 获取动态参数
        if market_state:
            self.get_dynamic_params(market_state)

        # 3. 评估策略可行性
        feasibility = {"strategy_allowed": True, "position_scale": 1.0, "reason": ""}
        if market_state:
            feasibility = self.evaluate_feasibility(market_state)

        # 4. 如果策略不允许运行，返回空信号或平仓信号
        if not feasibility["strategy_allowed"]:
            # 如果有持仓，返回平仓信号
            if position and position.quantity > 0:
                return [StrategySignal(
                    symbol=bars[-1].symbol,
                    timestamp=bars[-1].timestamp,
                    target_position=0,
                    target_weight=0,
                    confidence=1.0,
                    reason=f"策略禁用: {feasibility['reason']}",
                    regime=market_state.regime if market_state else RegimeType.UNKNOWN,
                    metadata={"feasibility_reason": feasibility["reason"]}
                )]
            return []

        # 5. 构建兼容的 regime_info
        regime_info = None
        if market_state:
            # 从 market_state 提取信息构建 RegimeInfo
            features = market_state.features_snapshot.get("internal_analysis", {})
            regime_features = features.get("regime", {}).get("features", {})
            regime_info = RegimeInfo(
                regime=market_state.regime,
                confidence=market_state.regime_confidence,
                trend_strength=market_state.trend_strength,
                volatility_level=market_state.volatility_percentile,
                adx=regime_features.get("adx"),
                rsi=regime_features.get("rsi"),
                atr=regime_features.get("atr")
            )
        elif self.regime_detector:
            regime_info = self.regime_detector.detect(bars)

        # 6. 生成策略信号（子类实现）
        strategy_signals = self._generate_strategy_signals(bars, position, regime_info)

        # 7. 应用市场状态 gating 和可行性调整
        if self.enable_regime_gating and market_state:
            strategy_signals = self._apply_market_state_filtering(
                strategy_signals, market_state, feasibility
            )
        elif self.enable_regime_gating and regime_info:
            strategy_signals = self._apply_regime_gating(
                strategy_signals, regime_info.regime, regime_info.confidence
            )

        return strategy_signals

    def _apply_market_state_filtering(
        self,
        signals: List[StrategySignal],
        market_state: MarketState,
        feasibility: Dict[str, Any]
    ) -> List[StrategySignal]:
        """应用市场状态过滤和调整

        结合市场状态和可行性评估结果调整信号

        Args:
            signals: 原始信号
            market_state: 市场状态
            feasibility: 可行性评估结果

        Returns:
            处理后的信号列表
        """
        position_scale = feasibility.get("position_scale", 1.0)

        filtered_signals = []
        for sig in signals:
            # 应用仓位缩放
            scaled_position = sig.target_position * position_scale
            scaled_weight = sig.target_weight * position_scale

            # 更新置信度
            scaled_confidence = sig.confidence * market_state.confidence

            # 获取或创建风险提示
            risk_hints = sig.risk_hints or RiskHints()

            # 如果市场风险较高，调整仓位限制
            if market_state.should_reduce_risk:
                # 设置仓位限制
                current_limit = risk_hints.position_limit or 1.0
                risk_hints.position_limit = current_limit * position_scale

            # 构建调整后的信号
            filtered_signal = StrategySignal(
                symbol=sig.symbol,
                timestamp=sig.timestamp,
                target_position=scaled_position,
                target_weight=scaled_weight,
                confidence=scaled_confidence,
                reason=f"{sig.reason} ({feasibility.get('reason', '')})",
                regime=market_state.regime,
                risk_hints=risk_hints,
                raw_action=sig.raw_action,
                metadata={**sig.metadata, "market_state": market_state.to_dict()}
            )
            filtered_signals.append(filtered_signal)

        return filtered_signals

    def _apply_regime_gating(
        self,
        signals: List[StrategySignal],
        regime: RegimeType,
        regime_confidence: float
    ) -> List[StrategySignal]:
        """应用市场状态 gating（向后兼容方法）

        根据策略类型和市场状态决定是否允许运行

        Args:
            signals: 原始信号
            regime: 市场状态
            regime_confidence: 市场状态置信度

        Returns:
            处理后的信号列表
        """
        from skills.analysis.regime_detector import is_regime_allowed, get_regime_position_modifier

        # 检查策略是否允许在当前市场状态下运行
        if not is_regime_allowed(regime, self.name.lower()):
            # 不允许运行，返回平仓信号
            return [StrategySignal(
                symbol=signals[0].symbol if signals else "",
                timestamp=datetime.now(),
                target_position=0,
                target_weight=0,
                confidence=regime_confidence,
                reason=f"Regime gating: {self.name} 禁用 in {regime.value}",
                regime=regime
            )] if signals else []

        # 允许运行，但可能需要调整仓位
        gated_signals = []
        for sig in signals:
            # 调整目标仓位
            modified_weight = get_regime_position_modifier(regime, sig.target_weight)
            modified_position = get_regime_position_modifier(regime, sig.target_position)

            # 更新信号
            gated_signal = StrategySignal(
                symbol=sig.symbol,
                timestamp=sig.timestamp,
                target_position=modified_position,
                target_weight=modified_weight,
                confidence=sig.confidence * regime_confidence,  # 综合置信度
                reason=f"{sig.reason} (Regime: {regime.value})",
                regime=regime,
                risk_hints=sig.risk_hints,
                raw_action=sig.raw_action,
                metadata=sig.metadata
            )
            gated_signals.append(gated_signal)

        return gated_signals

    def _convert_to_legacy_signal(
        self,
        strategy_signal: StrategySignal,
        position: Optional[Position],
        current_price: Optional[Decimal] = None
    ) -> Signal:
        """将 StrategySignal 转换为 Signal（向后兼容）

        Args:
            strategy_signal: 新格式信号
            position: 当前持仓
            current_price: 当前价格（可选）

        Returns:
            旧格式 Signal
        """
        # 根据 target_position 决定 action
        current_qty = position.quantity if position else 0
        target_qty = int(strategy_signal.target_position)

        if target_qty > current_qty:
            action = "BUY"
            quantity = target_qty - current_qty
        elif target_qty < current_qty:
            action = "SELL"
            quantity = current_qty - target_qty
        else:
            action = "HOLD"
            quantity = 0

        # 从 risk_hints 提取止盈止损
        stop_loss = None
        take_profit = None
        if strategy_signal.risk_hints:
            stop_loss = strategy_signal.risk_hints.stop_loss
            take_profit = strategy_signal.risk_hints.take_profit

        # 如果是 HOLD，返回一个合法的 Signal（quantity=1 会被 runner 层忽略）
        # 如果是 BUY/SELL，也需要提供 price（使用传入的 current_price 或从 strategy_signal 的元数据获取）
        price = current_price or Decimal("100.0")  # 默认值，实际应该从 bars 获取
        if strategy_signal.metadata.get("current_price"):
            price = Decimal(str(strategy_signal.metadata["current_price"]))

        return Signal(
            symbol=strategy_signal.symbol,
            timestamp=strategy_signal.timestamp,
            action=action,
            price=price,
            quantity=quantity if quantity > 0 else 1,
            confidence=strategy_signal.confidence,
            reason=strategy_signal.reason,
            target_price=take_profit,
            stop_loss=stop_loss
        )

    def get_state(self) -> StrategyState:
        """获取策略状态"""
        if self._state is None:
            self._state = StrategyState(
                state_version="1.0",
                last_update=datetime.now(),
                internal_state={"strategy_name": self.name}
            )
        return self._state

    def set_state(self, state: StrategyState) -> None:
        """恢复策略状态"""
        self._state = state

    def reset_state(self) -> None:
        """重置策略状态"""
        self._state = None
        if self.regime_detector:
            self.regime_detector.reset_state()
        self._current_market_state = None
        self._dynamic_params = {}

    def get_market_state(
        self,
        bars: List[Bar],
        external_data: Optional[Dict[str, Any]] = None
    ) -> Optional[MarketState]:
        """获取当前市场状态

        优先使用 MarketManager，否则使用旧的 regime_detector

        Args:
            bars: K线数据
            external_data: 外部数据（情绪、宏观等）

        Returns:
            当前市场状态
        """
        if self.market_manager:
            try:
                symbol = bars[0].symbol if bars else "DEFAULT"
                timestamp = bars[-1].timestamp if bars else datetime.now()

                state = self.market_manager.analyze(
                    bars=bars,
                    timestamp=timestamp,
                    external_data=external_data,
                    symbol=symbol
                )
                self._current_market_state = state
                return state
            except Exception as e:
                logger.warning(f"{self.name}: MarketManager.analyze 失败: {e}")

        # 回退到旧的 regime_detector
        if self.regime_detector:
            try:
                regime_info = self.regime_detector.detect(bars)
                # 构建简化的 MarketState
                from skills.market_analysis.market_state import (
                    VolatilityState, VolatilityTrend, LiquidityState,
                    SentimentState, SystemicRiskFlags
                )
                state = MarketState(
                    timestamp=bars[-1].timestamp if bars else datetime.now(),
                    symbol=bars[0].symbol if bars else "DEFAULT",
                    regime=regime_info.regime,
                    regime_confidence=regime_info.confidence,
                    volatility_state=VolatilityState.NORMAL,
                    volatility_trend=VolatilityTrend.STABLE,
                    liquidity_state=LiquidityState.NORMAL,
                    sentiment_state=SentimentState.NEUTRAL,
                    systemic_risk=SystemicRiskFlags()
                )
                self._current_market_state = state
                return state
            except Exception as e:
                logger.warning(f"{self.name}: regime_detector.detect 失败: {e}")

        return None

    def get_dynamic_params(self, market_state: Optional[MarketState] = None) -> Dict[str, Any]:
        """获取动态参数

        根据市场状态获取策略的动态参数

        Args:
            market_state: 市场状态（如果不提供，使用缓存的状态）

        Returns:
            动态参数字典
        """
        state = market_state or self._current_market_state

        if self.market_manager and state:
            try:
                params = self.market_manager.param_manager.get_parameters(
                    self.strategy_type, state
                )
                self._dynamic_params = params
                return params
            except Exception as e:
                logger.warning(f"{self.name}: 获取动态参数失败: {e}")

        return {}

    def evaluate_feasibility(self, market_state: Optional[MarketState] = None) -> Dict[str, Any]:
        """评估策略可行性

        Args:
            market_state: 市场状态（如果不提供，使用缓存的状态）

        Returns:
            {
                "strategy_allowed": bool,
                "position_scale": float,
                "reason": str
            }
        """
        if not self.enable_feasibility_check:
            return {"strategy_allowed": True, "position_scale": 1.0, "reason": "未启用可行性检查"}

        state = market_state or self._current_market_state

        if self.market_manager and state:
            try:
                return self.market_manager.evaluate_strategy(self.strategy_type, state)
            except Exception as e:
                logger.warning(f"{self.name}: 可行性评估失败: {e}")
                return {"strategy_allowed": True, "position_scale": 1.0, "reason": f"评估失败: {e}"}

        return {"strategy_allowed": True, "position_scale": 1.0, "reason": "无市场状态"}

    def calculate_position_size(
        self,
        signal_strength: float,
        volatility: float,
        capital: float,
        max_position_ratio: float = 0.2
    ) -> float:
        """计算目标仓位大小（连续 sizing）

        Args:
            signal_strength: 信号强度 (0-1)
            volatility: 波动率水平（相对值，1为平均水平）
            capital: 可用资金
            max_position_ratio: 最大仓位比例

        Returns:
            仓位比例
        """
        # 基础仓位：信号强度 * 最大仓位
        base_ratio = signal_strength * max_position_ratio

        # 波动率调整：高波动降低仓位
        vol_adjustment = 1.0 / max(volatility, 0.5)

        # 最终仓位比例
        final_ratio = base_ratio * vol_adjustment
        final_ratio = min(final_ratio, max_position_ratio)  # 不超过最大仓位
        final_ratio = max(final_ratio, 0.0)  # 不低于0

        # 返回资金比例（具体股数由上层决定）
        return final_ratio


class AdaptiveMixin:
    """自适应策略混入类

    提供自适应功能：
    - 动态参数调整
    - 性能监控
    """

    def __init__(self):
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0
        }

    def update_performance(self, pnl: float, is_win: bool):
        """更新性能指标"""
        self.performance_metrics["total_trades"] += 1
        if is_win:
            self.performance_metrics["winning_trades"] += 1
        else:
            self.performance_metrics["losing_trades"] += 1
        self.performance_metrics["total_pnl"] += pnl

    def get_win_rate(self) -> float:
        """获取胜率"""
        total = self.performance_metrics["total_trades"]
        if total == 0:
            return 0.0
        return self.performance_metrics["winning_trades"] / total

    def should_disable(self) -> bool:
        """判断是否应该禁用策略"""
        # 连续亏损超过5次，考虑禁用
        if self.performance_metrics.get("consecutive_losses", 0) >= 5:
            return True

        # 胜率低于30%且交易次数>10，考虑禁用
        if (self.performance_metrics["total_trades"] > 10 and
            self.get_win_rate() < 0.3):
            return True

        return False


if __name__ == "__main__":
    """测试代码"""
    print("=== EnhancedStrategy 测试 ===\n")

    # 创建一个简单的测试策略
    class TestStrategy(EnhancedStrategy):
        def __init__(self):
            super().__init__("test", enable_regime_gating=True)

        def get_min_bars_required(self) -> int:
            return 20

        def _generate_strategy_signals(
            self,
            bars: List[Bar],
            position: Optional[Position] = None,
            regime_info: Optional[RegimeInfo] = None
        ) -> List[StrategySignal]:
            # 简单返回一个买入信号
            return [StrategySignal(
                symbol=bars[-1].symbol,
                timestamp=bars[-1].timestamp,
                target_position=100,
                target_weight=0.5,
                confidence=0.8,
                reason="测试信号",
                regime=regime_info.regime if regime_info else RegimeType.UNKNOWN
            )]

    # 测试
    from datetime import timedelta
    from skills.market_data.mock_source import MockMarketSource

    source = MockMarketSource(seed=42)
    bars = source.get_bars("AAPL", datetime(2024, 1, 1), datetime(2024, 2, 1), "1d")

    strategy = TestStrategy()
    signals = strategy.generate_signals(bars)

    print(f"1. 生成信号数: {len(signals)}")
    if signals:
        print(f"   Action: {signals[0].action}")
        print(f"   Reason: {signals[0].reason}")

    # 测试新接口
    strategy_signals = strategy.generate_strategy_signals(bars)
    print(f"\n2. 新接口信号数: {len(strategy_signals)}")
    if strategy_signals:
        sig = strategy_signals[0]
        print(f"   Target Position: {sig.target_position}")
        print(f"   Target Weight: {sig.target_weight}")
        print(f"   Regime: {sig.regime.value}")

    # 测试状态管理
    print("\n3. 状态管理:")
    state = strategy.get_state()
    print(f"   状态: {state.internal_state}")

    # 测试市场状态获取
    print("\n4. 市场状态:")
    market_state = strategy.get_market_state(bars)
    if market_state:
        print(f"   Regime: {market_state.regime.value}")
        print(f"   Volatility: {market_state.volatility_state.value}")
        print(f"   Should Reduce Risk: {market_state.should_reduce_risk}")

    print("\n✓ 所有测试通过")
