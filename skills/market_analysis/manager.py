"""
市场管理器

统一的市场分析入口，整��所有检测器和外部信号
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from core.models import Bar
from skills.market_analysis.market_state import (
    MarketState,
    MarketStateBuilder,
    RegimeType,
    VolatilityState,
    VolatilityTrend,
    LiquidityState,
    SentimentState,
    SystemicRiskFlags
)
from skills.market_analysis.detectors import (
    RegimeDetector,
    VolatilityDetector,
    LiquidityDetector,
    MarketAnalyzer
)
from skills.market_analysis.external_signals import (
    SentimentDetector,
    MacroDetector,
    EventCalendar,
    ExternalSignalManager
)
from skills.market_analysis.strategy_feasibility import (
    StrategyFeasibilityEvaluator,
    DynamicParameterManager
)

logger = logging.getLogger(__name__)


class MarketManager:
    """
    市场管理器

    统一管理：
    1. 市场状态检测
    2. 外部信号处理
    3. 策略可行性评估
    4. 动态参数管理
    """

    def __init__(
        self,
        symbol: str = "DEFAULT",
        enable_external_signals: bool = True,
        enable_feasibility_eval: bool = True,
        events_file: Optional[str] = None
    ):
        """
        初始化市场管理器

        Args:
            symbol: 默认标的
            enable_external_signals: 是否启用外部信号
            enable_feasibility_eval: 是否启用策略可行性评估
            events_file: 事件日历文件路径
        """
        self.symbol = symbol

        # 市场状态构建器
        self.state_builder = MarketStateBuilder(symbol)

        # 检测器
        self.analyzer = MarketAnalyzer(
            regime_detector=RegimeDetector(),
            vol_detector=VolatilityDetector(),
            liq_detector=LiquidityDetector()
        )

        # 外部信号管理器
        self.external_signals = None
        if enable_external_signals:
            self.external_signals = ExternalSignalManager(
                enable_sentiment=True,
                enable_macro=True,
                enable_events=True,
                events_file=events_file
            )

        # 策略可行性评估
        self.feasibility_eval = StrategyFeasibilityEvaluator() if enable_feasibility_eval else None

        # 动态参数管理器
        self.param_manager = DynamicParameterManager()

        # 当前市场状态
        self._current_state: Optional[MarketState] = None

        # 状态历史
        self._state_history: List[MarketState] = []

    def analyze(
        self,
        bars: List[Bar],
        timestamp: Optional[datetime] = None,
        external_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None
    ) -> MarketState:
        """
        分析市场并生成完整的市场状态

        Args:
            bars: K线数据
            timestamp: 当前时间
            external_data: 外部数据（情绪、宏观、事件等）
            symbol: 标的代码

        Returns:
            MarketState
        """
        symbol = symbol or self.symbol
        timestamp = timestamp or (bars[-1].timestamp if bars else datetime.now())

        # 1. 获取内部市场分析结果
        analysis = self.analyzer.analyze(bars)

        # 2. 获取外部信号
        external_signals_data = {}
        if self.external_signals:
            external_signals_data = self.external_signals.update_signals(timestamp, external_data, symbol)

        # 3. 构建完整的市场状态
        state = self._build_market_state(bars, timestamp, analysis, external_signals_data, symbol)

        # 4. 保存状态
        self._current_state = state
        self._state_history.append(state)

        # 限制历史长度
        if len(self._state_history) > 1000:
            self._state_history = self._state_history[-1000:]

        return state

    def _build_market_state(
        self,
        bars: List[Bar],
        timestamp: datetime,
        analysis: Dict[str, Any],
        external_signals: Dict[str, Any],
        symbol: str
    ) -> MarketState:
        """构建 MarketState"""
        # 导入必要的类型
        from skills.market_analysis.market_state import (
            SystemicRiskFlags, SentimentState, VolatilityTrend, LiquidityState
        )

        # 从 analysis 中提取核心信息
        regime_result = analysis["regime"]
        vol_result = analysis["volatility"]
        liq_result = analysis["liquidity"]
        summary = analysis["summary"]

        # 处理外部信号
        sentiment_state = "NEUTRAL"
        sentiment_score = 0.0
        systemic_risk_flags = SystemicRiskFlags()

        if external_signals:
            sentiment_data = external_signals.get("sentiment", {})
            if sentiment_data:
                sentiment_state = sentiment_data.get("sentiment_state", "NEUTRAL")
                sentiment_score = sentiment_data.get("sentiment_score", 0.0)

            risk_data = external_signals.get("systemic_risk", {})
            event_data = external_signals.get("event_window", {})

            # 整合系统性风险
            systemic_risk_flags = SystemicRiskFlags(
                event_window=event_data.get("in_event_window", False),
                high_systemic_risk=risk_data.get("stress_level") == "HIGH",
                cross_market_stress=risk_data.get("market_stress", False),
                currency_crisis=risk_data.get("currency_crisis", False)
            )

        # 转换情绪状态字符串为枚举
        sentiment_enum = SentimentState.NEUTRAL
        if sentiment_state == "FEAR":
            sentiment_enum = SentimentState.FEAR
        elif sentiment_state == "GREED":
            sentiment_enum = SentimentState.GREED

        # 构建状态
        state = MarketState(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime_result.get("regime", RegimeType.UNKNOWN),
            regime_confidence=regime_result.get("confidence", 0.0),
            volatility_state=vol_result.get("state", VolatilityState.NORMAL),
            volatility_trend=vol_result.get("trend", VolatilityTrend.STABLE),
            volatility_percentile=vol_result.get("percentile", 0.5),
            realized_vol=vol_result.get("realized_vol", 0.0),
            atr_ratio=vol_result.get("atr_ratio", 0.0),
            liquidity_state=liq_result.get("state", LiquidityState.NORMAL),
            volume_ratio=liq_result.get("volume_ratio", 1.0),
            turnover_ratio=liq_result.get("turnover_ratio", 1.0),
            sentiment_state=sentiment_enum,
            sentiment_score=sentiment_score,
            systemic_risk=systemic_risk_flags,
            trend_strength=regime_result.get("features", {}).get("adx", 0) / 50,  # 归一化
            trend_direction="UP" if regime_result.get("features", {}).get("slope", 0) > 0 else "DOWN",
            confidence=summary.get("risk_level", "LOW") == "LOW" and 0.7 or 0.5,
            features_snapshot={
                "internal_analysis": analysis,
                "external_signals": external_signals
            }
        )

        return state

    def get_state(self) -> Optional[MarketState]:
        """获取当前市场状态"""
        return self._current_state

    def get_state_history(self, limit: int = 100) -> List[MarketState]:
        """获取状态历史"""
        return self._state_history[-limit:]

    def evaluate_strategy(
        self,
        strategy_type: str,
        market_state: Optional[MarketState] = None
    ) -> Dict[str, Any]:
        """
        评估策略可行性

        Args:
            strategy_type: 策略类型 (ma, breakout, ml)
            market_state: 市场状态（默认使用当前状态）

        Returns:
            {
                "strategy_allowed": bool,
                "position_scale": float,
                "dynamic_params": dict,
                "reason": str
            }
        """
        if market_state is None:
            market_state = self._current_state

        if market_state is None:
            return {
                "strategy_allowed": True,
                "position_scale": 1.0,
                "dynamic_params": {},
                "reason": "无市场状态数据"
            }

        # 1. 可行性评估
        if self.feasibility_eval:
            feasibility = self.feasibility_eval.evaluate(market_state, strategy_type)
        else:
            feasibility = {"strategy_allowed": True, "position_scale": 1.0, "reason": "未启用评估"}

        # 2. 获取动态参数
        dynamic_params = self.param_manager.get_parameters(strategy_type, market_state)

        return {
            "strategy_allowed": feasibility["strategy_allowed"],
            "position_scale": feasibility["position_scale"],
            "dynamic_params": dynamic_params,
            "reason": feasibility["reason"]
        }

    def save_state_history(self, filepath: str):
        """保存状态历史到文件"""
        import json

        data = {
            "symbol": self.symbol,
            "states": [state.to_dict() for state in self._state_history],
            "saved_at": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"已保存 {len(self._state_history)} 条市场状态到 {filepath}")

    def load_state_history(self, filepath: str):
        """从文件加载状态历史"""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.symbol = data["symbol"]
        self._state_history = [
            MarketState.from_dict(s) for s in data["states"]
        ]

        if self._state_history:
            self._current_state = self._state_history[-1]

        logger.info(f"已加载 {len(self._state_history)} 条市场状态")


# 单例模式的全局市场管理器
_global_market_manager: Optional[MarketManager] = None


def get_market_manager(
    symbol: str = "DEFAULT",
    **kwargs
) -> MarketManager:
    """获取全局市场管理器（单例模式）"""
    global _global_market_manager

    if _global_market_manager is None or _global_market_manager.symbol != symbol:
        _global_market_manager = MarketManager(symbol=symbol, **kwargs)

    return _global_market_manager


if __name__ == "__main__":
    """测试代码"""
    print("=== MarketManager 测试 ===\n")

    # 创建测试数据
    bars = []
    for i in range(50):
        from datetime import timedelta
        from decimal import Decimal
        from core.models import Bar

        date = datetime(2024, 1, 1) + timedelta(days=i)
        trend = i * 0.5
        noise = (i % 5 - 2) * 0.3
        close = 100 + trend + noise

        bars.append(Bar(
            symbol="TEST",
            timestamp=date,
            open=Decimal(str(close)),
            high=Decimal(str(close + 0.5)),
            low=Decimal(str(close - 0.5)),
            close=Decimal(str(close)),
            volume=1000000,
            interval="1d"
        ))

    # 创建市场管理器
    manager = MarketManager(symbol="TEST")

    print("1. 市场分析:")
    state = manager.analyze(bars)
    print(f"   Regime: {state.regime.value}")
    print(f"   Volatility: {state.volatility_state.value}")
    print(f"   Liquidity: {state.liquidity_state.value}")
    print(f"   Should Reduce Risk: {state.should_reduce_risk}")
    print(f"   Position Scale Factor: {state.position_scale_factor:.2f}")

    print("\n2. 策略评估:")
    eval_result = manager.evaluate_strategy("ma")
    print(f"   MA 策略允许: {eval_result['strategy_allowed']}")
    print(f"   仓位缩放: {eval_result['position_scale']:.2f}")
    print(f"   原因: {eval_result['reason']}")

    print("\n✓ 所有测试通过")
