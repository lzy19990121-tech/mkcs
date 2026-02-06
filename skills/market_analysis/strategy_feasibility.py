"""
策略可行性评估器

根据市场状态和历史表现评估策略是否应该启用
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict

from skills.market_analysis.market_state import MarketState

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformanceRecord:
    """策略历史表现记录"""
    regime: str
    volatility_state: str
    liquidity_state: str

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    # 样本量
    sample_count: int = 0


class StrategyFeasibilityEvaluator:
    """
    策略可行性评估器

    输入：
    - MarketState
    - 策略类型
    - 历史 runs

    输出：
    - strategy_allowed
    - position_scale
    - reason
    """

    def __init__(
        self,
        min_samples: int = 10,
        bad_performance_threshold: float = -0.05,  # -5% 亏损认为表现差
        good_performance_threshold: float = 0.02   # 2% 收益认为表现好
    ):
        self.min_samples = min_samples
        self.bad_performance_threshold = bad_performance_threshold
        self.good_performance_threshold = good_performance_threshold

        # 历史表现记录 {策略类型: {状态组合: PerformanceRecord}}
        self._performance_history: Dict[str, Dict[str, StrategyPerformanceRecord]] = defaultdict(
            lambda: defaultdict(lambda: StrategyPerformanceRecord(
                regime="", volatility_state="", liquidity_state=""
            ))
        )

    def evaluate(
        self,
        market_state: MarketState,
        strategy_type: str,
        historical_runs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        评估策略在当前市场状态下的可行性

        Returns:
            {
                "strategy_allowed": bool,
                "position_scale": float (0-1),
                "reason": str,
                "confidence": float (0-1)
            }
        """
        # 1. 创建状态组合键
        state_key = self._get_state_key(market_state)

        # 2. 从历史记录获取表现
        performance = self._get_performance_record(strategy_type, state_key)

        # 3. 如果没有历史数据，使用规则引擎
        if performance.sample_count < self.min_samples:
            return self._rule_based_evaluation(market_state, strategy_type)

        # 4. 基于历史表现评估
        return self._performance_based_evaluation(performance, market_state)

    def _get_state_key(self, market_state: MarketState) -> str:
        """创建状态组合键"""
        return f"{market_state.regime.value}_{market_state.volatility_state.value}_{market_state.liquidity_state.value}"

    def _get_performance_record(self, strategy_type: str, state_key: str) -> StrategyPerformanceRecord:
        """获取历史表现记录"""
        return self._performance_history[strategy_type][state_key]

    def _rule_based_evaluation(self, market_state: MarketState, strategy_type: str) -> Dict[str, Any]:
        """基于规则评估（无历史数据时使用）"""
        allowed = True
        position_scale = 1.0
        reasons = []

        # 危机状态：大多数策略禁用
        if market_state.regime.value == "CRISIS":
            if strategy_type in ["ma", "breakout"]:
                allowed = False
                position_scale = 0.0
                reasons.append("危机模式下策略禁用")
            elif strategy_type == "ml":
                position_scale = 0.1
                reasons.append("危机模式下ML降权")

        # 极端波动：降低仓位
        if market_state.volatility_state.value == "EXTREME":
            position_scale *= 0.3
            reasons.append("极端波动降权")

        # 低流动性：禁止开新仓
        if market_state.liquidity_state.value == "FROZEN":
            allowed = False
            position_scale = 0.0
            reasons.append("流动性枯竭禁止交易")
        elif market_state.liquidity_state.value == "THIN":
            position_scale *= 0.3
            reasons.append("流动性不足降权")

        # 震荡市：趋势策略降权
        if market_state.regime.value == "RANGE":
            if strategy_type in ["ma", "breakout"]:
                position_scale *= 0.3
                reasons.append("震荡市趋势策略降权")

        # 系统性风险事件窗口
        if market_state.systemic_risk.has_any_risk():
            if market_state.systemic_risk.event_window:
                allowed = False
                position_scale = 0.0
                reasons.append("事件窗口暂停新仓")
            else:
                position_scale *= 0.5
                reasons.append("系统性风险降权")

        # 极端情绪
        if market_state.sentiment_state.value == "FEAR":
            if strategy_type == "breakout":
                position_scale *= 0.3
                reasons.append("恐慌情绪突破策略降权")
        elif market_state.sentiment_state.value == "GREED":
            if strategy_type == "ma":
                position_scale *= 0.7
                reasons.append("贪婪情绪趋势策略降权")

        return {
            "strategy_allowed": allowed,
            "position_scale": max(0.0, min(1.0, position_scale)),
            "reason": "; ".join(reasons) if reasons else "无限制",
            "confidence": 0.7  # 基于规则的评估置信度中等
        }

    def _performance_based_evaluation(
        self,
        performance: StrategyPerformanceRecord,
        market_state: MarketState
    ) -> Dict[str, Any]:
        """基于历史表现评估"""
        if performance.sample_count < self.min_samples:
            return self._rule_based_evaluation(market_state, "default")

        # 根据历史表现决定
        if performance.total_return < self.bad_performance_threshold:
            # 历史上表现差
            return {
                "strategy_allowed": False,
                "position_scale": 0.0,
                "reason": f"历史上在此状态亏损{performance.total_return:.1%}",
                "confidence": 0.8
            }
        elif performance.total_return > self.good_performance_threshold:
            # 历史上表现好
            scale = min(1.2, performance.total_return * 5)  # 最多放大到1.2倍
            return {
                "strategy_allowed": True,
                "position_scale": scale,
                "reason": f"历史上在此状态盈利{performance.total_return:.1%}",
                "confidence": 0.8
            }
        else:
            # 表现一般
            return {
                "strategy_allowed": True,
                "position_scale": 0.5,
                "reason": "历史表现一般，保守仓位",
                "confidence": 0.5
            }

    def update_performance(
        self,
        strategy_type: str,
        market_state: MarketState,
        trade_result: Dict[str, Any]
    ):
        """
        更新策略表现记录

        Args:
            strategy_type: 策略类型
            market_state: 交易时的市场状态
            trade_result: 交易结果 {
                "pnl": float,
                "return": float,
                "is_win": bool
            }
        """
        state_key = self._get_state_key(market_state)
        record = self._performance_history[strategy_type][state_key]

        record.sample_count += 1
        record.total_trades += 1

        if trade_result.get("is_win"):
            record.winning_trades += 1
        else:
            record.losing_trades += 1

        record.total_return += trade_result.get("return", 0.0)
        # 简化，实际应该计算最大回撤
        record.max_drawdown = min(record.max_drawdown, trade_result.get("drawdown", 0.0))

        # 更新 Sharpe（简化）
        if record.sample_count > 1:
            avg_return = record.total_return / record.sample_count
            # 假设无风险利率为0
            record.sharpe_ratio = avg_return / 0.1  # 简化分母


class DynamicParameterManager:
    """
    动态参数管理器

    根据市场状态动态调整策略参数
    """

    def __init__(self):
        self._parameter_rules: Dict[str, Dict[str, Any]] = {
            "ma": {
                "trend": {
                    "fast_period": 5,
                    "slow_period": 20
                },
                "range": {
                    "fast_period": 8,
                    "slow_period": 25  # 震荡时使用更长周期
                },
                "high_vol": {
                    "fast_period": 10,
                    "slow_period": 30  # 高波动使用更长周期
                }
            },
            "breakout": {
                "trend": {
                    "period": 20,
                    "threshold_multiplier": 1.5
                },
                "range": {
                    "period": 30,
                    "threshold_multiplier": 2.0  # 震荡时提高阈值
                },
                "high_vol": {
                    "period": 25,
                    "threshold_multiplier": 2.5
                }
            },
            "ml": {
                "confidence_threshold": {
                    "trend": 0.5,
                    "range": 0.7,    # 震荡时提高置信度阈值
                    "high_vol": 0.8  # 高波动时提高置信度阈值
                }
            }
        }

    def get_parameters(
        self,
        strategy_type: str,
        market_state: MarketState
    ) -> Dict[str, Any]:
        """
        获取动态参数

        Args:
            strategy_type: 策略类型
            market_state: 当前市场状态

        Returns:
            参数字典
        """
        # 确定参数配置集
        if market_state.regime.value == "CRISIS":
            config_set = "high_vol"  # 危机模式使用高波动配置
        elif market_state.volatility_state.value == "EXTREME":
            config_set = "high_vol"
        elif market_state.regime.value == "RANGE":
            config_set = "range"
        else:
            config_set = "trend"

        strategy_params = self._parameter_rules.get(strategy_type, {})
        params = strategy_params.get(config_set, strategy_params.get("trend", {}))

        return params.copy()


if __name__ == "__main__":
    """测试代码"""
    from skills.market_analysis.market_state import MarketState, RegimeType, VolatilityState, LiquidityState, SentimentState

    print("=== Strategy Feasibility Evaluator 测试 ===\n")

    # 创建测试市场状态
    market_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime=RegimeType.RANGE,
        volatility_state=VolatilityState.NORMAL,
        liquidity_state=LiquidityState.NORMAL,
        sentiment_state=SentimentState.FEAR
    )

    print("1. 规则基评估:")
    evaluator = StrategyFeasibilityEvaluator()
    result = evaluator.evaluate(market_state, "ma")
    print(f"   MA策略允许: {result['strategy_allowed']}")
    print(f"   仓位缩放: {result['position_scale']:.2f}")
    print(f"   原因: {result['reason']}")

    print("\n2. 危机模式评估:")
    crisis_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime=RegimeType.CRISIS,
        volatility_state=VolatilityState.EXTREME,
        liquidity_state=LiquidityState.THIN,
        sentiment_state=SentimentState.FEAR
    )

    result = evaluator.evaluate(crisis_state, "ma")
    print(f"   MA策略允许: {result['strategy_allowed']}")
    print(f"   仓位缩放: {result['position_scale']:.2f}")

    print("\n3. 动态参数:")
    param_mgr = DynamicParameterManager()
    params = param_mgr.get_parameters("ma", market_state)
    print(f"   MA参数: {params}")

    trend_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime=RegimeType.TREND,
        volatility_state=VolatilityState.NORMAL
    )
    trend_params = param_mgr.get_parameters("ma", trend_state)
    print(f"   趋势市MA参数: {trend_params}")

    print("\n✓ 所有测试通过")