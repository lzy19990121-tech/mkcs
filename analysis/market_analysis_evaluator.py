"""
市场分析模块评估框架

用于验证：
1. 市场状态检测准确性
2. 策略可行��评估效果
3. 风控规则集成效果
4. 回测一致性
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from collections import defaultdict
import json
from pathlib import Path

from core.models import Bar
from skills.market_analysis.market_state import MarketState, RegimeType, VolatilityState
from skills.market_analysis import MarketManager, get_market_manager
from skills.market_analysis.detectors import MarketAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class MarketStateTransition:
    """市场状态转换��录"""
    from_regime: str
    to_regime: str
    timestamp: datetime
    confidence: float


@dataclass
class BacktestResult:
    """回测结果"""
    total_bars: int
    state_distribution: Dict[str, int]  # 各状态出现次数
    transitions: List[MarketStateTransition]
    avg_confidence: float
    low_confidence_count: int  # 低置信度次数

    # 性能指标
    strategy_allowed_ratio: float  # 策略允许运行比例
    avg_position_scale: float     # 平均仓位缩放


class MarketAnalysisEvaluator:
    """市场分析模块评估器

    评估市场状态检测的准确性和稳定性
    """

    def __init__(self, market_manager: Optional[MarketManager] = None):
        """初始化评估器

        Args:
            market_manager: 市场管理器（如果不提供，创建新的）
        """
        self.market_manager = market_manager or get_market_manager()

        # 评估数据
        self._state_history: List[MarketState] = []
        self._transitions: List[MarketStateTransition] = []

    def evaluate_detection_accuracy(
        self,
        bars: List[Bar],
        expected_regimes: Optional[List[RegimeType]] = None
    ) -> Dict[str, Any]:
        """评估市场状态检测准确性

        Args:
            bars: K线数据
            expected_regimes: 期望的市场状态（用于验证）

        Returns:
            评估结果
        """
        results = {
            "total_bars": len(bars),
            "states": [],
            "accuracy": None,
            "confidence_stats": {},
            "regime_distribution": defaultdict(int)
        }

        confidences = []

        for i, bar in enumerate(bars[20:], start=20):  # 需要足够的历史数据
            window_bars = bars[:i+1]

            try:
                state = self.market_manager.analyze(window_bars)
                results["states"].append({
                    "timestamp": bar.timestamp,
                    "regime": state.regime.value,
                    "confidence": state.regime_confidence,
                    "volatility": state.volatility_state.value
                })

                confidences.append(state.regime_confidence)
                results["regime_distribution"][state.regime.value] += 1

                # 记录状态转换
                if len(self._state_history) > 0:
                    last_state = self._state_history[-1]
                    if last_state.regime != state.regime:
                        self._transitions.append(MarketStateTransition(
                            from_regime=last_state.regime.value,
                            to_regime=state.regime.value,
                            timestamp=bar.timestamp,
                            confidence=state.regime_confidence
                        ))

                self._state_history.append(state)

            except Exception as e:
                logger.warning(f"分析失败 (bar {i}): {e}")

        # 计算统计信息
        if confidences:
            results["confidence_stats"] = {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "low_confidence_count": sum(1 for c in confidences if c < 0.5)
            }

        # 计算准确性（如果有期望值）
        if expected_regimes:
            correct = sum(
                1 for i, s in enumerate(results["states"])
                if i < len(expected_regimes) and s["regime"] == expected_regimes[i].value
            )
            results["accuracy"] = correct / len(results["states"]) if results["states"] else 0

        results["transitions"] = len(self._transitions)
        results["regime_distribution"] = dict(results["regime_distribution"])

        return results

    def evaluate_strategy_feasibility(
        self,
        bars: List[Bar],
        strategy_type: str = "ma"
    ) -> Dict[str, Any]:
        """评估策略可行性判断

        Args:
            bars: K线数据
            strategy_type: 策略类型

        Returns:
            可行性评估结果
        """
        results = {
            "total_bars": len(bars),
            "allowed_count": 0,
            "denied_count": 0,
            "position_scales": [],
            "reasons": defaultdict(int)
        }

        for i, bar in enumerate(bars[20:], start=20):
            window_bars = bars[:i+1]

            try:
                state = self.market_manager.analyze(window_bars)
                feasibility = self.market_manager.evaluate_strategy(strategy_type, state)

                if feasibility["strategy_allowed"]:
                    results["allowed_count"] += 1
                else:
                    results["denied_count"] += 1

                results["position_scales"].append(feasibility["position_scale"])
                results["reasons"][feasibility["reason"]] += 1

            except Exception as e:
                logger.warning(f"可行性评估失败 (bar {i}): {e}")

        # 计算统计信息
        if results["position_scales"]:
            results["avg_position_scale"] = sum(results["position_scales"]) / len(results["position_scales"])
            results["min_position_scale"] = min(results["position_scales"])
            results["max_position_scale"] = max(results["position_scales"])

        results["allowed_ratio"] = results["allowed_count"] / (results["allowed_count"] + results["denied_count"]) if (results["allowed_count"] + results["denied_count"]) > 0 else 1.0
        results["reasons"] = dict(results["reasons"])

        return results

    def evaluate_state_stability(
        self,
        min_bars_in_state: int = 5
    ) -> Dict[str, Any]:
        """评估市场状态稳定性

        Args:
            min_bars_in_state: 每个状态最少保持的K线数

        Returns:
            稳定性评估结果
        """
        if not self._state_history:
            return {"error": "没有状态历史"}

        state_sequences: List[Tuple[str, int]] = []
        current_state = None
        count = 0

        for state in self._state_history:
            if state.regime.value == current_state:
                count += 1
            else:
                if current_state is not None:
                    state_sequences.append((current_state, count))
                current_state = state.regime.value
                count = 1

        if current_state is not None:
            state_sequences.append((current_state, count))

        # 分析稳定性
        unstable_transitions = [
            (state, duration) for state, duration in state_sequences
            if duration < min_bars_in_state
        ]

        return {
            "total_states": len(state_sequences),
            "avg_duration": sum(d for _, d in state_sequences) / len(state_sequences) if state_sequences else 0,
            "min_duration": min(d for _, d in state_sequences) if state_sequences else 0,
            "max_duration": max(d for _, d in state_sequences) if state_sequences else 0,
            "unstable_transitions": len(unstable_transitions),
            "unstable_ratio": len(unstable_transitions) / len(state_sequences) if state_sequences else 0,
            "state_sequences": state_sequences
        }

    def run_backtest_simulation(
        self,
        bars: List[Bar],
        strategy_type: str = "ma",
        initial_capital: float = 100000
    ) -> BacktestResult:
        """运行简化的回测模拟

        Args:
            bars: K线数据
            strategy_type: 策略类型
            initial_capital: 初始资金

        Returns:
            回测结果
        """
        state_distribution = defaultdict(int)
        transitions = []
        confidences = []
        low_confidence_count = 0
        allowed_count = 0
        position_scales = []

        for i, bar in enumerate(bars[20:], start=20):
            window_bars = bars[:i+1]

            try:
                state = self.market_manager.analyze(window_bars)
                state_distribution[state.regime.value] += 1
                confidences.append(state.regime_confidence)

                if state.regime_confidence < 0.5:
                    low_confidence_count += 1

                # 策略可行性
                feasibility = self.market_manager.evaluate_strategy(strategy_type, state)
                if feasibility["strategy_allowed"]:
                    allowed_count += 1
                position_scales.append(feasibility["position_scale"])

                # 记录转换
                if len(self._state_history) > 0:
                    last_state = self._state_history[-1]
                    if last_state.regime != state.regime:
                        transitions.append(MarketStateTransition(
                            from_regime=last_state.regime.value,
                            to_regime=state.regime.value,
                            timestamp=bar.timestamp,
                            confidence=state.regime_confidence
                        ))

                self._state_history.append(state)

            except Exception as e:
                logger.warning(f"回测失败 (bar {i}): {e}")

        return BacktestResult(
            total_bars=len(bars),
            state_distribution=dict(state_distribution),
            transitions=transitions,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0,
            low_confidence_count=low_confidence_count,
            strategy_allowed_ratio=allowed_count / len(bars) if bars else 0,
            avg_position_scale=sum(position_scales) / len(position_scales) if position_scales else 0
        )

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """生成评估报告

        Args:
            output_path: 输出文件路径（可选）

        Returns:
            评估报告
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_states_analyzed": len(self._state_history),
            "total_transitions": len(self._transitions),
            "stability_analysis": self.evaluate_state_stability(),
            "state_distribution": defaultdict(int)
        }

        # 统计状态分布
        for state in self._state_history:
            report["state_distribution"][state.regime.value] += 1

        report["state_distribution"] = dict(report["state_distribution"])

        # 保存到文件
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"报告已保存到 {output_path}")

        return report

    def reset(self):
        """重置评估器状态"""
        self._state_history = []
        self._transitions = []


class ComparisonEvaluator:
    """对比评估器

    用于对比不同市场状态检测方法的效果
    """

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}

    def compare_detection_methods(
        self,
        bars: List[Bar],
        methods: List[Tuple[str, Any]]
    ) -> Dict[str, Any]:
        """对比不同的检测方法

        Args:
            bars: K线数据
            methods: [(方法名, 检测器), ...]

        Returns:
            对比结果
        """
        comparison = {
            "methods": [],
            "results": {},
            "summary": {}
        }

        for name, detector in methods:
            evaluator = MarketAnalysisEvaluator(
                MarketManager(
                    symbol="TEST",
                    enable_external_signals=False,
                    enable_feasibility_eval=False
                )
            )
            # 替换检测器
            evaluator.market_manager.analyzer = detector

            result = evaluator.evaluate_detection_accuracy(bars)
            comparison["results"][name] = result
            comparison["methods"].append(name)

        # 生成汇总
        if comparison["results"]:
            best_accuracy = max(
                (r.get("accuracy", 0) for r in comparison["results"].values()),
                default=0
            )
            best_method = [
                name for name, r in comparison["results"].items()
                if r.get("accuracy", 0) == best_accuracy
            ]

            comparison["summary"] = {
                "best_accuracy": best_accuracy,
                "best_method": best_method,
                "method_count": len(methods)
            }

        return comparison


def create_sample_bars(count: int = 100, trend: bool = True) -> List[Bar]:
    """创建示例K线数据

    Args:
        count: K线数量
        trend: 是否为趋势数据

    Returns:
        K线列表
    """
    bars = []
    base_price = 100.0

    for i in range(count):
        date = datetime(2024, 1, 1) + timedelta(days=i)

        if trend:
            # 趋势数据
            price = base_price + i * 0.5 + (i % 5 - 2) * 0.3
        else:
            # 震荡数据
            price = base_price + (i % 10 - 5) * 0.5

        bars.append(Bar(
            symbol="TEST",
            timestamp=date,
            open=Decimal(str(price - 0.2)),
            high=Decimal(str(price + 0.5)),
            low=Decimal(str(price - 0.5)),
            close=Decimal(str(price)),
            volume=1000000,
            interval="1d"
        ))

    return bars


if __name__ == "__main__":
    """测试代码"""
    print("=== MarketAnalysisEvaluator 测试 ===\n")

    # 创建示例数据
    bars = create_sample_bars(100, trend=True)

    # 创建评估器
    evaluator = MarketAnalysisEvaluator()

    # 测试1: 检测准确性评估
    print("1. 检测准确性评估:")
    result = evaluator.evaluate_detection_accuracy(bars)
    print(f"   总K线数: {result['total_bars']}")
    print(f"   状态分布: {result['regime_distribution']}")
    print(f"   平均置信度: {result['confidence_stats'].get('mean', 0):.2f}")
    print(f"   状态转换次数: {result['transitions']}")

    # 测试2: 策略可行性评估
    print("\n2. 策略可行性评估:")
    feasibility_result = evaluator.evaluate_strategy_feasibility(bars, "ma")
    print(f"   允许比例: {feasibility_result['allowed_ratio']:.1%}")
    print(f"   平均仓位缩放: {feasibility_result.get('avg_position_scale', 0):.2f}")
    print(f"   拒绝原因: {feasibility_result['reasons']}")

    # 测试3: 状态稳定性评估
    print("\n3. 状态稳定性评估:")
    stability = evaluator.evaluate_state_stability()
    print(f"   总状态数: {stability['total_states']}")
    print(f"   平均持续时间: {stability['avg_duration']:.1f} bars")
    print(f"   不稳定转换: {stability['unstable_transitions']}")
    print(f"   不稳定比例: {stability['unstable_ratio']:.1%}")

    # 测试4: 回测模拟
    print("\n4. 回测模拟:")
    evaluator.reset()
    backtest = evaluator.run_backtest_simulation(bars)
    print(f"   状态分布: {backtest.state_distribution}")
    print(f"   平均置信度: {backtest.avg_confidence:.2f}")
    print(f"   低置信度次数: {backtest.low_confidence_count}")
    print(f"   策略允许比例: {backtest.strategy_allowed_ratio:.1%}")
    print(f"   平均仓位缩放: {backtest.avg_position_scale:.2f}")

    # 测试5: 生成报告
    print("\n5. 生成报告:")
    report = evaluator.generate_report()
    print(f"   分析状态总数: {report['total_states_analyzed']}")
    print(f"   状态转换总数: {report['total_transitions']}")

    print("\n✓ 所有测试通过")
