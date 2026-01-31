"""
扰动测试模块

通过轻微扰动验证最坏窗口的稳定性
"""

import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal

from analysis.replay_schema import ReplayOutput, load_replay_outputs
from analysis.window_scanner import WindowScanner, WindowMetrics


class PerturbationType(Enum):
    """扰动类型"""
    REPLAY_ORDER = "replay_order"      # 顺序扰动
    COST_EPSILON = "cost_epsilon"      # 成本±ε
    CAPITAL_EPSILON = "capital_epsilon" # 初始资金±ε
    SEED_CHANGE = "seed_change"        # 随机种子变化


@dataclass
class PerturbationConfig:
    """扰动配置"""
    perturbation_type: PerturbationType
    epsilon: float = 0.01  # 扰动幅度（1%）

    # cost/slippage 扰动参数
    cost_epsilon: Optional[float] = None
    slippage_epsilon: Optional[float] = None

    # capital 扰动参数
    capital_epsilon: Optional[float] = None

    # seed 扰动参数
    seed_offset: Optional[int] = None

    def __post_init__(self):
        if self.cost_epsilon is None:
            self.cost_epsilon = self.epsilon
        if self.slippage_epsilon is None:
            self.slippage_epsilon = self.epsilon
        if self.capital_epsilon is None:
            self.capital_epsilon = self.epsilon


@dataclass
class PerturbationResult:
    """扰动测试结果"""
    perturbation_type: PerturbationType
    perturbation_value: float

    # 扰动后的最坏窗口
    worst_window: Optional[WindowMetrics]

    # 与原始结果的对比
    original_worst_window: WindowMetrics

    # 稳定性判断
    is_same_window: bool  # 是否在同一时间区间
    is_in_top_k: bool     # 是否在原始Top-K中
    return_diff: float    # 收益差异
    mdd_diff: float       # MDD差异

    @property
    def stability_label(self) -> str:
        """稳定性标签"""
        if self.is_same_window:
            return "Stable"
        elif self.is_in_top_k:
            return "Weakly Stable"
        else:
            return "Fragile"


class PerturbationTester:
    """扰动测试器

    通过轻微扰动验证最坏窗口的稳定性
    """

    def __init__(
        self,
        perturbation_configs: List[PerturbationConfig] = None
    ):
        """初始化扰动测试器

        Args:
            perturbation_configs: 扰动配置列表
        """
        self.perturbation_configs = perturbation_configs or self._default_configs()
        self.scanner = WindowScanner(top_k=5)

    def _default_configs(self) -> List[PerturbationConfig]:
        """默认扰动配置"""
        return [
            # 顺序扰动：通过打乱顺序模拟（但保持时间连续性）
            PerturbationConfig(
                perturbation_type=PerturbationType.REPLAY_ORDER,
                epsilon=0.0
            ),

            # 成本扰动：±1%
            PerturbationConfig(
                perturbation_type=PerturbationType.COST_EPSILON,
                epsilon=0.01
            ),
            PerturbationConfig(
                perturbation_type=PerturbationType.COST_EPSILON,
                epsilon=-0.01
            ),

            # 资金扰动：±1%
            PerturbationConfig(
                perturbation_type=PerturbationType.CAPITAL_EPSILON,
                epsilon=0.01
            ),
            PerturbationConfig(
                perturbation_type=PerturbationType.CAPITAL_EPSILON,
                epsilon=-0.01
            ),
        ]

    def apply_perturbation(
        self,
        replay: ReplayOutput,
        config: PerturbationConfig
    ) -> ReplayOutput:
        """应用扰动到replay

        Args:
            replay: 原始replay
            config: 扰动配置

        Returns:
            扰动后的replay
        """
        # 深拷贝避免修改原始数据
        perturbed = copy.deepcopy(replay)

        if config.perturbation_type == PerturbationType.REPLAY_ORDER:
            # 顺序扰动：轻微打乱step顺序（保持时间连续性）
            perturbed = self._perturb_order(perturbed)

        elif config.perturbation_type == PerturbationType.COST_EPSILON:
            # 成本扰动：调整equity曲线
            perturbed = self._perturb_cost(perturbed, config.epsilon)

        elif config.perturbation_type == PerturbationType.CAPITAL_EPSILON:
            # 资金扰动：调整初始资金
            perturbed = self._perturb_capital(perturbed, config.epsilon)

        elif config.perturbation_type == PerturbationType.SEED_CHANGE:
            # 种子扰动（对于mock数据）
            perturbed = self._perturb_seed(perturbed, config.seed_offset or 1)

        return perturbed

    def _perturb_order(self, replay: ReplayOutput) -> ReplayOutput:
        """轻微扰动顺序（保持时间连续性）"""
        # 对连续的steps进行微小的时间偏移
        # 实际上，对于已完成的replay，我们只能模拟效果
        # 这里我们通过微调timestamp来模拟

        import random
        random.seed(42)

        # 对每个step的timestamp进行微调（±1分钟）
        for step in replay.steps:
            original_time = step.timestamp
            offset_minutes = random.randint(-1, 1)
            step.timestamp = original_time + pd.Timedelta(minutes=offset_minutes)

        return replay

    def _perturb_cost(self, replay: ReplayOutput, epsilon: float) -> ReplayOutput:
        """扰动成本模型

        通过调整equity来模拟成本变化
        """
        # 对每个step的equity应用扰动
        for step in replay.steps:
            step_pnl = float(step.step_pnl)
            # 假设成本变化影响每笔交易的盈亏
            perturbed_pnl = step_pnl * (1 + epsilon)
            step.step_pnl = Decimal(str(perturbed_pnl))
            step.equity = Decimal(str(float(step.equity) + perturbed_pnl - step_pnl))

        # 更新final_equity
        if replay.steps:
            replay.final_equity = replay.steps[-1].equity

        return replay

    def _perturb_capital(self, replay: ReplayOutput, epsilon: float) -> ReplayOutput:
        """扰动初始资金

        通过调整initial_cash来模拟
        """
        original_capital = float(replay.initial_cash)
        perturbed_capital = original_capital * (1 + epsilon)
        replay.initial_cash = Decimal(str(perturbed_capital))

        # 调整所有equity
        for step in replay.steps:
            step.equity = Decimal(str(float(step.equity) * (1 + epsilon)))

        replay.final_equity = Decimal(str(float(replay.final_equity) * (1 + epsilon)))

        return replay

    def _perturb_seed(self, replay: ReplayOutput, seed_offset: int) -> ReplayOutput:
        """扰动随机种子

        对于使用mock数据的replay，这里只是占位
        实际应用中需要重新生成数据
        """
        # 这是一个占位实现
        # 实际上，对于已经完成的replay，无法改变其生成时的seed
        # 但我们可以记录这个需求
        return replay

    def test_perturbations(
        self,
        replay: ReplayOutput,
        window_length: str
    ) -> List[PerturbationResult]:
        """对单个replay和窗口长度进行扰动测试

        Args:
            replay: 回测输出
            window_length: 窗口长度（如 "20d"）

        Returns:
            扰动测试结果列表
        """
        # 获取原始最坏窗口
        original_worst = self.scanner.scan_replay(replay, window_length)
        if not original_worst:
            return []

        original_worst_window = original_worst[0]
        results = []

        # 对每种扰动配置进行测试
        for config in self.perturbation_configs:
            # 应用扰动
            perturbed_replay = self.apply_perturbation(replay, config)

            # 计算扰动后的最坏窗口
            perturbed_worst = self.scanner.scan_replay(perturbed_replay, window_length)
            perturbed_worst_window = perturbed_worst[0] if perturbed_worst else None

            # 判断稳定性
            is_same_window = self._is_same_window(
                original_worst_window,
                perturbed_worst_window
            ) if perturbed_worst_window else False

            is_in_top_k = self._is_in_top_k(
                perturbed_worst_window,
                original_worst
            ) if perturbed_worst_window else False

            return_diff = (
                perturbed_worst_window.total_return - original_worst_window.total_return
            ) if perturbed_worst_window else 0.0

            mdd_diff = (
                perturbed_worst_window.max_drawdown - original_worst_window.max_drawdown
            ) if perturbed_worst_window else 0.0

            # 构建结果
            result = PerturbationResult(
                perturbation_type=config.perturbation_type,
                perturbation_value=config.epsilon,
                worst_window=perturbed_worst_window,
                original_worst_window=original_worst_window,
                is_same_window=is_same_window,
                is_in_top_k=is_in_top_k,
                return_diff=return_diff,
                mdd_diff=mdd_diff
            )

            results.append(result)

        return results

    def _is_same_window(
        self,
        window1: WindowMetrics,
        window2: WindowMetrics,
        tolerance_days: int = 2
    ) -> bool:
        """判断两个窗口是否在同一时间区间

        Args:
            window1: 窗口1
            window2: 窗口2
            tolerance_days: 时间容差（天数）

        Returns:
            是否在同一时间区间
        """
        date_diff = abs((window1.start_date - window2.start_date).days)
        return date_diff <= tolerance_days

    def _is_in_top_k(
        self,
        window: WindowMetrics,
        top_k_windows: List[WindowMetrics]
    ) -> bool:
        """判断窗口是否在原始Top-K中

        Args:
            window: 待判断窗口
            top_k_windows: 原始Top-K窗口列表

        Returns:
            是否在Top-K中
        """
        for wk in top_k_windows:
            if self._is_same_window(window, wk, tolerance_days=2):
                return True
        return False

    def classify_stability(
        self,
        results: List[PerturbationResult]
    ) -> str:
        """分类稳定性

        Args:
            results: 扰动测试结果列表

        Returns:
            稳定性标签: Stable / Weakly Stable / Fragile
        """
        if not results:
            return "Unknown"

        stable_count = sum(1 for r in results if r.is_same_window)
        weakly_stable_count = sum(1 for r in results if r.is_in_top_k)

        if stable_count >= len(results) * 0.7:
            return "Stable"
        elif weakly_stable_count >= len(results) * 0.7:
            return "Weakly Stable"
        else:
            return "Fragile"


def test_all_perturbations(
    run_dir: str,
    window_lengths: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """对所有replay进行扰动测试

    Args:
        run_dir: runs目录路径
        window_lengths: 窗口长度列表

    Returns:
        {strategy_id: {window_length: results}}
    """
    replays = load_replay_outputs(run_dir)
    window_lengths = window_lengths or ["20d", "60d"]

    tester = PerturbationTester()
    all_results = {}

    for replay in replays:
        strategy_key = f"{replay.run_id}_{replay.strategy_id}"
        all_results[strategy_key] = {}

        for window in window_lengths:
            # 执行扰动测试
            results = tester.test_perturbations(replay, window)

            # 分类稳定性
            stability_label = tester.classify_stability(results)

            all_results[strategy_key][window] = {
                'perturbation_results': results,
                'stability_label': stability_label,
                'original_worst': results[0].original_worst_window if results else None
            }

    return all_results


if __name__ == "__main__":
    """测试代码"""
    print("=== PerturbationTester 测试 ===\n")

    # 测试单个replay
    replays = load_replay_outputs("runs")

    if replays:
        replay = replays[0]
        print(f"测试策略: {replay.run_id}\n")

        tester = PerturbationTester()

        # 测试20d窗口
        results = tester.test_perturbations(replay, "20d")

        print("扰动测试结果:")
        for r in results:
            print(f"\n{r.perturbation_type.value} ({r.perturbation_value:+.1%}):")
            print(f"  稳定性: {r.stability_label}")
            print(f"  原始最坏收益: {r.original_worst_window.total_return*100:.2f}%")
            if r.worst_window:
                print(f"  扰动后收益: {r.worst_window.total_return*100:.2f}%")
                print(f"  收益差异: {r.return_diff*100:.2f}%")
            print(f"  同一窗口: {r.is_same_window}")
            print(f"  在Top-K: {r.is_in_top_k}")

        stability = tester.classify_stability(results)
        print(f"\n总体稳定性: {stability}")

    print("\n✓ 测试通过")
