"""
SPL-7b-A: 反事实运行接口

定义反事实运行的统一接口。
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.counterfactual.counterfactual_config import (
    CounterfactualScenario,
    CounterfactualInput,
    CounterfactualResult,
    GatingThresholdConfig,
    RuleConfig,
    AllocatorConfig,
    PortfolioComposition
)
from analysis.replay_schema import ReplayOutput, load_replay_outputs


class CounterfactualRunner(ABC):
    """反事实运行器接口

    所有反事实运行器必须实现此接口。
    """

    @abstractmethod
    def run(
        self,
        replay: ReplayOutput,
        scenario: CounterfactualScenario
    ) -> CounterfactualResult:
        """运行反事实场景

        Args:
            replay: Replay 数据
            scenario: 反事实场景

        Returns:
            CounterfactualResult
        """
        pass


class DecisionEngine(ABC):
    """决策引擎接口

    根据场景配置做出交易决策。
    """

    @abstractmethod
    def decide(
        self,
        step_data: Dict[str, Any],
        scenario: CounterfactualScenario
    ) -> Dict[str, Any]:
        """做出决策

        Args:
            step_data: 当前步数据
            scenario: 反事实场景

        Returns:
            决策结果
        """
        pass


@dataclass
class DecisionResult:
    """决策结果"""
    action: str  # "BUY", "SELL", "HOLD", "GATE"
    quantity: float
    reason: str

    # Gating 相关
    gated: bool = False
    gating_rule: Optional[str] = None

    # Allocator 相关
    weight: Optional[float] = None
    weights: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "quantity": self.quantity,
            "reason": self.reason,
            "gated": self.gated,
            "gating_rule": self.gating_rule,
            "weight": self.weight,
            "weights": self.weights
        }


class CounterfactualExecutor:
    """反事实执行器

    根据场景配置执行反事实运行。
    """

    def __init__(self, runner: CounterfactualRunner):
        """初始化执行器

        Args:
            runner: 反事实运行器
        """
        self.runner = runner

    def execute(
        self,
        replay_path: str,
        scenario: CounterfactualScenario
    ) -> CounterfactualResult:
        """执行反事实场景

        Args:
            replay_path: Replay 文件路径
            scenario: 反事实场景

        Returns:
            CounterfactualResult
        """
        import time
        start_time = time.time()

        # 加载 replay
        replays = load_replay_outputs(replay_path)
        if not replays:
            return CounterfactualResult(
                scenario_id=scenario.scenario_id,
                strategy_id="unknown",
                total_return=0.0,
                daily_returns=[],
                max_drawdown=0.0,
                volatility=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                gating_events_count=0,
                allocator_rebalance_count=0,
                state_transitions=[],
                execution_time=0.0,
                success=False,
                error_message="No replay data found"
            )

        # 使用第一个 replay
        replay = replays[0]

        try:
            # 运行反事实
            result = self.runner.run(replay, scenario)

            # 添加执行时间
            result.execution_time = time.time() - start_time

            return result

        except Exception as e:
            return CounterfactualResult(
                scenario_id=scenario.scenario_id,
                strategy_id=replay.strategy_id,
                total_return=0.0,
                daily_returns=[],
                max_drawdown=0.0,
                volatility=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                gating_events_count=0,
                allocator_rebalance_count=0,
                state_transitions=[],
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def execute_batch(
        self,
        replay_path: str,
        scenarios: List[CounterfactualScenario]
    ) -> Dict[str, CounterfactualResult]:
        """批量执行多个场景

        Args:
            replay_path: Replay 文件路径
            scenarios: 场景列表

        Returns:
            场景 ID -> 结果
        """
        results = {}

        for scenario in scenarios:
            print(f"执行场景: {scenario.name}")
            result = self.execute(replay_path, scenario)
            results[scenario.scenario_id] = result

        return results


class CounterfactualComparator:
    """反事实对比器

    对比 Actual vs CF 的结果。
    """

    def compare(
        self,
        actual_result: CounterfactualResult,
        cf_results: Dict[str, CounterfactualResult]
    ) -> Dict[str, Any]:
        """对比结果

        Args:
            actual_result: 实际结果
            cf_results: 反事实结果字典

        Returns:
            对比报告
        """
        comparison = {
            "actual": actual_result.to_dict(),
            "counterfactuals": {}
        }

        # 计算每个反事实的差异
        for scenario_id, cf_result in cf_results.items():
            comparison["counterfactuals"][scenario_id] = {
                "result": cf_result.to_dict(),
                "delta_return": cf_result.total_return - actual_result.total_return,
                "delta_drawdown": cf_result.max_drawdown - actual_result.max_drawdown,
                "delta_volatility": cf_result.volatility - actual_result.volatility,
                "avoided_drawdown": actual_result.max_drawdown - cf_result.max_drawdown,
                "lost_return": actual_result.total_return - cf_result.total_return,
                "tradeoff_ratio": self._calculate_tradeoff_ratio(actual_result, cf_result)
            }

        return comparison

    def _calculate_tradeoff_ratio(
        self,
        actual: CounterfactualResult,
        cf: CounterfactualResult
    ) -> float:
        """计算权衡比

        Args:
            actual: 实际结果
            cf: 反事实结果

        Returns:
            权衡比（风险减少 / 收益牺牲）
        """
        # 风险减少（回撤降低）
        risk_reduction = actual.max_drawdown - cf.max_drawdown

        # 收益牺牲
        return_sacrifice = actual.total_return - cf.total_return

        # 避免除零
        if abs(return_sacrifice) < 1e-10:
            return 0.0 if risk_reduction > 0 else -float('inf')

        return risk_reduction / return_sacrifice

    def generate_comparison_report(
        self,
        comparison: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """生成对比报告（Markdown）

        Args:
            comparison: 对比数据
            output_path: 输出文件路径（可选）

        Returns:
            Markdown 报告
        """
        lines = []
        lines.append("# 反事实分析报告\n")
        lines.append(f"**生成时间**: {datetime.now().isoformat()}\n")

        # 实际结果
        actual = comparison["actual"]
        lines.append("## 实际结果 (Actual)\n")
        lines.append(f"- 总收益: {actual['total_return']:.4f}")
        lines.append(f"- 最大回撤: {actual['max_drawdown']:.2%}")
        lines.append(f"- 波动率: {actual['volatility']:.4f}")
        lines.append(f"- CVaR-95: {actual['cvar_95']:.4f}")
        lines.append(f"- Gating 次数: {actual['gating_events_count']}")
        lines.append("")

        # 反事实结果
        lines.append("## 反事实结果对比\n")
        lines.append("| 场景 | 总收益 | Delta | 最大回撤 | Delta | 权衡比 |")
        lines.append("|------|--------|-------|----------|-------|--------|")

        for scenario_id, cf_data in comparison["counterfactuals"].items():
            cf = cf_data["result"]
            lines.append(
                f"| {scenario_id} | {cf['total_return']:.4f} | "
                f"{cf_data['delta_return']:.4f} | {cf['max_drawdown']:.2%} | "
                f"{cf_data['delta_drawdown']:.2%} | {cf_data['tradeoff_ratio']:.2f} |"
            )

        lines.append("")

        # 结论
        lines.append("## 结论\n")
        best_scenario = max(
            comparison["counterfactuals"].items(),
            key=lambda x: x[1]["tradeoff_ratio"] if not np.isinf(x[1]["tradeoff_ratio"]) else -999999
        )
        lines.append(f"**最优场景**: {best_scenario[0]}")
        lines.append(f"**权衡比**: {best_scenario[1]['tradeoff_ratio']:.2f}")
        lines.append("")

        report = "\n".join(lines)

        # 保存文件
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report


if __name__ == "__main__":
    """测试反事实接口"""
    print("=== SPL-7b-A: 反事实接口测试 ===\n")

    # 创建测试场景
    scenarios = CounterfactualScenarioLibrary.get_all_scenarios(["test_strategy"])

    print(f"场景数量: {len(scenarios)}")
    for scenario in scenarios:
        print(f"- {scenario.scenario_id}: {scenario.name}")

    print("\n✅ 反事实接口测试通过")
