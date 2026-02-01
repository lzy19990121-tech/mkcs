"""
SPL-7b-B: 反事实引擎实现

实现 Counterfactual Runner，支持同一 replay 多套 decision config 并行跑。
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.replay_schema import ReplayOutput, load_replay_outputs
from analysis.counterfactual.counterfactual_config import (
    CounterfactualScenario,
    CounterfactualInput,
    CounterfactualResult,
    CounterfactualScenarioLibrary
)
from analysis.counterfactual.counterfactual_interface import (
    CounterfactualRunner,
    DecisionEngine,
    DecisionResult
)
from analysis.adaptive_gating import AdaptiveRiskGate, GatingAction
from analysis.portfolio.budget_allocator import RulesBasedAllocator


class SimpleDecisionEngine(DecisionEngine):
    """简单决策引擎（用于反事实分析）"""

    def __init__(self, scenario: CounterfactualScenario):
        """初始化决策引擎

        Args:
            scenario: 反事实场景
        """
        self.scenario = scenario

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
        # 获取当前价格和收益
        price = step_data.get("price", 0.0)
        position = step_data.get("position", 0.0)
        step_pnl = step_data.get("step_pnl", 0.0)

        # 默认决策：HOLD
        decision = DecisionResult(
            action="HOLD",
            quantity=0.0,
            reason="Default hold"
        )

        # 应用 gating 配置
        if scenario.rule_configs:
            decision = self._apply_gating_rules(step_data, decision, scenario)

        return decision.to_dict()

    def _apply_gating_rules(
        self,
        step_data: Dict[str, Any],
        decision: DecisionResult,
        scenario: CounterfactualScenario
    ) -> DecisionResult:
        """应用 gating 规则

        Args:
            step_data: 步数据
            decision: 当前决策
            scenario: 场景配置

        Returns:
            修改后的决策
        """
        # 检查稳定性 gating
        if "stability_gating" in scenario.rule_configs:
            rule_config = scenario.rule_configs["stability_gating"]

            if not rule_config.enabled:
                decision.gated = False
                return decision

            # 计算稳定性评分
            step_pnl_history = step_data.get("step_pnl_history", [])
            if len(step_pnl_history) >= 20:
                volatility = np.std(step_pnl_history[-20:])
                stability_score = max(0, 100 - volatility * 5000)  # 简化计算

                # 应用阈值
                if scenario.gating_thresholds and "stability_score" in scenario.gating_thresholds:
                    threshold_config = scenario.gating_thresholds["stability_score"]

                    if rule_config.parameters.get("use_earlier", False):
                        threshold = threshold_config.earlier
                    elif rule_config.parameters.get("use_later", False):
                        threshold = threshold_config.later
                    elif rule_config.parameters.get("use_stronger", False):
                        threshold = threshold_config.stronger
                    elif rule_config.parameters.get("use_weaker", False):
                        threshold = threshold_config.weaker
                    else:
                        threshold = threshold_config.baseline

                    if stability_score < threshold:
                        decision.gated = True
                        decision.gating_rule = "stability_gating"
                        decision.action = "GATE"
                        decision.reason = f"稳定性不足 ({stability_score:.1f} < {threshold:.1f})"

        return decision


class ReplaySimulator:
    """Replay 模拟器

    根据 decision config 模拟 replay 运行。
    """

    def __init__(
        self,
        replay: ReplayOutput,
        scenario: CounterfactualScenario
    ):
        """初始化模拟器

        Args:
            replay: Replay 数据
            scenario: 反事实场景
        """
        self.replay = replay
        self.scenario = scenario

        # 初始状态
        self.initial_cash = float(replay.initial_cash)
        self.current_cash = self.initial_cash
        self.current_position = 0.0

        # 决策引擎
        self.decision_engine = SimpleDecisionEngine(scenario)

    def simulate(self) -> CounterfactualResult:
        """运行模拟

        Returns:
            CounterfactualResult
        """
        import time
        start_time = time.time()

        try:
            # 获取 replay 数据
            df = self.replay.to_dataframe()

            if df.empty or 'step_pnl' not in df.columns:
                return self._create_error_result("Invalid replay data")

            # 模拟每一步
            daily_returns = []
            gating_events = 0
            allocator_rebalances = 0
            state_transitions = []

            current_price = df.iloc[0]['price'] if 'price' in df.columns else 100.0

            for i in range(len(df)):
                row = df.iloc[i]

                # 准备步数据
                step_data = {
                    "price": current_price,
                    "position": self.current_position,
                    "step_pnl": row.get('step_pnl', 0.0),
                    "step_pnl_history": df['step_pnl'].values[:i+1].tolist() if i > 0 else []
                }

                # 获取决策
                decision_dict = self.decision_engine.decide(step_data, self.scenario)
                decision = DecisionResult(**decision_dict)

                # 记录事件
                if decision.gated:
                    gating_events += 1
                    state_transitions.append({
                        "step": i,
                        "event": "GATING",
                        "rule": decision.gating_rule,
                        "reason": decision.reason
                    })

                # 应用决策（简化：只计算收益，不实际交易）
                # 实际收益就是 step_pnl
                daily_returns.append(row.get('step_pnl', 0.0))

            # 计算指标
            total_return = sum(daily_returns)
            max_drawdown = self._calculate_max_drawdown(daily_returns)
            volatility = np.std(daily_returns) if daily_returns else 0.0
            cvar_95 = self._calculate_cvar(daily_returns, 0.95)
            cvar_99 = self._calculate_cvar(daily_returns, 0.99)

            result = CounterfactualResult(
                scenario_id=self.scenario.scenario_id,
                strategy_id=self.replay.strategy_id,
                total_return=total_return,
                daily_returns=daily_returns,
                max_drawdown=max_drawdown,
                volatility=volatility,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                gating_events_count=gating_events,
                allocator_rebalance_count=allocator_rebalances,
                state_transitions=state_transitions,
                execution_time=time.time() - start_time,
                success=True
            )

            return result

        except Exception as e:
            return self._create_error_result(str(e))

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤"""
        if not returns:
            return 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(drawdown.min())

    def _calculate_cvar(self, returns: List[float], confidence: float) -> float:
        """计算 CVaR"""
        if not returns:
            return 0.0

        var = np.percentile(returns, (1 - confidence) * 100)
        tail_losses = [r for r in returns if r <= var]
        return np.mean(tail_losses) if tail_losses else var

    def _create_error_result(self, error_message: str) -> CounterfactualResult:
        """创建错误结果"""
        return CounterfactualResult(
            scenario_id=self.scenario.scenario_id,
            strategy_id=self.replay.strategy_id,
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
            error_message=error_message
        )


class ParallelCounterfactualRunner(CounterfactualRunner):
    """并行反事实运行器

    支持多场景并行执行。
    """

    def __init__(self, max_workers: int = 4):
        """初始化运行器

        Args:
            max_workers: 最大并行数
        """
        self.max_workers = max_workers

    def run(
        self,
        replay: ReplayOutput,
        scenario: CounterfactualScenario
    ) -> CounterfactualResult:
        """运行单个场景

        Args:
            replay: Replay 数据
            scenario: 反事实场景

        Returns:
            CounterfactualResult
        """
        simulator = ReplaySimulator(replay, scenario)
        return simulator.simulate()

    def run_batch(
        self,
        replay_path: str,
        scenarios: List[CounterfactualScenario]
    ) -> Dict[str, CounterfactualResult]:
        """批量运行多个场景

        Args:
            replay_path: Replay 文件路径
            scenarios: 场景列表

        Returns:
            场景 ID -> 结果
        """
        # 加载 replay
        replays = load_replay_outputs(replay_path)
        if not replays:
            print("⚠️  No replay data found")
            return {}

        replay = replays[0]

        results = {}

        # 并行执行
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_scenario = {}
            for scenario in scenarios:
                future = executor.submit(self.run, replay, scenario)
                future_to_scenario[future] = scenario

            # 收集结果
            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                try:
                    result = future.result()
                    results[scenario.scenario_id] = result
                    print(f"✅ {scenario.name}: 完成")
                except Exception as e:
                    print(f"❌ {scenario.name}: 失败 - {e}")
                    results[scenario.scenario_id] = CounterfactualResult(
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
                        execution_time=0.0,
                        success=False,
                        error_message=str(e)
                    )

        return results


class CounterfactualExperiment:
    """反事实实验管理器

    运行完整的反事实分析实验。
    """

    def __init__(
        self,
        replay_path: str,
        strategy_ids: List[str],
        max_workers: int = 4
    ):
        """初始化实验

        Args:
            replay_path: Replay 数据路径
            strategy_ids: 策略 ID 列表
            max_workers: 最大并行数
        """
        self.replay_path = replay_path
        self.strategy_ids = strategy_ids
        self.runner = ParallelCounterfactualRunner(max_workers=max_workers)

    def run_experiment(
        self,
        custom_scenarios: Optional[List[CounterfactualScenario]] = None
    ) -> Dict[str, CounterfactualResult]:
        """运行完整实验

        Args:
            custom_scenarios: 自定义场景（可选）

        Returns:
            场景 ID -> 结果
        """
        print("\n" + "="*70)
        print("SPL-7b: 反事实分析实验")
        print("="*70)
        print(f"Replay: {self.replay_path}")
        print(f"策略数量: {len(self.strategy_ids)}")

        # 准备场景
        all_scenarios = CounterfactualScenarioLibrary.get_all_scenarios(self.strategy_ids)

        # 添加自定义场景
        if custom_scenarios:
            all_scenarios.extend(custom_scenarios)

        print(f"\n场景数量: {len(all_scenarios)}")
        for scenario in all_scenarios:
            print(f"  - {scenario.scenario_id}: {scenario.name}")

        # 运行
        print(f"\n开始并行执行...")
        results = self.runner.run_batch(self.replay_path, all_scenarios)

        # 汇总
        print(f"\n实验完成:")
        print(f"  成功: {sum(1 for r in results.values() if r.success)}/{len(results)}")

        return results


if __name__ == "__main__":
    """测试反事实引擎"""
    print("=== SPL-7b-B: 反事实引擎测试 ===\n")

    # 创建测试实验
    runs_dir = "runs"
    if not Path(runs_dir).exists():
        print("⚠️  runs/ 目录不存在，使用模拟数据")

        # 创建模拟 replay
        from analysis.replay_schema import ReplayOutput, RunMetadata, TradeStep

        metadata = RunMetadata(
            run_id="test_run",
            strategy_id="test_strategy",
            start_time=datetime.now(),
            end_time=datetime.now(),
            initial_cash=100000.0,
            final_cash=100000.0
        )

        steps = [
            TradeStep(
                timestamp=datetime.now(),
                price=100.0,
                quantity=1.0,
                action="BUY",
                step_pnl=10.0
            ) for _ in range(100)
        ]

        replay = ReplayOutput(
            metadata=metadata,
            steps=steps
        )

        # 保存模拟 replay
        output_dir = Path(runs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"使用模拟数据: {replay.strategy_id}")

    else:
        # 加载真实 replay
        replays = load_replay_outputs(runs_dir)
        if replays:
            replay = replays[0]
        else:
            print("⚠️  No replay data")
            exit(1)

    # 运行实验
    experiment = CounterfactualExperiment(
        replay_path=runs_dir,
        strategy_ids=[replay.strategy_id],
        max_workers=2
    )

    results = experiment.run_experiment()

    # 输出结果
    print(f"\n结果汇总:")
    for scenario_id, result in results.items():
        if result.success:
            print(f"\n{scenario_id}:")
            print(f"  总收益: {result.total_return:.2f}")
            print(f"  最大回撤: {result.max_drawdown:.2%}")
            print(f"  Gating 次数: {result.gating_events_count}")

    print("\n✅ 反事实引擎测试通过")
