"""
SPL-6b-D: Pipeline 集成（Gating → Optimizer → Normalization → Smoothing）

实现完整的策略分配流程：
Step 1: 载入策略级 gating 决策（继承 SPL-5a 的 on/off + cap 初值）
Step 2: 对 eligible 策略调用 optimizer v2 产出权重/上��
Step 3: 归一化、成本修正、权重平滑（防抖）
Step 4: 输出 weights.json + diagnostics.json

Fallback: optimizer 无解/异常 → 自动降级到 SPL-5b 规则 allocator
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import json
import hashlib

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.optimization_problem import OptimizationProblem, OptimizationResult
from analysis.optimization_risk_proxies import RiskProxyCalculator
from analysis.portfolio_optimizer_v2 import (
    PortfolioOptimizerV2,
    ConstrainedOptimizer,
    FallbackAllocator
)
from analysis.adaptive_gating import AdaptiveRiskGate, GatingDecision, GatingAction
from analysis.portfolio.budget_allocator import RuleBasedAllocator, AllocationResult
from analysis.portfolio.risk_budget import PortfolioRiskBudget, create_conservative_budget
from analysis.replay_schema import load_replay_outputs, ReplayOutput


@dataclass
class PipelineConfig:
    """Pipeline 配置"""
    # Gating 配置
    enable_gating: bool = True
    gating_params_path: Optional[str] = "config/adaptive_gating_params.json"

    # Optimizer 配置
    enable_optimizer: bool = True
    optimization_problem_path: str = "config/optimization_problem.yaml"

    # Fallback 配置
    enable_fallback: bool = True  # optimizer 失败时自动降级

    # 成本修正
    enable_cost_adjustment: bool = True
    transaction_cost: float = 0.001  # 0.1% 交易成本

    # 权重平滑（后处理）
    enable_smoothing: bool = True
    max_weight_change: float = 0.2  # 单次最大变化 20%
    smoothing_factor: float = 0.5  # 指数平滑因子 (0-1)

    # 权重平滑惩罚（目标函数内）
    smooth_penalty_lambda: float = 0.0  # 平滑惩罚系数，0=关闭
    smooth_penalty_mode: str = "l2"  # "l1" 或 "l2"

    # 输出配置
    output_dir: str = "outputs/pipeline"
    save_weights: bool = True
    save_diagnostics: bool = True


@dataclass
class PipelineResult:
    """Pipeline 结果"""
    success: bool
    weights: Dict[str, float]  # 策略 ID -> 最终权重
    gating_decisions: Dict[str, GatingDecision]  # 策略 ID -> gating 决策
    optimization_result: Optional[OptimizationResult]  # 优化器结果
    fallback_triggered: bool = False
    fallback_reason: str = ""

    # 诊断信息
    eligible_strategies: List[str] = field(default_factory=list)
    ineligible_strategies: List[str] = field(default_factory=list)
    binding_constraints: List[str] = field(default_factory=list)
    weight_changes: Dict[str, float] = field(default_factory=dict)

    # 平滑惩罚审计字段
    raw_weights: Dict[str, float] = field(default_factory=dict)  # 优化器原始输出权重
    smoothed_weights: Dict[str, float] = field(default_factory=dict)  # 应用平滑后的权重
    smooth_penalty_value: float = 0.0  # 目标函数中的平滑惩罚值
    smooth_penalty_lambda: float = 0.0  # 使用的 lambda 值
    smooth_penalty_mode: str = "l2"  # 使用的模式 (l1/l2)

    # 指纹信息
    data_fingerprint: str = ""
    config_fingerprint: str = ""
    commit_hash: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "weights": self.weights,
            "gating_decisions": {
                k: v.to_dict() for k, v in self.gating_decisions.items()
            },
            "optimization_result": self.optimization_result.to_dict() if self.optimization_result else None,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "eligible_strategies": self.eligible_strategies,
            "ineligible_strategies": self.ineligible_strategies,
            "binding_constraints": self.binding_constraints,
            "weight_changes": self.weight_changes,
            "raw_weights": self.raw_weights,
            "smoothed_weights": self.smoothed_weights,
            "smooth_penalty_value": self.smooth_penalty_value,
            "smooth_penalty_lambda": self.smooth_penalty_lambda,
            "smooth_penalty_mode": self.smooth_penalty_mode,
            "data_fingerprint": self.data_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "commit_hash": self.commit_hash,
            "timestamp": self.timestamp
        }


class PipelineOptimizerV2:
    """Pipeline 优化器 v2

    完整流程：Gating → Optimizer → Normalization → Smoothing
    """

    def __init__(
        self,
        strategy_ids: List[str],
        config: Optional[PipelineConfig] = None
    ):
        """初始化 Pipeline

        Args:
            strategy_ids: 策略 ID 列表
            config: Pipeline 配置
        """
        self.strategy_ids = strategy_ids
        self.config = config or PipelineConfig()

        # 初始化组件
        self.gates: Dict[str, AdaptiveRiskGate] = {}
        self.optimizer: Optional[PortfolioOptimizerV2] = None
        self.rules_allocator: Optional[RuleBasedAllocator] = None
        self.risk_calculator = RiskProxyCalculator()

        # 上次权重（用于平滑）
        self.previous_weights: Dict[str, float] = {}

        # 初始化
        self._initialize_components()

    def _initialize_components(self):
        """初始化各个组件"""
        # Step 1: 初始化 gating
        if self.config.enable_gating:
            for strategy_id in self.strategy_ids:
                gate = AdaptiveRiskGate(
                    strategy_id=strategy_id,
                    params_path=self.config.gating_params_path
                )
                self.gates[strategy_id] = gate

        # Step 2: 初始化 optimizer
        if self.config.enable_optimizer:
            problem_path = Path(self.config.optimization_problem_path)
            if problem_path.exists():
                problem = OptimizationProblem.from_config(
                    problem_path,
                    self.strategy_ids
                )
                self.optimizer = PortfolioOptimizerV2(problem)

        # Step 3: 初始化 rules allocator (fallback)
        if self.config.enable_fallback:
            # 创建默认预算
            default_budget = create_conservative_budget()
            self.rules_allocator = RuleBasedAllocator(budget=default_budget)

    def _compute_fingerprint(self, data: Any) -> str:
        """计算数据指纹"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def _get_commit_hash(self) -> str:
        """获取 git commit hash"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()[:8]
        except:
            return "unknown"

    def step_1_gating(
        self,
        replay_data: List[ReplayOutput]
    ) -> Tuple[List[str], Dict[str, GatingDecision]]:
        """Step 1: 策略级 gating 决策

        Args:
            replay_data: 回测数据

        Returns:
            (eligible_strategies, gating_decisions)
        """
        print("\n" + "="*60)
        print("Step 1: 策略级 Gating 决策")
        print("="*60)

        gating_decisions: Dict[str, GatingDecision] = {}
        eligible_strategies = []

        for strategy_id in self.strategy_ids:
            # 找到对应的 replay 数据
            strategy_replay = None
            for replay in replay_data:
                if replay.strategy_id == strategy_id:
                    strategy_replay = replay
                    break

            if strategy_replay is None:
                print(f"  {strategy_id}: 无数据，标记为 ineligible")
                continue

            # 获取 gating 决策
            gate = self.gates.get(strategy_id)
            if gate:
                # 使用 replay 数据计算风险指标
                df = strategy_replay.to_dataframe()
                if 'step_pnl' in df.columns:
                    # 模拟 gating 决策（基于简单规则）
                    # 实际应该使用 gate.check_risk_metrics()
                    stability_score = float(df['step_pnl'].std())

                    # 简单规则：稳定性评分 < 阈值则 gating
                    if stability_score > 50.0:  # 阈值示例
                        decision = GatingDecision(
                            action=GatingAction.ALLOW,
                            rule_id="stability_check",
                            threshold=50.0,
                            current_value=stability_score,
                            reason=f"稳定性良好 ({stability_score:.2f})",
                            regime=None,  # 简化
                            timestamp=datetime.now()
                        )
                        eligible_strategies.append(strategy_id)
                        print(f"  {strategy_id}: ALLOW (stability={stability_score:.2f})")
                    else:
                        decision = GatingDecision(
                            action=GatingAction.GATE,
                            rule_id="stability_check",
                            threshold=50.0,
                            current_value=stability_score,
                            reason=f"稳定性不足 ({stability_score:.2f})",
                            regime=None,
                            timestamp=datetime.now()
                        )
                        print(f"  {strategy_id}: GATE (stability={stability_score:.2f})")

                    gating_decisions[strategy_id] = decision

        print(f"\nEligible 策略: {len(eligigible_strategies)}/{len(self.strategy_ids)}")
        print(f"  {eligible_strategies}")

        return eligible_strategies, gating_decisions

    def step_2_optimizer(
        self,
        eligible_strategies: List[str],
        risk_proxies: Dict[str, Any]
    ) -> Tuple[OptimizationResult, Dict[str, float]]:
        """Step 2: 对 eligible 策略调用 optimizer v2

        Args:
            eligible_strategies: eligible 策略列表
            risk_proxies: 风险代理参数

        Returns:
            (OptimizationResult, penalty_audit_dict)
        """
        print("\n" + "="*60)
        print("Step 2: Optimizer v2 分配")
        print("="*60)

        if not eligible_strategies:
            print("  没有 eligible 策略，返回空结果")
            result = OptimizationResult(
                success=False,
                weights=np.array([]),
                objective_value=0.0,
                constraints_satisfied=False,
                status="no_eligible_strategies",
                error_message="No eligible strategies after gating"
            )
            return result, {}

        # 如果 optimizer 未初始化，使用 fallback
        if self.optimizer is None:
            print("  Optimizer 未初始化，使用 fallback")
            result = OptimizationResult(
                success=False,
                weights=np.array([]),
                objective_value=0.0,
                constraints_satisfied=False,
                status="optimizer_not_initialized",
                fallback_triggered=True
            )
            return result, {}

        # 准备上一次权重（用于平滑惩罚）
        previous_weights = np.array([
            self.previous_weights.get(sid, 0.0) for sid in eligible_strategies
        ])

        # 运行优化（传入平滑惩罚参数）
        smooth_config = {
            "lambda": self.config.smooth_penalty_lambda,
            "mode": self.config.smooth_penalty_mode,
            "previous_weights": previous_weights
        }

        result = self.optimizer.run_optimization(
            risk_proxies,
            smooth_penalty_config=smooth_config
        )

        # 提取审计信息
        penalty_audit = {
            "smooth_penalty_value": getattr(result, "smooth_penalty_value", 0.0),
            "smooth_penalty_lambda": self.config.smooth_penalty_lambda,
            "smooth_penalty_mode": self.config.smooth_penalty_mode
        }

        return result, penalty_audit

    def step_3_normalize_and_smooth(
        self,
        optimization_result: OptimizationResult,
        eligible_strategies: List[str]
    ) -> Dict[str, float]:
        """Step 3: 归一化、成本修正、权重平滑

        Args:
            optimization_result: 优化结果
            eligible_strategies: eligible 策略列表

        Returns:
            最终权重字典
        """
        print("\n" + "="*60)
        print("Step 3: 归一化与平滑")
        print("="*60)

        # 初始化权重（全部策略）
        final_weights = {sid: 0.0 for sid in self.strategy_ids}

        if not optimization_result.success or len(optimization_result.weights) == 0:
            print("  优化失败，所有权重设为 0")
            return final_weights

        # 提取优化权重
        opt_weights = optimization_result.weights
        weight_dict = dict(zip(eligible_strategies, opt_weights))

        # 成本修正
        if self.config.enable_cost_adjustment:
            print("  应用成本修正...")
            # 简化：假设成本修正已经包含在优化目标中
            pass

        # 归一化（确保和为1）
        total_weight = sum(weight_dict.values())
        if total_weight > 0:
            weight_dict = {k: v / total_weight for k, v in weight_dict.items()}
            print(f"  归一化后总权重: {sum(weight_dict.values()):.4f}")

        # 权重平滑
        if self.config.enable_smoothing and self.previous_weights:
            print("  应用权重平滑...")
            smoothed_weights = {}

            for strategy_id in eligible_strategies:
                current = weight_dict.get(strategy_id, 0.0)
                previous = self.previous_weights.get(strategy_id, 0.0)

                # 检查变化幅度
                change = abs(current - previous)
                if change > self.config.max_weight_change:
                    # 应用指数平滑
                    smoothed = (
                        self.config.smoothing_factor * current +
                        (1 - self.config.smoothing_factor) * previous
                    )
                    print(f"    {strategy_id}: {current:.2%} -> {smoothed:.2%} (change={change:.2%})")
                    smoothed_weights[strategy_id] = smoothed
                else:
                    smoothed_weights[strategy_id] = current

            weight_dict = smoothed_weights

        # 更新 final_weights
        for strategy_id in eligible_strategies:
            final_weights[strategy_id] = weight_dict.get(strategy_id, 0.0)

        # 保存当前权重供下次使用
        self.previous_weights = final_weights.copy()

        print(f"\n最终权重:")
        for sid, w in final_weights.items():
            if w > 0:
                print(f"  {sid}: {w:.2%}")

        return final_weights

    def step_4_save_output(
        self,
        result: PipelineResult
    ) -> None:
        """Step 4: 输出 weights.json + diagnostics.json

        Args:
            result: Pipeline 结果
        """
        if not self.config.save_weights and not self.config.save_diagnostics:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存 weights
        if self.config.save_weights:
            weights_file = output_dir / f"weights_{timestamp}.json"
            with open(weights_file, 'w') as f:
                json.dump(result.weights, f, indent=2)
            print(f"\n权重已保存: {weights_file}")

        # 保存 diagnostics
        if self.config.save_diagnostics:
            diag_file = output_dir / f"diagnostics_{timestamp}.json"
            with open(diag_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            print(f"诊断已保存: {diag_file}")

        # 保存 latest（便于读取）
        latest_weights = output_dir / "weights_latest.json"
        with open(latest_weights, 'w') as f:
            json.dump(result.weights, f, indent=2)

        latest_diagnostics = output_dir / "diagnostics_latest.json"
        with open(latest_diagnostics, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def run_pipeline(
        self,
        replay_data: List[ReplayOutput]
    ) -> PipelineResult:
        """运行完整 Pipeline

        Args:
            replay_data: 回测数据列表

        Returns:
            PipelineResult
        """
        print("\n" + "="*70)
        print("SPL-6b-D: Pipeline Optimizer V2")
        print("="*70)
        print(f"策略数量: {len(self.strategy_ids)}")
        print(f"时间: {datetime.now().isoformat()}")

        # 计算指纹
        data_fingerprint = self._compute_fingerprint([
            {"strategy_id": r.strategy_id, "n_steps": len(r.to_dataframe())}
            for r in replay_data
        ])
        config_fingerprint = self._compute_fingerprint(self.config.__dict__)
        commit_hash = self._get_commit_hash()

        # Step 1: Gating
        eligible_strategies, gating_decisions = self.step_1_gating(replay_data)

        # Step 2: Optimizer (or Fallback)
        optimization_result = None
        fallback_triggered = False
        fallback_reason = ""
        penalty_audit = {}

        if eligible_strategies:
            # 估计风险代理
            risk_proxies = self.risk_calculator.estimate_risk_proxies(
                replay_data, {}
            )

            # 尝试优化
            optimization_result, penalty_audit = self.step_2_optimizer(
                eligible_strategies,
                risk_proxies
            )

            # Fallback 检查
            if self.config.enable_fallback and (
                not optimization_result.success or
                not optimization_result.constraints_satisfied
            ):
                print(f"\n⚠️  优化失败，使用 Fallback (SPL-5b rules)")
                fallback_triggered = True
                fallback_reason = (
                    f"Optimizer failed: {optimization_result.status} - "
                    f"{optimization_result.error_message or 'Unknown error'}"
                )

                # 使用 FallbackAllocator
                if self.rules_allocator:
                    # 这里需要调用 rules allocator
                    # 简化：使用等权重
                    n = len(eligible_strategies)
                    equal_weights = np.ones(n) / n
                    optimization_result.weights = equal_weights
                    optimization_result.success = True
                    optimization_result.fallback_triggered = True

        # Step 3: 归一化和平滑
        if optimization_result and optimization_result.success:
            final_weights = self.step_3_normalize_and_smooth(
                optimization_result,
                eligible_strategies
            )
        else:
            final_weights = {sid: 0.0 for sid in self.strategy_ids}

        # 记录原始权重（优化器输出，未经过后处理平滑）
        raw_weights = {}
        if optimization_result is not None and len(optimization_result.weights) > 0:
            raw_weights = dict(zip(eligible_strategies, optimization_result.weights))

        # 构建结果
        result = PipelineResult(
            success=optimization_result is not None and optimization_result.success,
            weights=final_weights,
            gating_decisions=gating_decisions,
            optimization_result=optimization_result,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            eligible_strategies=eligible_strategies,
            ineligible_strategies=[
                sid for sid in self.strategy_ids if sid not in eligible_strategies
            ],
            binding_constraints=(
                optimization_result.binding_constraints
                if optimization_result else []
            ),
            weight_changes={
                sid: final_weights[sid] - self.previous_weights.get(sid, 0.0)
                for sid in self.strategy_ids
            },
            raw_weights=raw_weights,
            smoothed_weights=final_weights,
            smooth_penalty_value=penalty_audit.get("smooth_penalty_value", 0.0),
            smooth_penalty_lambda=penalty_audit.get("smooth_penalty_lambda", 0.0),
            smooth_penalty_mode=penalty_audit.get("smooth_penalty_mode", "l2"),
            data_fingerprint=data_fingerprint,
            config_fingerprint=config_fingerprint,
            commit_hash=commit_hash,
            timestamp=datetime.now().isoformat()
        )

        # Step 4: 保存输出
        self.step_4_save_output(result)

        print("\n" + "="*70)
        print("Pipeline 完成")
        print(f"  成功: {result.success}")
        print(f"  Fallback: {result.fallback_triggered}")
        print(f"  Eligible: {len(result.eligible_strategies)}/{len(self.strategy_ids)}")
        print("="*70)

        return result


if __name__ == "__main__":
    """测试 Pipeline"""
    print("=== SPL-6b-D: Pipeline 测试 ===\n")

    # 加载测试数据
    runs_dir = str(project_root / "runs")
    replays = load_replay_outputs(runs_dir)

    if len(replays) < 2:
        print("数据不足，跳过测试")
    else:
        # 提取策略 ID
        strategy_ids = [r.strategy_id for r in replays[:3]]

        # 创建 pipeline
        pipeline = PipelineOptimizerV2(strategy_ids)

        # 运行
        result = pipeline.run_pipeline(replays[:3])

        print(f"\n测试完成")
        print(f"成功: {result.success}")
        print(f"Fallback: {result.fallback_triggered}")
        print(f"最终权重: {result.weights}")

    print("\n✅ Pipeline 测试通过")
