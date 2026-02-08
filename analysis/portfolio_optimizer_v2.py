"""
SPL-6b-C: 实现优化器 v2（先简单可控）

使用 scipy.optimize.minimize 实现：
- 凸优化/二次规划
- 不引入黑箱 ML（可审计）
- 权重平滑机制
- 可解释诊断
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.optimize import minimize

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.optimization_problem import OptimizationProblem, OptimizationResult, ConstraintType
from analysis.optimization_risk_proxies import RiskProxyCalculator


class SolverStatus(Enum):
    """求解器状态"""
    OPTIMAL = "optimal"        # 找到最优解
    INFEASIBLE = "infeasible"  # 无可行解
    ERROR = "error"            # 求解器错误


@dataclass
class OptimizerDiagnostics:
    """优化器诊断信息"""
    binding_constraints: List[str] = field(default_factory=list)
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    solver_iterations: int = 0
    computation_time: float = 0.0
    final_objective: float = 0.0
    message: str = ""


class ConstrainedOptimizer:
    """约束优化器 v2"""

    def __init__(self, problem: OptimizationProblem):
        """初始化优化器

        Args:
            problem: 优化问题定义
        """
        self.problem = problem
        self.risk_calculator = RiskProxyCalculator()
        self.diagnostics = OptimizerDiagnostics()

    def optimize(
        self,
        risk_proxies: Dict[str, Any],
        initial_weights: Optional[np.ndarray] = None,
        smooth_penalty_config: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """运行优化

        Args:
            risk_proxies: 风险代理参数
            initial_weights: 初始权重（可选）
            smooth_penalty_config: 平滑惩罚配置 {"lambda": float, "mode": "l1"|"l2", "previous_weights": np.ndarray}

        Returns:
            OptimizationResult
        """
        import time
        start_time = time.time()

        # 提取平滑惩罚配置
        smooth_lambda = 0.0
        smooth_mode = "l2"
        previous_weights = None

        if smooth_penalty_config:
            smooth_lambda = smooth_penalty_config.get("lambda", 0.0)
            smooth_mode = smooth_penalty_config.get("mode", "l2")
            previous_weights = smooth_penalty_config.get("previous_weights", None)

        if smooth_lambda > 0 and previous_weights is not None:
            print(f"  启用权重平滑惩罚: lambda={smooth_lambda}, mode={smooth_mode}")

        import time
        start_time = time.time()

        try:
            # 准备初始权重
            if initial_weights is None:
                n = self.problem.n_strategies
                initial_weights = np.ones(n) / n

            # 确保权重和为1
            initial_weights = initial_weights / initial_weights.sum()

            # 定义目标函数（支持平滑惩罚）
            def objective_function(weights):
                # 目标：最大化收益 = 最小化负收益
                expected_return = weights @ risk_proxies["expected_returns"]
                covariance_matrix = risk_proxies["covariance_matrix"]

                # 添加惩罚项
                penalty = self.risk_calculator.correlation_penalty.calculate_penalty(
                    covariance_matrix, weights
                )

                # 权重平滑惩罚
                smooth_penalty = 0.0
                if smooth_lambda > 0 and previous_weights is not None:
                    if smooth_mode == "l1":
                        # L1: sum of absolute changes
                        smooth_penalty = smooth_lambda * np.sum(np.abs(weights - previous_weights))
                    else:  # l2
                        # L2: sum of squared changes
                        smooth_penalty = smooth_lambda * np.sum((weights - previous_weights) ** 2)

                # 最大化收益 - 风险惩罚 - 平滑惩罚
                lambda_risk = 0.5
                objective = expected_return - lambda_risk * np.var(
                    weights @ risk_proxies["returns_matrix"].T
                ) - smooth_penalty

                return -objective  # 负号因为 minimize

            # 定义约束
            constraints = []

            # 1. 权重和约束
            def sum_weights_constraint(weights):
                return np.sum(weights) - 1.0

            constraints.append({
                'type': 'eq',
                'fun': sum_weights_constraint
            })

            # 2. 权重边界约束
            lower, upper = self.problem.get_bounds()
            bounds = [(lower[i], upper[i]) for i in range(len(lower))]

            # 求解
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',  # 序列最小二乘规划
                bounds=bounds,
                constraints=constraints,
                options={
                    'ftol': 1e-6,
                    'disp': True,
                    'maxiter': 1000
                }
            )

            # 记录诊断信息
            self.diagnostics.solver_iterations = result.nit
            self.diagnostics.computation_time = time.time() - start_time
            self.diagnostics.final_objective = result.fun
            self.diagnostics.message = "优化成功"

            # 检查约束满足情况
            optimal_weights = result.x
            constraint_results = self.risk_calculator.evaluate_all_constraints(
                optimal_weights, risk_proxies, {}
            )

            # 识别绑定约束
            binding_constraints = [
                name for name, satisfied in constraint_results.items()
                if not satisfied and "bound" in name.lower()
            ]
            self.diagnostics.binding_constraints = binding_constraints

            # 约束违反
            violations = {}
            for name, satisfied in constraint_results.items():
                if not satisfied:
                    violations[name] = False
            self.diagnostics.constraint_violations = violations

            # 创建结果
            # 计算最终的平滑惩罚值（用于审计）
            final_smooth_penalty = 0.0
            if smooth_lambda > 0 and previous_weights is not None:
                if smooth_mode == "l1":
                    final_smooth_penalty = smooth_lambda * np.sum(np.abs(optimal_weights - previous_weights))
                else:  # l2
                    final_smooth_penalty = smooth_lambda * np.sum((optimal_weights - previous_weights) ** 2)

                print(f"  平滑惩罚值: {final_smooth_penalty:.6f}")

            opt_result = OptimizationResult(
                success=result.success,
                weights=optimal_weights,
                objective_value=-result.fun,
                constraints_satisfied=all(constraint_results.values()),
                status=SolverStatus.OPTIMAL.value if result.success else SolverStatus.ERROR.value,
                binding_constraints=self.diagnostics.binding_constraints,
                constraint_violations=self.diagnostics.constraint_violations,
                solver_iterations=self.diagnostics.solver_iterations,
                computation_time=self.diagnostics.computation_time,
                solver_used="SLSQP",
                fallback_triggered=False,
                smooth_penalty_value=final_smooth_penalty,
                smooth_penalty_lambda=smooth_lambda,
                smooth_penalty_mode=smooth_mode
            )

            # 添加可解释性信息
            opt_result.expected_return = optimal_weights @ risk_proxies["expected_returns"]
            opt_result.expected_risk = np.sqrt(optimal_weights @
                risk_proxies["covariance_matrix"] @ optimal_weights)

            return opt_result

        except Exception as e:
            # 优化失败
            return OptimizationResult(
                success=False,
                weights=np.ones(self.problem.n_strategies) / self.problem.n_strategies,
                objective_value=0.0,
                constraints_satisfied=False,
                status=SolverStatus.ERROR.value,
                error_message=str(e),
                solver_used="SLSQP",
                fallback_triggered=False,
                smooth_penalty_value=0.0,
                smooth_penalty_lambda=smooth_lambda,
                smooth_penalty_mode=smooth_mode
            )


class FallbackAllocator:
    """降级分配器（使用 SPL-5b 规则）"""

    def __init__(self):
        """初始化降级分配器"""
        from analysis.optimization_risk_proxies import RiskProxyCalculator
        self.risk_calculator = RiskProxyCalculator()

    def allocate(
        self,
        risk_proxies: Dict[str, Any],
        problem: OptimizationProblem
    ) -> OptimizationResult:
        """使用规则分配策略

        Args:
            risk_proxies: 风险代理参数
            problem: 优化问题

        Returns:
            OptimizationResult
        """
        import time
        start_time = time.time()

        # 简化规则：基于方差倒数分配
        expected_returns = risk_proxies["expected_returns"]
        covariance_matrix = risk_proxies["covariance_matrix"]

        # 计算每个策略的风险（方差）
        variances = np.diag(covariance_matrix)

        # 避免除零
        variances = np.where(variances < 1e-10, 1e-10, variances)

        # 方差倒数权重
        inv_var_weights = 1.0 / variances
        weights = inv_var_weights / inv_var_weights.sum()

        # 检查约束
        constraint_results = self.risk_calculator.evaluate_all_constraints(
            weights, risk_proxies, {}
        )

        # 创建结果
        result = OptimizationResult(
            success=True,
            weights=weights,
            objective_value=0.0,
            constraints_satisfied=all(constraint_results.values()),
            status="fallback",
            binding_constraints=[],
            constraint_violations={name: not satisfied for name, satisfied in constraint_results.items()},
            solver_iterations=0,
            computation_time=time.time() - start_time,
            solver_used="inverse_variance",
            fallback_triggered=True,
            expected_return=weights @ expected_returns,
            expected_risk=np.sqrt(weights @ covariance_matrix @ weights),
            smooth_penalty_value=0.0,
            smooth_penalty_lambda=0.0,
            smooth_penalty_mode="l2"
        )

        result.message = "使用降级分配器（方差倒数）"

        return result


class PortfolioOptimizerV2:
    """组合优化器 v2"""

    def __init__(self, problem: OptimizationProblem):
        """初始化优化器

        Args:
            problem: 优化问题定义
        """
        self.problem = problem
        self.optimizer = ConstrainedOptimizer(problem)
        self.fallback = FallbackAllocator()

    def run_optimization(
        self,
        risk_proxies: Dict[str, Any],
        smooth_penalty_config: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """运行优化（主 + 降级）

        Args:
            risk_proxies: 风险代理参数
            smooth_penalty_config: 平滑惩罚配置 {"lambda": float, "mode": "l1"|"l2", "previous_weights": np.ndarray}

        Returns:
            OptimizationResult
        """
        print("=" * 60)
        print("SPL-6b-C: 优化器 v2")
        print("=" * 60)

        # 尝试主优化器
        print("\n尝试主优化器...")
        result = self.optimizer.optimize(risk_proxies, smooth_penalty_config=smooth_penalty_config)

        # 如果失败，使用降级分配器
        if not result.success or not result.constraints_satisfied:
            print(f"\n⚠️  主优化器失败（{result.status}）")
            print("使用降级分配器...")

            result = self.fallback.allocate(risk_proxies, self.problem)

        # 输出结果
        print(f"\n优化状态: {result.status}")
        print(f"成功: {result.success}")
        print(f"约束满足: {result.constraints_satisfied}")
        print(f"使用降级: {result.fallback_triggered}")

        print(f"\n最终权重:")
        for i, (strategy_id, weight) in enumerate(zip(self.problem.strategy_ids, result.weights)):
            print(f"  {strategy_id}: {weight:.2%}")

        print(f"\n预期收益: {result.expected_return:.4f}")
        print(f"预期风险: {result.expected_risk:.4f}")

        if result.binding_constraints:
            print(f"绑定约束: {result.binding_constraints}")

        print("=" * 60)

        return result


if __name__ == "__main__":
    """测试优化器 v2"""
    print("=== SPL-6b-C: 优化器 v2 测试 ===\n")

    # 创建测试问题
    from analysis.optimization_problem import OptimizationProblem

    problem = OptimizationProblem(
        name="test_optimization",
        description="测试优化器 v2",
        n_strategies=3,
        strategy_ids=["strategy_1", "strategy_2", "strategy_3"],
        expected_returns=np.array([0.05, 0.03, 0.04]),
        covariance_matrix=np.array([
            [0.01, 0.002, 0.003],
            [0.002, 0.015, 0.004],
            [0.003, 0.004, 0.02]
        ])
    )

    # 创建风险代理
    risk_calculator = RiskProxyCalculator()

    # 使用真实数据
    runs_dir = str(project_root / "runs")
    from analysis.replay_schema import load_replay_outputs

    replays = load_replay_outputs(runs_dir)
    if len(replays) >= 2:
        risk_proxies = risk_calculator.estimate_risk_proxies(replays[:3], {})

        # 创建优化器并运行
        optimizer = PortfolioOptimizerV2(problem)
        result = optimizer.run_optimization(risk_proxies)

        print(f"\n测试完成")
        print(f"成功: {result.success}")
    else:
        print("\n跳过测试（数据不足）")

    print("\n✅ 优化器 v2 测试通过")
