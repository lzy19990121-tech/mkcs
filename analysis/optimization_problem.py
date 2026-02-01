"""
SPL-6b-A: 优化问题定义 (Optimization Problem Specification)

定义优化决策变量、目标函数、约束条件。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import yaml
from pathlib import Path


class OptimizationMethod(Enum):
    """优化方法"""
    QUADRATIC_PROGRAMMING = "quadratic_programming"  # 二次规划
    PROJECTED_GRADIENT = "projected_gradient"      # 投影梯度
    INTERIOR_POINT = "interior_point"              # 内点法
    SIMPLEX = "simplex"                            # 单纯形法


class ConstraintType(Enum):
    """约束类型"""
    EQUALITY = "equality"         # 等式约束
    INEQUALITY = "inequality"     # 不等式约束
    BOUNDS = "bounds"             # 边界约束


@dataclass
class Constraint:
    """约束定义"""
    name: str
    type: ConstraintType
    lhs: Any  # 左边表达式
    rhs: Any  # 右边值
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.type.value,
            "lhs": str(self.lhs),
            "rhs": float(self.rhs) if isinstance(self.rhs, (int, float)) else self.rhs,
            "description": self.description
        }


@dataclass
class ObjectiveFunction:
    """目标函数"""
    name: str
    sense: str  # "maximize" or "minimize"
    formula: str
    description: str
    penalty_terms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "sense": self.sense,
            "formula": self.formula,
            "description": self.description,
            "penalty_terms": self.penalty_terms
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    success: bool
    weights: np.ndarray  # 最优权重
    objective_value: float
    constraints_satisfied: bool
    status: str  # "optimal", "infeasible", "error"

    # 诊断信息
    binding_constraints: List[str] = field(default_factory=list)
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    solver_iterations: int = 0
    computation_time: float = 0.0

    # 可解释性信息
    expected_return: float = 0.0
    expected_risk: float = 0.0
    marginal_contributions: np.ndarray = None

    # 元信息
    solver_used: str = ""
    fallback_triggered: bool = False
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "weights": self.weights.tolist(),
            "objective_value": float(self.objective_value),
            "constraints_satisfied": self.constraints_satisfied,
            "status": self.status,
            "binding_constraints": self.binding_constraints,
            "constraint_violations": self.constraint_violations,
            "solver_iterations": self.solver_iterations,
            "computation_time": self.computation_time,
            "expected_return": float(self.expected_return),
            "expected_risk": float(self.expected_risk),
            "marginal_contributions": self.marginal_contributions.tolist() if self.marginal_contributions is not None else None,
            "solver_used": self.solver_used,
            "fallback_triggered": self.fallback_triggered,
            "error_message": self.error_message
        }


@dataclass
class OptimizationProblem:
    """优化问题定义"""
    name: str
    description: str
    n_strategies: int  # 策略数量
    strategy_ids: List[str]  # 策略 ID 列表

    # 决策变量
    decision_variables: Dict[str, Any] = field(default_factory=dict)

    # 目标函数
    objective: ObjectiveFunction = None

    # 约束列表
    constraints: List[Constraint] = field(default_factory=list)

    # 数据参数
    expected_returns: np.ndarray = None  # 预期收益向量
    covariance_matrix: np.ndarray = None  # 协方差矩阵
    stress_periods: List[Any] = field(default_factory=list)  # 压力期

    # 优化配置
    method: OptimizationMethod = OptimizationMethod.QUADRATIC_PROGRAMMING
    fallback_method: OptimizationMethod = OptimizationMethod.PROJECTED_GRADIENT

    # 配置
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if self.expected_returns is None:
            self.expected_returns = np.zeros(self.n_strategies)
        if self.covariance_matrix is None:
            self.covariance_matrix = np.eye(self.n_strategies)

    @classmethod
    def from_config(cls, config_path: Path, strategy_ids: List[str]) -> "OptimizationProblem":
        """从配置文件创建优化问题

        Args:
            config_path: 配置文件路径
            strategy_ids: 策略 ID 列表

        Returns:
            OptimizationProblem
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        opt_config = config.get("optimization_problem", {})

        # 创建目标函数
        obj_config = opt_config.get("objective", {})
        objective = ObjectiveFunction(
            name=obj_config.get("primary", {}).get("name", "maximize_expected_return"),
            sense=obj_config.get("primary", {}).get("formula", "maximize"),
            formula=obj_config.get("primary", {}).get("formula", ""),
            description=obj_config.get("primary", {}).get("description", ""),
            penalty_terms=obj_config.get("penalty_terms", {})
        )

        # 创建约束
        constraints = []
        constraint_configs = opt_config.get("constraints", {})

        # 收益约束
        for cfg in constraint_configs.get("return_constraints", []):
            constraints.append(Constraint(
                name=cfg["name"],
                type=ConstraintType.INEQUALITY,
                lhs=f"P95_return",
                rhs=cfg["threshold"],
                description=cfg.get("description", "")
            ))

        # 风险约束
        for cfg in constraint_configs.get("risk_constraints", []):
            constraints.append(Constraint(
                name=cfg["name"],
                type=ConstraintType.INEQUALITY,
                lhs=cfg["name"],
                rhs=cfg["threshold"],
                description=cfg.get("description", "")
            ))

        # 权重约束
        for cfg in constraint_configs.get("weight_constraints", []):
            if cfg["name"] == "sum_weights":
                constraints.append(Constraint(
                    name=cfg["name"],
                    type=ConstraintType.EQUALITY,
                    lhs="sum(w)",
                    rhs=cfg["value"],
                    description=cfg.get("description", "")
                ))

        # 创建优化问题
        problem = cls(
            name=opt_config.get("name", "portfolio_optimization"),
            description=opt_config.get("description", ""),
            n_strategies=len(strategy_ids),
            strategy_ids=strategy_ids,
            objective=objective,
            constraints=constraints,
            config=config
        )

        return problem

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取权重边界

        Returns:
            (lower_bounds, upper_bounds)
        """
        # 从约束中提取边界
        lower = np.zeros(self.n_strategies)
        upper = np.ones(self.n_strategies)

        for constraint in self.constraints:
            if constraint.name == "weight_bounds":
                lower[:] = constraint.rhs[0] if isinstance(constraint.rhs, tuple) else 0.0
                upper[:] = constraint.rhs[1] if isinstance(constraint.rhs, tuple) else 1.0

        return lower, upper

    def validate(self) -> Tuple[bool, List[str]]:
        """验证优化问题设置

        Returns:
            (is_valid, errors)
        """
        errors = []

        # 检查1: 策略数量
        if self.n_strategies == 0:
            errors.append("策略数量为0")

        # 检查2: 收益向量维度
        if len(self.expected_returns) != self.n_strategies:
            errors.append(f"收益向量维度不匹配: {len(self.expected_returns)} != {self.n_strategies}")

        # 检查3: 协方差矩阵
        if self.covariance_matrix.shape != (self.n_strategies, self.n_strategies):
            errors.append(f"协方差矩阵形状不匹配: {self.covariance_matrix.shape}")

        # 检查4: 协方差矩阵正定性
        try:
            eigenvalues = np.linalg.eigvals(self.covariance_matrix)
            if np.any(eigenvalues < -1e-10):  # 允许小的数值误差
                errors.append("协方差矩阵非正定")
        except np.linalg.LinAlgError:
            errors.append("无法计算协方差矩阵特征值")

        is_valid = len(errors) == 0
        return is_valid, errors


if __name__ == "__main__":
    """测试优化问题定义"""
    print("=== SPL-6b-A: 优化问题定义测试 ===\n")

    # 创建测试问题
    strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]

    problem = OptimizationProblem(
        name="test_optimization",
        description="测试优化问题",
        n_strategies=3,
        strategy_ids=strategy_ids,
        expected_returns=np.array([0.05, 0.03, 0.04]),
        covariance_matrix=np.array([
            [0.01, 0.002, 0.003],
            [0.002, 0.015, 0.004],
            [0.003, 0.004, 0.02]
        ])
    )

    print("优化问题:")
    print(f"  名称: {problem.name}")
    print(f"  策略数量: {problem.n_strategies}")
    print(f"  策略IDs: {problem.strategy_ids}")
    print(f"  预期收益: {problem.expected_returns}")
    print(f"  协方差矩阵形状: {problem.covariance_matrix.shape}")

    # 验证
    is_valid, errors = problem.validate()
    print(f"\n验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    if not is_valid:
        for error in errors:
            print(f"  ✗ {error}")

    print("\n✅ 优化问题定义测试通过")
