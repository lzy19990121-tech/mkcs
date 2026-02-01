"""
SPL-6b-B: 构建可优化的风险代理（Risk Surrogates）

将难优化的 worst-case 指标转成可计算代理：
- CVaR 约束
- Variance / Semi-variance
- Correlation-aware penalty
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.replay_schema import load_replay_outputs, ReplayOutput
from analysis.window_scanner import WindowScanner


class RiskProxyType(Enum):
    """风险代理类型"""
    CVAR = "cvar"                          # 条件风险价值
    VARIANCE = "variance"                    # 方差
    SEMI_VARIANCE = "semi_variance"          # 半方差
    TAIL_CORRELATION = "tail_correlation"    # 尾部相关性
    CORRELATION_PENALTY = "correlation_penalty" # 相关性惩罚


@dataclass
class CVaRConstraint:
    """CVaR 约束

    Conditional Value at Risk: 在给定置信水平下的平均尾部损失
    比直接 worst-window 更易优化
    """
    confidence_level: float = 0.95  # 置信水平（95%）
    max_cvar: float = -0.10          # 最大 CVaR（-10%）
    description: str = "组合 P95 条件风险价值不超过 -10%"

    def calculate_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """计算加权组合收益的 CVaR

        Args:
            returns: 收益矩阵 (n_samples x n_strategies)
            weights: 权重向量 (n_strategies,)

        Returns:
            CVaR 值
        """
        # 计算组合收益
        portfolio_returns = returns @ weights

        # 计算 VaR（在给定置信水平下的分位数）
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)

        # 计算 CVaR（VaR 以下的平均损失）
        tail_losses = portfolio_returns[portfolio_returns <= var]
        cvar = tail_losses.mean() if len(tail_losses) > 0 else var

        return cvar

    def is_satisfied(
        self,
        returns: np.ndarray,
        weights: np.ndarray
    ) -> bool:
        """检查约束是否满足

        Args:
            returns: 收益矩阵
            weights: 权重向量

        Returns:
            是否满足约束
        """
        cvar = self.calculate_cvar(returns, weights)
        return cvar >= self.max_cvar  # CVaR 越小越好，所以 >= 阈值表示满足


@dataclass
class VarianceConstraint:
    """方差约束

    使用组合方差作为风险度量
    """
    max_variance: float = 0.08  # 最大方差 8%
    description: str = "组合方差不超过 8%"

    def calculate_variance(
        self,
        covariance_matrix: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """计算组合方差

        Args:
            covariance_matrix: 协方差矩阵 (n x n)
            weights: 权重向量 (n,)

        Returns:
            组合方差
        """
        return weights @ covariance_matrix @ weights

    def is_satisfied(
        self,
        covariance_matrix: np.ndarray,
        weights: np.ndarray
    ) -> bool:
        """检查约束是否满足"""
        variance = self.calculate_variance(covariance_matrix, weights)
        return variance <= self.max_variance


@dataclass
class SemiVarianceConstraint:
    """半方差约束

    只考虑下行风险（负收益）的方差
    更符合投资者对损失的真实感受
    """
    max_semi_variance: float = 0.04  # 最大半方差 4%
    description: str = "半方差不超过 4%"

    def calculate_semi_variance(
        self,
        returns: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """计算半方差

        Args:
            returns: 收益矩阵 (n_samples x n_strategies)
            weights: 权重向量 (n_strategies,)

        Returns:
            半方差
        """
        # 计算组合收益
        portfolio_returns = returns @ weights

        # 只考虑负收益
        negative_returns = portfolio_returns[portfolio_returns < 0]

        if len(negative_returns) == 0:
            return 0.0

        # 计算负收益的方差
        mean_negative = negative_returns.mean()
        semi_variance = ((negative_returns - mean_negative) ** 2).mean()

        return semi_variance

    def is_satisfied(
        self,
        returns: np.ndarray,
        weights: np.ndarray
    ) -> bool:
        """检查约束是否满足"""
        semi_var = self.calculate_semi_variance(returns, weights)
        return semi_var <= self.max_semi_variance


@dataclass
class TailCorrelationConstraint:
    """尾部相关性约束

    限制压力期（市场下跌时）的相关性
    防止协同爆炸
    """
    max_tail_correlation: float = 0.5  # 最大尾部相关性 0.5
    description: str = "压力期平均相关性不超过 0.5"
    stress_threshold: float = -0.02  # 压力期定义：收益 < -2%

    def calculate_tail_correlation(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        stress_mask: Optional[np.ndarray] = None
    ) -> float:
        """计算尾部相关性

        Args:
            returns: 收益矩阵 (n_samples x n_strategies)
            weights: 权重向量 (n_strategies,)
            stress_mask: 压力期掩码 (n_samples,)

        Returns:
            平均尾部相关性
        """
        # 计算组合收益
        portfolio_returns = returns @ weights

        # 确定压力期
        if stress_mask is None:
            stress_mask = portfolio_returns < self.stress_threshold

        # 如果没有压力期，返回 0
        if not stress_mask.any():
            return 0.0

        # 获取压力期的收益
        stress_returns = returns[stress_mask, :]

        # 如果只有一个策略，相关性和为 0
        if stress_returns.shape[1] < 2:
            return 0.0

        # 计算压力期的相关系数矩阵
        # 使用加权平均（按组合权重）
        weighted_returns = stress_returns * weights[np.newaxis, :]

        # 计算相关性矩阵
        corr_matrix = np.corrcoef(weighted_returns.T)

        # 提取上三角（排除对角线）
        n = corr_matrix.shape[0]
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]

        # 平均相关性
        if len(upper_tri) > 0:
            # 只考虑显著相关的（绝对值 > 0.3）
            significant_upper_tri = upper_tri[np.abs(upper_tri) > 0.3]
            if len(significant_upper_tri) > 0:
                return significant_upper_tri.mean()
            else:
                return 0.0
        return 0.0

    def is_satisfied(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        stress_mask: Optional[np.ndarray] = None
    ) -> bool:
        """检查约束是否满足"""
        tail_corr = self.calculate_tail_correlation(returns, weights, stress_mask)
        return tail_corr <= self.max_tail_correlation


@dataclass
class CorrelationPenalty:
    """相关性惩罚

    在优化目标中添加相关性惩罚项，降低组合中高相关性的策略权重
    """
    correlation_threshold: float = 0.7  # 高相关性阈值
    penalty_weight: float = 0.1     # 惩罚权重
    description: str = "对高相关性策略添加惩罚"

    def calculate_penalty(
        self,
        covariance_matrix: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """计算相关性惩罚

        Args:
            covariance_matrix: 协方差矩阵
            weights: 当前权重

        Returns:
            惩罚值
        """
        # 计算相关系数矩阵
        std_devs = np.sqrt(np.diag(covariance_matrix))
        corr_matrix = covariance_matrix / np.outer(std_devs, std_devs)

        # 处理 NaN（如果有方差为0的情况）
        corr_matrix = np.nan_to_num(corr_matrix, 0.0)

        # 计算加权相关性惩罚
        n = len(weights)
        penalty = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                # 如果相关性超过阈值且权重都较大，添加惩罚
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    weight_product = weights[i] * weights[j]
                    penalty += weight_product * abs(corr_matrix[i, j])

        return self.penalty_weight * penalty


class RiskProxyCalculator:
    """风险代理计算器"""

    def __init__(self):
        """初始化计算器"""
        self.cvar_constraint = CVaRConstraint()
        self.variance_constraint = VarianceConstraint()
        self.semi_variance_constraint = SemiVarianceConstraint()
        self.tail_correlation_constraint = TailCorrelationConstraint()
        self.correlation_penalty = CorrelationPenalty()

    def estimate_risk_proxies(
        self,
        replay_data: List[ReplayOutput],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """估计风险代理参数

        Args:
            replay_data: 回测数据列表
            config: 配置参数

        Returns:
            风险代理参数字典
        """
        print("=== 估计风险代理参数 ===\n")

        # 1. 计算收益矩阵
        returns_matrix = self._compute_returns_matrix(replay_data)
        n_samples, n_strategies = returns_matrix.shape

        print(f"收益矩阵形状: {returns_matrix.shape}")
        print(f"样本数: {n_samples}, 策略数: {n_strategies}")

        # 2. 计算协方差矩阵
        covariance_matrix = np.cov(returns_matrix.T)
        print(f"协方差矩阵形状: {covariance_matrix.shape}")

        # 3. 识别压力期
        stress_mask = self._identify_stress_periods(returns_matrix)
        n_stress = stress_mask.sum()
        print(f"压力期数量: {n_stress} ({n_stress/n_samples*100:.1f}%)")

        # 4. 计算尾部相关性
        # 使用等权重计算
        equal_weights = np.ones(n_strategies) / n_strategies
        tail_corr = self.tail_correlation_constraint.calculate_tail_correlation(
            returns_matrix, equal_weights, stress_mask
        )
        print(f"尾部相关性: {tail_corr:.4f}")

        # 5. 返回所有代理参数
        return {
            "returns_matrix": returns_matrix,
            "covariance_matrix": covariance_matrix,
            "expected_returns": returns_matrix.mean(axis=0),
            "stress_mask": stress_mask,
            "tail_correlation": tail_corr,
            "n_samples": n_samples,
            "n_strategies": n_strategies
        }

    def _compute_returns_matrix(self, replay_data: List[ReplayOutput]) -> np.ndarray:
        """计算收益矩阵

        Args:
            replay_data: 回测数据列表

        Returns:
            收益矩阵 (n_samples x n_strategies)
        """
        returns_list = []

        for replay in replay_data:
            df = replay.to_dataframe()
            if 'step_pnl' in df.columns:
                # 转换为收益率（相对于初始权益）
                initial_equity = float(replay.initial_cash)
                returns = df['step_pnl'].values / initial_equity
                returns_list.append(returns)

        # 找到最小长度并填充
        min_length = min(len(r) for r in returns_list)
        trimmed_returns = np.array([r[:min_length] for r in returns_list])

        return trimmed_returns.T  # 转置为 n_samples x n_strategies

    def _identify_stress_periods(
        self,
        returns_matrix: np.ndarray,
        threshold: float = -0.02
    ) -> np.ndarray:
        """识别压力期

        Args:
            returns_matrix: 收益矩阵
            threshold: 压力阈值

        Returns:
            压力期掩码
        """
        # 计算等权重组合收益
        equal_weights = np.ones(returns_matrix.shape[1]) / returns_matrix.shape[1]
        portfolio_returns = returns_matrix @ equal_weights

        # 压力期：组合收益低于阈值
        stress_mask = portfolio_returns < threshold

        return stress_mask

    def evaluate_all_constraints(
        self,
        weights: np.ndarray,
        risk_proxies: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, bool]:
        """评估所有约束

        Args:
            weights: 权重向量
            risk_proxies: 风险代理参数
            config: 配置

        Returns:
            约束满足情况
        """
        returns_matrix = risk_proxies["returns_matrix"]
        covariance_matrix = risk_proxies["covariance_matrix"]
        stress_mask = risk_proxies.get("stress_mask")

        results = {}

        # CVaR 约束
        results["cvar"] = self.cvar_constraint.is_satisfied(
            returns_matrix, weights
        )

        # 方差约束
        results["variance"] = self.variance_constraint.is_satisfied(
            covariance_matrix, weights
        )

        # 半方差约束
        results["semi_variance"] = self.semi_variance_constraint.is_satisfied(
            returns_matrix, weights
        )

        # 尾部相关性约束
        results["tail_correlation"] = self.tail_correlation_constraint.is_satisfied(
            returns_matrix, weights, stress_mask
        )

        return results

    def calculate_objective_with_penalty(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """计算带惩罚的目标函数值

        Args:
            expected_returns: 预期收益向量
            covariance_matrix: 协方差矩阵
            weights: 权重向量

        Returns:
            目标函数值（负数，因为是最大化问题）
        """
        # 主要目标：最大化收益
        expected_return = weights @ expected_returns

        # 风险项（方差）
        variance = weights @ covariance_matrix @ weights

        # 相关性惩罚
        penalty = self.correlation_penalty.calculate_penalty(covariance_matrix, weights)

        # 目标 = 期望收益 - 方差 - 相关性惩罚
        # 这里我们最大化收益，所以目标 = expected_return - lambda * risk - penalty
        lambda_risk = 0.5  # 风险厌恶系数
        objective = expected_return - lambda_risk * variance - penalty

        return -objective  # 负号因为 scipy.optimize 最小化


if __name__ == "__main__":
    """测试风险代理"""
    print("=== SPL-6b-B: 风险代理测试 ===\n")

    calculator = RiskProxyCalculator()

    # 创建模拟数据
    np.random.seed(42)
    n_samples = 100
    n_strategies = 3

    returns_matrix = np.random.randn(n_samples, n_strategies) * 0.01
    # 添加一些负收益（压力期）
    returns_matrix[10:20] -= 0.03

    print(f"模拟收益矩阵: {returns_matrix.shape}")
    print(f"均值: {returns_matrix.mean(axis=0)}")
    print(f"标准差: {returns_matrix.std(axis=0)}")

    # 使用真实数据测试（如果存在）
    runs_dir = str(project_root / "runs")
    if Path(runs_dir).exists():
        from analysis.replay_schema import load_replay_outputs
        replays = load_replay_outputs(runs_dir)

        if len(replays) >= 2:
            risk_proxies = calculator.estimate_risk_proxies(replays[:3], {})
            print(f"\n预期收益: {risk_proxies['expected_returns']}")
            print(f"尾部相关性: {risk_proxies['tail_correlation']:.4f}")

            # 测试权重评估
            weights = np.ones(risk_proxies['n_strategies']) / risk_proxies['n_strategies']
            results = calculator.evaluate_all_constraints(weights, risk_proxies, {})

            print(f"\n约束满足情况:")
            for name, satisfied in results.items():
                print(f"  {name}: {'✓' if satisfied else '✗'}")
        else:
            print("\n使用模拟数据测试")
            # 使用模拟数据
            covariance_matrix = np.cov(returns_matrix.T)
            weights = np.array([0.4, 0.3, 0.3])

            results = {}
            results["variance"] = calculator.variance_constraint.is_satisfied(covariance_matrix, weights)
            results["tail_correlation"] = calculator.tail_correlation_constraint.is_satisfied(returns_matrix, weights)

            print(f"\n约束满足情况:")
            for name, satisfied in results.items():
                print(f"  {name}: {'✓' if satisfied else '✗'}")

    print("\n✅ 风险代理测试通过")
