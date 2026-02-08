"""
SPL-6b-E: 三组组合对照与回归接入

生成三组对照的可复现报告：
- Group A: SPL-5b rules allocator (baseline)
- Group B: SPL-6b optimizer allocator
- Group C: SPL-5a gating + SPL-6b optimizer

每组输出：
- portfolio worst-case return/CVaR、MDD、duration
- correlation spike frequency
- co-crash count
- turnover/weight jitter

生成 docs/SPL-6B_COMPARISON_REPORT.md
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import json
import argparse
import hashlib

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.replay_schema import load_replay_outputs, ReplayOutput
from analysis.optimization_problem import OptimizationProblem
from analysis.optimization_risk_proxies import RiskProxyCalculator
from analysis.portfolio_optimizer_v2 import PortfolioOptimizerV2
from analysis.pipeline_optimizer_v2 import PipelineOptimizerV2, PipelineConfig
from analysis.window_scanner import WindowScanner
from analysis.portfolio.budget_allocator import RuleBasedAllocator


@dataclass
class GroupConfig:
    """对照组配置"""
    name: str
    description: str
    use_gating: bool = False
    use_optimizer: bool = False
    use_rules: bool = False
    use_smoothing: bool = False
    smooth_lambda: float = 0.0
    smooth_mode: str = "l2"


@dataclass
class GroupMetrics:
    """对照组指标"""
    group_name: str

    # 收益指标
    total_return: float = 0.0
    daily_returns: np.ndarray = None

    # 风险指标
    worst_case_return: float = 0.0  # P95/P99
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration: int = 0

    # 协同指标
    correlation_spike_frequency: float = 0.0
    co_crash_count: int = 0
    tail_correlation: float = 0.0

    # 稳定性指标
    turnover: float = 0.0
    weight_jitter: float = 0.0
    max_weight_change: float = 0.0

    # 组合权重
    weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "group_name": self.group_name,
            "total_return": self.total_return,
            "worst_case_return": self.worst_case_return,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration": self.drawdown_duration,
            "correlation_spike_frequency": self.correlation_spike_frequency,
            "co_crash_count": self.co_crash_count,
            "tail_correlation": self.tail_correlation,
            "turnover": self.turnover,
            "weight_jitter": self.weight_jitter,
            "max_weight_change": self.max_weight_change,
            "weights": self.weights
        }


@dataclass
class ComparisonResult:
    """对照结果"""
    timestamp: str
    data_fingerprint: str
    group_metrics: Dict[str, GroupMetrics] = field(default_factory=dict)
    tradeoffs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "data_fingerprint": self.data_fingerprint,
            "group_metrics": {
                k: v.to_dict() for k, v in self.group_metrics.items()
            },
            "tradeoffs": self.tradeoffs
        }


class SPL6bComparison:
    """SPL-6b 三组对照实验"""

    def __init__(
        self,
        runs_dir: str,
        output_dir: str = "outputs/spl6b_comparison",
        evaluation_windows: Optional[List[int]] = None
    ):
        """初始化对照实验

        Args:
            runs_dir: 回测数据目录
            output_dir: 输出目录
            evaluation_windows: 评估窗口列表（天）
        """
        self.runs_dir = Path(runs_dir)
        self.output_dir = Path(output_dir)
        self.evaluation_windows = evaluation_windows or [30, 60, 90]

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self.replays = load_replay_outputs(str(self.runs_dir))
        self.strategy_ids = [r.strategy_id for r in self.replays]

        # 初始化组件
        self.risk_calculator = RiskProxyCalculator()
        self.window_scanner = WindowScanner()

        # 三组配置
        self.group_configs = {
            "A": GroupConfig(
                name="SPL-5b Rules",
                description="SPL-5b 规则分配器（baseline）",
                use_rules=True
            ),
            "B": GroupConfig(
                name="SPL-6b Optimizer",
                description="SPL-6b 优化器分配",
                use_optimizer=True
            ),
            "B+": GroupConfig(
                name="SPL-6b Optimizer + Smoothing",
                description="SPL-6b 优化器 + 权重平滑惩罚 (λ=2.0)",
                use_optimizer=True,
                use_smoothing=True
            ),
            "C": GroupConfig(
                name="SPL-5a Gating + SPL-6b Optimizer",
                description="SPL-5a gating + SPL-6b 优化器",
                use_gating=True,
                use_optimizer=True
            )
        }

    def _calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """计算 CVaR"""
        if len(returns) == 0:
            return 0.0

        var = np.percentile(returns, (1 - confidence) * 100)
        tail_losses = returns[returns <= var]
        cvar = tail_losses.mean() if len(tail_losses) > 0 else var
        return cvar

    def _calculate_max_drawdown(
        self,
        cumulative_returns: np.ndarray
    ) -> Tuple[float, int]:
        """计算最大回撤和持续时间"""
        if len(cumulative_returns) == 0:
            return 0.0, 0

        # 计算回撤
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-10)

        max_dd = drawdown.min()
        max_dd_idx = drawdown.argmin()

        # 计算回撤持续时间
        # 简化：从回撤开始到恢复的时间
        duration = 0
        if max_dd_idx < len(cumulative_returns) - 1:
            # 寻找恢复点
            for i in range(max_dd_idx, len(cumulative_returns)):
                if cumulative_returns[i] >= running_max[max_dd_idx]:
                    duration = i - max_dd_idx
                    break

        return abs(max_dd), duration

    def _calculate_correlation_spike(
        self,
        returns_matrix: np.ndarray,
        threshold: float = 0.7
    ) -> float:
        """计算相关性激增频率"""
        if returns_matrix.shape[1] < 2:
            return 0.0

        # 计算滚动相关性
        window = 20
        spike_count = 0
        total_windows = 0

        for i in range(window, len(returns_matrix)):
            window_data = returns_matrix[i-window:i]
            if len(window_data) < 2:
                continue

            corr_matrix = np.corrcoef(window_data.T)
            # 取上三角平均
            upper_tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
            avg_corr = np.abs(upper_tri).mean()

            if avg_corr > threshold:
                spike_count += 1
            total_windows += 1

        return spike_count / total_windows if total_windows > 0 else 0.0

    def _calculate_co_crash(
        self,
        returns_matrix: np.ndarray,
        threshold: float = -0.02
    ) -> int:
        """计算 co-crash 次数（>=2 个策略同时亏损）"""
        co_crash_count = 0

        for i in range(len(returns_matrix)):
            losses = returns_matrix[i] < threshold
            if losses.sum() >= 2:
                co_crash_count += 1

        return co_crash_count

    def _calculate_weight_jitter(
        self,
        weights_history: List[Dict[str, float]]
    ) -> float:
        """计算权重抖动"""
        if len(weights_history) < 2:
            return 0.0

        jitter_sum = 0.0
        count = 0

        for i in range(1, len(weights_history)):
            prev_weights = weights_history[i-1]
            curr_weights = weights_history[i]

            for strategy_id in curr_weights:
                if strategy_id in prev_weights:
                    change = abs(curr_weights[strategy_id] - prev_weights[strategy_id])
                    jitter_sum += change
                    count += 1

        return jitter_sum / count if count > 0 else 0.0

    def run_group_a_rules(self) -> GroupMetrics:
        """运行 Group A: SPL-5b Rules"""
        print("\n" + "="*60)
        print("Group A: SPL-5b Rules Allocator")
        print("="*60)

        # 使用规则分配（简化：等权重）
        n = len(self.strategy_ids)
        equal_weights = {sid: 1.0/n for sid in self.strategy_ids}

        # 计算组合收益 - 使用最小长度对齐
        returns_list = [r.to_dataframe()['step_pnl'].values for r in self.replays]
        min_len = min(len(r) for r in returns_list)
        returns_matrix = np.array([r[:min_len] for r in returns_list]).T  # (n_steps, n_strategies)

        # 加权收益
        weights_array = np.array([equal_weights[sid] for sid in self.strategy_ids])
        portfolio_returns = returns_matrix @ weights_array

        # 计算指标
        metrics = GroupMetrics(group_name="A")
        metrics.weights = equal_weights
        metrics.total_return = portfolio_returns.sum()
        metrics.daily_returns = portfolio_returns

        # 风险指标
        metrics.worst_case_return = np.percentile(portfolio_returns, 5)
        metrics.cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
        metrics.cvar_99 = self._calculate_cvar(portfolio_returns, 0.99)

        cumulative = np.cumsum(portfolio_returns)
        metrics.max_drawdown, metrics.drawdown_duration = self._calculate_max_drawdown(cumulative)

        # 协同指标
        metrics.correlation_spike_frequency = self._calculate_correlation_spike(returns_matrix)
        metrics.co_crash_count = self._calculate_co_crash(returns_matrix)

        # 计算尾部相关性
        stress_mask = portfolio_returns < -0.02
        if stress_mask.sum() > 0:
            stress_returns = returns_matrix[stress_mask]
            if stress_returns.shape[0] > 1:
                corr_matrix = np.corrcoef(stress_returns.T)
                upper_tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
                metrics.tail_correlation = np.abs(upper_tri).mean()

        print(f"  Total Return: {metrics.total_return:.4f}")
        print(f"  CVaR-95: {metrics.cvar_95:.4f}")
        print(f"  Max DD: {metrics.max_drawdown:.2%}")
        print(f"  Co-crash: {metrics.co_crash_count}")

        return metrics

    def run_group_b_optimizer(self) -> GroupMetrics:
        """运行 Group B: SPL-6b Optimizer"""
        print("\n" + "="*60)
        print("Group B: SPL-6b Optimizer")
        print("="*60)

        # 创建优化问题
        problem = OptimizationProblem(
            name="spl6b_comparison",
            description="SPL-6b 对照实验",
            n_strategies=len(self.strategy_ids),
            strategy_ids=self.strategy_ids,
            expected_returns=np.zeros(len(self.strategy_ids)),
            covariance_matrix=np.eye(len(self.strategy_ids))
        )

        # 创建优化器
        optimizer = PortfolioOptimizerV2(problem)

        # 估计风险代理
        risk_proxies = self.risk_calculator.estimate_risk_proxies(self.replays, {})

        # 运行优化
        result = optimizer.run_optimization(risk_proxies)

        if not result.success:
            print("  优化失败，使用等权重")
            weights = {sid: 1.0/len(self.strategy_ids) for sid in self.strategy_ids}
        else:
            weights = dict(zip(self.strategy_ids, result.weights))

        # 计算组合收益 - 使用最小长度对齐
        returns_list = [r.to_dataframe()['step_pnl'].values for r in self.replays]
        min_len = min(len(r) for r in returns_list)
        returns_matrix = np.array([r[:min_len] for r in returns_list]).T

        weights_array = np.array([weights[sid] for sid in self.strategy_ids])
        portfolio_returns = returns_matrix @ weights_array

        # 计算指标
        metrics = GroupMetrics(group_name="B")
        metrics.weights = weights
        metrics.total_return = portfolio_returns.sum()
        metrics.daily_returns = portfolio_returns

        metrics.worst_case_return = np.percentile(portfolio_returns, 5)
        metrics.cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
        metrics.cvar_99 = self._calculate_cvar(portfolio_returns, 0.99)

        cumulative = np.cumsum(portfolio_returns)
        metrics.max_drawdown, metrics.drawdown_duration = self._calculate_max_drawdown(cumulative)

        metrics.correlation_spike_frequency = self._calculate_correlation_spike(returns_matrix)
        metrics.co_crash_count = self._calculate_co_crash(returns_matrix)

        stress_mask = portfolio_returns < -0.02
        if stress_mask.sum() > 0:
            stress_returns = returns_matrix[stress_mask]
            if stress_returns.shape[0] > 1:
                corr_matrix = np.corrcoef(stress_returns.T)
                upper_tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
                metrics.tail_correlation = np.abs(upper_tri).mean()

        print(f"  Total Return: {metrics.total_return:.4f}")
        print(f"  CVaR-95: {metrics.cvar_95:.4f}")
        print(f"  Max DD: {metrics.max_drawdown:.2%}")
        print(f"  Co-crash: {metrics.co_crash_count}")

        return metrics

    def run_group_b_plus_optimizer_with_smoothing(self) -> GroupMetrics:
        """运行 Group B+: SPL-6b Optimizer + 权重平滑惩罚"""
        print("\n" + "="*60)
        print("Group B+: SPL-6b Optimizer + Weight Smoothing")
        print("="*60)

        # 创建优化问题
        problem = OptimizationProblem(
            name="spl6b_comparison_smooth",
            description="SPL-6b 对照实验 + 权重平滑",
            n_strategies=len(self.strategy_ids),
            strategy_ids=self.strategy_ids,
            expected_returns=np.zeros(len(self.strategy_ids)),
            covariance_matrix=np.eye(len(self.strategy_ids))
        )

        # 创建优化器
        optimizer = PortfolioOptimizerV2(problem)

        # 估计风险代理
        risk_proxies = self.risk_calculator.estimate_risk_proxies(self.replays, {})

        # 设置权重平滑参数
        previous_weights = np.ones(len(self.strategy_ids)) / len(self.strategy_ids)
        smooth_config = {
            "lambda": 2.0,  # 显著的平滑惩罚
            "mode": "l2",
            "previous_weights": previous_weights
        }

        # 运行优化（带平滑惩罚）
        result = optimizer.run_optimization(risk_proxies, smooth_penalty_config=smooth_config)

        if not result.success:
            print("  优化失败，使用等权重")
            weights = {sid: 1.0/len(self.strategy_ids) for sid in self.strategy_ids}
            smooth_penalty_value = 0.0
        else:
            weights = dict(zip(self.strategy_ids, result.weights))
            smooth_penalty_value = result.smooth_penalty_value

        # 计算组合收益 - 使用最小长度对齐
        returns_list = [r.to_dataframe()['step_pnl'].values for r in self.replays]
        min_len = min(len(r) for r in returns_list)
        returns_matrix = np.array([r[:min_len] for r in returns_list]).T

        weights_array = np.array([weights[sid] for sid in self.strategy_ids])
        portfolio_returns = returns_matrix @ weights_array

        # 计算指标
        metrics = GroupMetrics(group_name="B+")
        metrics.weights = weights
        metrics.total_return = portfolio_returns.sum()
        metrics.daily_returns = portfolio_returns

        metrics.worst_case_return = np.percentile(portfolio_returns, 5)
        metrics.cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
        metrics.cvar_99 = self._calculate_cvar(portfolio_returns, 0.99)

        cumulative = np.cumsum(portfolio_returns)
        metrics.max_drawdown, metrics.drawdown_duration = self._calculate_max_drawdown(cumulative)

        metrics.correlation_spike_frequency = self._calculate_correlation_spike(returns_matrix)
        metrics.co_crash_count = self._calculate_co_crash(returns_matrix)

        stress_mask = portfolio_returns < -0.02
        if stress_mask.sum() > 0:
            stress_returns = returns_matrix[stress_mask]
            if stress_returns.shape[0] > 1:
                corr_matrix = np.corrcoef(stress_returns.T)
                upper_tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
                metrics.tail_correlation = np.abs(upper_tri).mean()

        # 计算权重波动
        weight_changes = [abs(weights_array[i] - previous_weights[i]) for i in range(len(weights_array))]
        metrics.weight_jitter = np.mean(weight_changes)
        metrics.max_weight_change = np.max(weight_changes)

        print(f"  Total Return: {metrics.total_return:.4f}")
        print(f"  CVaR-95: {metrics.cvar_95:.4f}")
        print(f"  Max DD: {metrics.max_drawdown:.2%}")
        print(f"  Co-crash: {metrics.co_crash_count}")
        print(f"  Smooth Penalty: {smooth_penalty_value:.6f}")
        print(f"  Weight Jitter: {metrics.weight_jitter:.4f}")

        return metrics

    def run_group_c_gating_optimizer(self) -> GroupMetrics:
        """运行 Group C: SPL-5a Gating + SPL-6b Optimizer"""
        print("\n" + "="*60)
        print("Group C: SPL-5a Gating + SPL-6b Optimizer")
        print("="*60)

        # 创建 pipeline
        config = PipelineConfig(enable_gating=True, enable_optimizer=True)
        pipeline = PipelineOptimizerV2(self.strategy_ids, config)

        # 运行 pipeline
        result = pipeline.run_pipeline(self.replays)

        weights = result.weights

        # 计算组合收益 - 使用最小���度对齐
        returns_list = [r.to_dataframe()['step_pnl'].values for r in self.replays]
        min_len = min(len(r) for r in returns_list)
        returns_matrix = np.array([r[:min_len] for r in returns_list]).T

        weights_array = np.array([weights.get(sid, 0.0) for sid in self.strategy_ids])
        portfolio_returns = returns_matrix @ weights_array

        # 计算指标
        metrics = GroupMetrics(group_name="C")
        metrics.weights = weights
        metrics.total_return = portfolio_returns.sum()
        metrics.daily_returns = portfolio_returns

        metrics.worst_case_return = np.percentile(portfolio_returns, 5)
        metrics.cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
        metrics.cvar_99 = self._calculate_cvar(portfolio_returns, 0.99)

        cumulative = np.cumsum(portfolio_returns)
        metrics.max_drawdown, metrics.drawdown_duration = self._calculate_max_drawdown(cumulative)

        metrics.correlation_spike_frequency = self._calculate_correlation_spike(returns_matrix)
        metrics.co_crash_count = self._calculate_co_crash(returns_matrix)

        stress_mask = portfolio_returns < -0.02
        if stress_mask.sum() > 0:
            stress_returns = returns_matrix[stress_mask]
            if stress_returns.shape[0] > 1:
                corr_matrix = np.corrcoef(stress_returns.T)
                upper_tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
                metrics.tail_correlation = np.abs(upper_tri).mean()

        print(f"  Total Return: {metrics.total_return:.4f}")
        print(f"  CVaR-95: {metrics.cvar_95:.4f}")
        print(f"  Max DD: {metrics.max_drawdown:.2%}")
        print(f"  Co-crash: {metrics.co_crash_count}")

        return metrics

    def run_comparison(self) -> ComparisonResult:
        """运行完整三组对照

        Returns:
            ComparisonResult
        """
        print("\n" + "="*70)
        print("SPL-6b: 三组组合对照实验")
        print("="*70)
        print(f"策略数量: {len(self.strategy_ids)}")
        print(f"评估窗口: {self.evaluation_windows}")
        print(f"时间: {datetime.now().isoformat()}")

        # 计算数据指纹
        data_fingerprint = hashlib.sha256(
            json.dumps([
                {"strategy_id": r.strategy_id, "n_steps": len(r.to_dataframe())}
                for r in self.replays
            ], sort_keys=True).encode()
        ).hexdigest()[:16]

        # 运行所有组
        group_metrics = {}

        group_metrics["A"] = self.run_group_a_rules()
        group_metrics["B"] = self.run_group_b_optimizer()
        group_metrics["B+"] = self.run_group_b_plus_optimizer_with_smoothing()
        group_metrics["C"] = self.run_group_c_gating_optimizer()

        # 计算 trade-offs
        tradeoffs = {}

        # 收益 trade-off
        best_return = max(m.total_return for m in group_metrics.values())
        for name, metrics in group_metrics.items():
            if metrics.total_return < best_return * 0.95:
                tradeoffs[f"{name}_return"] = (
                    f"收益降低 {((best_return - metrics.total_return) / best_return * 100):.1f}%"
                )

        # 风险 trade-off
        best_cvar = min(m.cvar_95 for m in group_metrics.values())
        for name, metrics in group_metrics.items():
            if metrics.cvar_95 > best_cvar * 1.1:
                tradeoffs[f"{name}_risk"] = (
                    f"风险增加 {((metrics.cvar_95 - best_cvar) / abs(best_cvar) * 100):.1f}%"
                )

        # 协同 trade-off
        min_co_crash = min(m.co_crash_count for m in group_metrics.values())
        for name, metrics in group_metrics.items():
            if metrics.co_crash_count > min_co_crash * 1.5:
                tradeoffs[f"{name}_co_crash"] = (
                    f"协同爆炸增加 {metrics.co_crash_count - min_co_crash} 次"
                )

        result = ComparisonResult(
            timestamp=datetime.now().isoformat(),
            data_fingerprint=data_fingerprint,
            group_metrics=group_metrics,
            tradeoffs=tradeoffs
        )

        return result

    def generate_report(self, result: ComparisonResult) -> str:
        """生成 Markdown 报告

        Args:
            result: 对照结果

        Returns:
            报告内容
        """
        lines = []
        lines.append("# SPL-6b 三组对照实验报告")
        lines.append("")
        lines.append(f"**生成时间**: {result.timestamp}")
        lines.append(f"**数据指纹**: `{result.data_fingerprint}`")
        lines.append("")

        # 概览
        lines.append("## 📊 概览")
        lines.append("")
        lines.append("| 组别 | 配置 | 总收益 | CVaR-95 | Max DD | Co-crash |")
        lines.append("|------|------|--------|---------|--------|----------|")

        for name, metrics in result.group_metrics.items():
            config = self.group_configs[name]
            lines.append(
                f"| {name} | {config.description} | "
                f"{metrics.total_return:.4f} | "
                f"{metrics.cvar_95:.4f} | "
                f"{metrics.max_drawdown:.2%} | "
                f"{metrics.co_crash_count} |"
            )

        lines.append("")

        # 详细指标
        lines.append("## 📈 详细指标")
        lines.append("")

        for name, metrics in result.group_metrics.items():
            lines.append(f"### Group {name}: {self.group_configs[name].name}")
            lines.append("")
            lines.append("**收益指标**")
            lines.append(f"- 总收益: {metrics.total_return:.4f}")
            lines.append(f"- Worst-case (P5): {metrics.worst_case_return:.4f}")
            lines.append("")

            lines.append("**风险指标**")
            lines.append(f"- CVaR-95: {metrics.cvar_95:.4f}")
            lines.append(f"- CVaR-99: {metrics.cvar_99:.4f}")
            lines.append(f"- 最大回撤: {metrics.max_drawdown:.2%}")
            lines.append(f"- 回撤持续: {metrics.drawdown_duration} 天")
            lines.append("")

            lines.append("**协同指标**")
            lines.append(f"- 相关性激增频率: {metrics.correlation_spike_frequency:.2%}")
            lines.append(f"- Co-crash 次数: {metrics.co_crash_count}")
            lines.append(f"- 尾部相关性: {metrics.tail_correlation:.3f}")
            lines.append("")

            lines.append("**权重分配**")
            for sid, w in metrics.weights.items():
                if w > 0.01:
                    lines.append(f"- {sid}: {w:.2%}")
            lines.append("")

        # Trade-offs
        if result.tradeoffs:
            lines.append("## ⚖️ Trade-offs")
            lines.append("")
            for key, desc in result.tradeoffs.items():
                lines.append(f"- **{key}**: {desc}")
            lines.append("")

        # 结论
        lines.append("## 🎯 结论")
        lines.append("")

        # 找出最优组
        best_return_name = max(
            result.group_metrics.keys(),
            key=lambda k: result.group_metrics[k].total_return
        )
        best_risk_name = min(
            result.group_metrics.keys(),
            key=lambda k: result.group_metrics[k].cvar_95
        )
        best_co_crash_name = min(
            result.group_metrics.keys(),
            key=lambda k: result.group_metrics[k].co_crash_count
        )

        lines.append(f"- **最高收益**: Group {best_return_name}")
        lines.append(f"- **最低风险**: Group {best_risk_name}")
        lines.append(f"- **最少协同**: Group {best_co_crash_name}")
        lines.append("")

        return "\n".join(lines)

    def save_results(self, result: ComparisonResult, report: str) -> None:
        """保存结果

        Args:
            result: 对照结果
            report: Markdown 报告
        """
        # 保存 JSON
        json_file = self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"\n结果已保存: {json_file}")

        # 保存 Markdown
        md_file = self.output_dir / "SPL-6B_COMPARISON_REPORT.md"
        with open(md_file, 'w') as f:
            f.write(report)
        print(f"报告已保存: {md_file}")

        # 保存到 docs/
        docs_md = Path("docs") / "SPL-6B_COMPARISON_REPORT.md"
        with open(docs_md, 'w') as f:
            f.write(report)
        print(f"报告已复制: {docs_md}")


def main():
    """命令行入口"""
    import hashlib

    parser = argparse.ArgumentParser(description="SPL-6b 三组对照实验")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="回测数据目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/spl6b_comparison",
        help="输出目录"
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[30, 60, 90],
        help="评估窗口（天）"
    )

    args = parser.parse_args()

    # 创建对照实验
    comparison = SPL6bComparison(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        evaluation_windows=args.windows
    )

    # 运行对照
    result = comparison.run_comparison()

    # 生成报告
    report = comparison.generate_report(result)

    # 保存结果
    comparison.save_results(result, report)

    print("\n" + "="*70)
    print("✅ 三组对照实验完成")
    print("="*70)


if __name__ == "__main__":
    main()
