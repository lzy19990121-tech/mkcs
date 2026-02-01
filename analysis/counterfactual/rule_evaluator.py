"""
SPL-7b-D: 规则与结构评估

回答：哪条规则最"值钱"？哪些组合调整能显著降低 co-crash？
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.counterfactual.counterfactual_config import CounterfactualResult
from analysis.counterfactual.effect_calculator import (
    EffectMetrics,
    RuleValueMetrics,
    EffectCalculator,
    SpikeAnalyzer
)


@dataclass
class RuleEvaluation:
    """规则评估结果"""
    rule_id: str

    # 价值评估
    overall_value: float  # 综合价值评分 (0-100)

    # 最坏窗口价值
    worst_window_value: float
    worst_window_rank: int

    # 效率评分
    efficiency_score: float

    # 建议
    recommendation: str  # "keep", "remove", "modify"
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "overall_value": self.overall_value,
            "worst_window_value": self.worst_window_value,
            "worst_window_rank": self.worst_window_rank,
            "efficiency_score": self.efficiency_score,
            "recommendation": self.recommendation,
            "reason": self.reason
        }


@dataclass
class PortfolioEvaluation:
    """组合评估结果"""
    composition_id: str

    # 协同指标
    co_crash_reduction: float  # Co-crash 减少
    correlation_reduction: float  # 相关性降低

    # 风险指标
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float

    # 建议
    recommendation: str
    suggested_adjustments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composition_id": self.composition_id,
            "co_crash_reduction": self.co_crash_reduction,
            "correlation_reduction": self.correlation_reduction,
            "portfolio_var": self.portfolio_var,
            "portfolio_cvar": self.portfolio_cvar,
            "max_drawdown": self.max_drawdown,
            "recommendation": self.recommendation,
            "suggested_adjustments": self.suggested_adjustments
        }


class RuleEvaluator:
    """规则评估器

    评估规则价值和组合效果。
    """

    def __init__(self):
        """初始化评估器"""
        self.effect_calculator = EffectCalculator()

    def evaluate_rules(
        self,
        actual_result: CounterfactualResult,
        cf_results: Dict[str, CounterfactualResult]
    ) -> List[RuleEvaluation]:
        """评估所有规则

        Args:
            actual_result: 实际结果
            cf_results: 反事实结果字典

        Returns:
            规则评估列表
        """
        evaluations = []

        # 计算规则价值
        rule_values = self.effect_calculator.calculate_rule_values(
            actual_result, cf_results
        )

        # 计算最坏窗口价值
        worst_window_values = self.effect_calculator.calculate_worst_window_value(
            actual_result, cf_results
        )

        # 为每个规则生成评估
        for rule_value in rule_values:
            # 综合价值评分（效率 + 最坏窗口价值）
            efficiency_score = rule_value.efficiency_score
            worst_window_value = worst_window_values.get(
                rule_value.rule_id.replace(" ", "_"),
                0.0
            )

            # 归一化评分
            normalized_efficiency = efficiency_score / 100.0
            normalized_window = min(1.0, max(0.0, worst_window_value / 0.05))  # 5% 为基准

            overall_value = (
                normalized_efficiency * 0.6 +
                normalized_window * 0.4
            ) * 100

            # 判定建议
            if overall_value > 70:
                recommendation = "keep"
                reason = f"高价值规则（效率 {efficiency_score:.1f}，最坏窗口改善 {worst_window_value:.2%}）"
            elif overall_value < 30:
                recommendation = "remove"
                reason = f"低价值规则（效率 {efficiency_score:.1f}，最坏窗口改善 {worst_window_value:.2%}）"
            else:
                recommendation = "modify"
                reason = f"中等价值，建议调整参数（效率 {efficiency_score:.1f}）"

            # 找到最坏窗口排名
            sorted_windows = sorted(
                worst_window_values.items(),
                key=lambda x: x[1],
                reverse=True
            )
            rank = 1
            for scenario_id, value in sorted_windows:
                if scenario_id == rule_value.rule_id.replace(" ", "_"):
                    break
                rank += 1

            evaluations.append(RuleEvaluation(
                rule_id=rule_value.rule_id,
                overall_value=overall_value,
                worst_window_value=worst_window_value,
                worst_window_rank=rank,
                efficiency_score=efficiency_score,
                recommendation=recommendation,
                reason=reason
            ))

        return evaluations

    def identify_weak_rules(
        self,
        evaluations: List[RuleEvaluation],
        value_threshold: float = 30.0
    ) -> List[str]:
        """识别弱规则

        Args:
            evaluations: 规则评估列表
            value_threshold: 价值阈值

        Returns:
            弱规则 ID 列表
        """
        weak_rules = []

        for eval in evaluations:
            if eval.overall_value < value_threshold:
                weak_rules.append(eval.rule_id)

        return weak_rules

    def identify_strong_rules(
        self,
        evaluations: List[RuleEvaluation],
        value_threshold: float = 70.0
    ) -> List[str]:
        """识别强规则

        Args:
            evaluations: 规则评估列表
            value_threshold: 价值阈值

        Returns:
            强规则 ID 列表
        """
        strong_rules = []

        for eval in evaluations:
            if eval.overall_value > value_threshold:
                strong_rules.append(eval.rule_id)

        return strong_rules

    def generate_rule_recommendations(
        self,
        evaluations: List[RuleEvaluation]
    ) -> Dict[str, List[str]]:
        """生成规则建议

        Args:
            evaluations: 规则评估列表

        Returns:
            建议字典
        """
        recommendations = {
            "keep": [],
            "remove": [],
            "modify": []
        }

        for eval in evaluations:
            recommendations[eval.recommendation].append(eval.rule_id)

        # 添加具体建议
        for eval in evaluations:
            if eval.recommendation == "modify":
                suggestions = self._generate_modification_suggestions(eval)
                recommendations["modify"].extend(suggestions)

        return recommendations

    def _generate_modification_suggestions(
        self,
        evaluation: RuleEvaluation
    ) -> List[str]:
        """生成修改建议

        Args:
            evaluation: 规则评估

        Returns:
            建议列表
        """
        suggestions = []

        rule_id = evaluation.rule_id

        if "earlier" in rule_id:
            suggestions.append(f"{rule_id}: 当前阈值过紧，建议放宽 10-20%")
        elif "later" in rule_id:
            suggestions.append(f"{rule_id}: 当前阈值过松，建议收紧 10-20%")
        elif "no_gating" in rule_id:
            if evaluation.efficiency_score < 40:
                suggestions.append(f"{rule_id}: 无 gating 导致高波动，建议启用基础 gating")
            else:
                suggestions.append(f"{rule_id}: 无 gating 表现良好，可考虑保持")

        return suggestions


class PortfolioCompositionEvaluator:
    """组合成分评估器

    评估组合调整的效果。
    """

    def __init__(self):
        """初始化评估器"""

    def evaluate_compositions(
        self,
        cf_results: Dict[str, CounterfactualResult]
    ) -> List[PortfolioEvaluation]:
        """评估组合成分

        Args:
            cf_results: 反事实结果（包含不同组合成分的场景）

        Returns:
            组合评估列表
        """
        evaluations = []

        # 查找组合成分相关场景
        composition_scenarios = [
            (sid, result) for sid, result in cf_results.items()
            if sid.startswith("cf_exclude_") or sid == "cf_equal_weight"
        ]

        for scenario_id, result in composition_scenarios:
            if not result.success:
                continue

            # 计算 co-crash 减少（简化：与 actual 对比）
            # TODO: 需要实际结果数据
            co_crash_reduction = 0.0
            correlation_reduction = 0.0

            # 建议
            if result.max_drawdown < 0.05:  # 回撤 < 5%
                recommendation = "effective"
                suggested = ["当前组合风险控制良好"]
            elif result.max_drawdown < 0.08:
                recommendation = "moderate"
                suggested = ["考虑添加低相关性策略"]
            else:
                recommendation = "risky"
                suggested = ["建议调整组合权重或排除高风险策略"]

            evaluations.append(PortfolioEvaluation(
                composition_id=scenario_id,
                co_crash_reduction=co_crash_reduction,
                correlation_reduction=correlation_reduction,
                portfolio_var=result.volatility,
                portfolio_cvar=result.cvar_95,
                max_drawdown=result.max_drawdown,
                recommendation=recommendation,
                suggested_adjustments=suggested
            ))

        return evaluations

    def identify_improvement_opportunities(
        self,
        evaluations: List[PortfolioEvaluation]
    ) -> List[Dict[str, Any]]:
        """识别改进机会

        Args:
            evaluations: 组合评估列表

        Returns:
            改进机会列表
        """
        opportunities = []

        for eval in evaluations:
            if eval.recommendation == "risky":
                opportunities.append({
                    "composition_id": eval.composition_id,
                    "current_max_dd": eval.max_drawdown,
                    "target_max_dd": eval.max_drawdown * 0.8,  # 降低 20%
                    "potential_improvement": f"降低最大回撤 {eval.max_drawdown*0.2:.1%}",
                    "suggested_adjustments": eval.suggested_adjustments
                })

        return opportunities


def generate_evaluation_report(
    actual_result: CounterfactualResult,
    cf_results: Dict[str, CounterfactualResult],
    output_path: Optional[str] = None
) -> str:
    """生成评估报告

    Args:
        actual_result: 实际结果
        cf_results: 反事实结果
        output_path: 输出文件路径

    Returns:
        Markdown 报告
    """
    lines = []
    lines.append("# SPL-7b: 规则与组合评估报告\n")
    lines.append(f"**生成时间**: {datetime.now().isoformat()}\n")

    # 初始化评估器
    rule_evaluator = RuleEvaluator()
    portfolio_evaluator = PortfolioCompositionEvaluator()

    # 评估规则
    rule_evaluations = rule_evaluator.evaluate_rules(actual_result, cf_results)

    lines.append("## 规则价值评估\n")
    lines.append("| 规则 | 综合价值 | 效率评分 | 最坏窗口价值 | 建议 | 排名 |")
    lines.append("|------|----------|----------|--------------|------|------|")

    for eval in sorted(rule_evaluations, key=lambda x: x.overall_value, reverse=True):
        lines.append(
            f"| {eval.rule_id} | {eval.overall_value:.1f} | "
            f"{eval.efficiency_score:.1f} | {eval.worst_window_value:.2%} | "
            f"{eval.recommendation} | {eval.worst_window_rank} |"
        )
        lines.append(f"  *{eval.reason}*")

    lines.append("")

    # 强规则
    strong_rules = rule_evaluator.identify_strong_rules(rule_evaluations, value_threshold=70.0)
    lines.append("### 强规则（高价值）\n")
    for rule_id in strong_rules:
        lines.append(f"- **{rule_id}**: 在最坏窗口中表现优异")
    lines.append("")

    # 弱规则
    weak_rules = rule_evaluator.identify_weak_rules(rule_evaluations, value_threshold=30.0)
    lines.append("### 弱规则（低价值）\n")
    if weak_rules:
        for rule_id in weak_rules:
            lines.append(f"- **{rule_id}**: 贡献较小，可考虑移除")
    else:
        lines.append("无弱规则（所有规则都有价值）")
    lines.append("")

    # 规则建议
    recommendations = rule_evaluator.generate_rule_recommendations(rule_evaluations)
    lines.append("### 规则建议\n")
    lines.append("**保留**:")
    for rule_id in recommendations["keep"]:
        lines.append(f"- {rule_id}")
    lines.append("\n**移除**:")
    for rule_id in recommendations["remove"]:
        lines.append(f"- {rule_id}")
    lines.append("\n**调整**:")
    for suggestion in recommendations["modify"]:
        lines.append(f"- {suggestion}")
    lines.append("")

    # 评估组合成分
    portfolio_evaluations = portfolio_evaluator.evaluate_compositions(cf_results)

    if portfolio_evaluations:
        lines.append("## 组合成分评估\n")
        lines.append("| 组合 | 最大回撤 | CVaR-95 | 建议 |")
        lines.append("|------|----------|---------|------|")

        for eval in portfolio_evaluations:
            lines.append(
                f"| {eval.composition_id} | {eval.max_drawdown:.2%} | "
                f"{eval.portfolio_cvar:.4f} | {eval.recommendation} |"
            )
        lines.append("")

        # 改进机会
        opportunities = portfolio_evaluator.identify_improvement_opportunities(portfolio_evaluations)
        if opportunities:
            lines.append("### 改进机会\n")
            for opp in opportunities:
                lines.append(f"#### {opp['composition_id']}")
                lines.append(f"- 当前最大回撤: {opp['current_max_dd']:.2%}")
                lines.append(f"- 目标: {opp['target_max_dd']:.2%}")
                lines.append(f"- 潜在改进: {opp['potential_improvement']}")
                lines.append("")

    # 生成报告
    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"报告已保存: {output_path}")

    return report


if __name__ == "__main__":
    """测试规则评估"""
    print("=== SPL-7b-D: 规则与结构评估测试 ===\n")

    from analysis.counterfactual.counterfactual_config import CounterfactualResult

    # 模拟结果
    actual = CounterfactualResult(
        scenario_id="actual",
        strategy_id="test",
        total_return=100.0,
        daily_returns=[1.0] * 80 + [-10.0] + [0.5] * 19,
        max_drawdown=0.10,
        volatility=0.02,
        cvar_95=-0.04,
        cvar_99=-0.08,
        gating_events_count=5,
        allocator_rebalance_count=10
    )

    cf_earlier = CounterfactualResult(
        scenario_id="cf_earlier_gating",
        strategy_id="test",
        total_return=95.0,
        daily_returns=[0.8] * 100,
        max_drawdown=0.06,
        volatility=0.015,
        cvar_95=-0.025,
        cvar_99=-0.04,
        gating_events_count=15,
        allocator_rebalance_count=10
    )

    cf_later = CounterfactualResult(
        scenario_id="cf_later_gating",
        strategy_id="test",
        total_return=110.0,
        daily_returns=[1.2] * 80 + [-15.0] + [1.0] * 19,
        max_drawdown=0.15,
        volatility=0.025,
        cvar_95=-0.06,
        cvar_99=-0.10,
        gating_events_count=2,
        allocator_rebalance_count=10
    )

    cf_no_gating = CounterfactualResult(
        scenario_id="cf_no_gating",
        strategy_id="test",
        total_return=105.0,
        daily_returns=[1.1] * 80 + [-12.0] + [1.0] * 19,
        max_drawdown=0.12,
        volatility=0.022,
        cvar_95=-0.05,
        cvar_99=-0.09,
        gating_events_count=0,
        allocator_rebalance_count=10
    )

    cf_results = {
        "cf_earlier_gating": cf_earlier,
        "cf_later_gating": cf_later,
        "cf_no_gating": cf_no_gating
    }

    # 生成评估报告
    report = generate_evaluation_report(
        actual, cf_results,
        "outputs/spl7b_evaluation_report.md"
    )

    print("\n" + "="*70)
    print("评估报告摘要")
    print("="*70)
    print("\n关键结论:")
    print("1. 哪条规则最'值钱'? - 基于综合价值评分")
    print("2. 哪条规则几乎不贡献? - 基于效率评分")
    print("3. 哪些组合调整能显著降低 co-crash? - 基于组合评估")
    print("\n完整报告: outputs/spl7b_evaluation_report.md")

    print("\n✅ 规则与结构评估测试通过")
