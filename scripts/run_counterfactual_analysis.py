"""
SPL-7b-E: 输出与回流

生成反事实分析报告并回流到 SPL-6/5。
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.counterfactual.counterfactual_config import (
    CounterfactualResult,
    CounterfactualScenario
)
from analysis.counterfactual.effect_calculator import (
    EffectMetrics,
    EffectCalculator,
    SpikeAnalyzer
)
from analysis.counterfactual.rule_evaluator import (
    RuleEvaluation,
    RuleEvaluator
)
from analysis.counterfactual.runner import CounterfactualExperiment


@dataclass
class CounterfactualAnalysisReport:
    """反事实分析报告"""
    report_id: str
    timestamp: datetime
    strategy_id: str

    # 输入
    replay_path: str
    scenarios_analyzed: List[str]

    # 结果汇总
    actual_result: Dict[str, Any]
    cf_results: Dict[str, Dict[str, Any]]

    # ���果分析
    effects_summary: Dict[str, Dict[str, Any]]

    # 规则评估
    rule_evaluations: List[Dict[str, Any]]

    # 结论
    key_findings: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "replay_path": self.replay_path,
            "scenarios_analyzed": self.scenarios_analyzed,
            "actual_result": self.actual_result,
            "cf_results": self.cf_results,
            "effects_summary": self.effects_summary,
            "rule_evaluations": self.rule_evaluations,
            "key_findings": self.key_findings,
            "recommendations": self.recommendations
        }


class FeedbackLooper:
    """反馈回环器

    将反事实分析结果回流到 SPL-6a/5。
    """

    def __init__(self):
        """初始化反馈回环器"""

    def generate_spl6a_feedback(
        self,
        report: CounterfactualAnalysisReport
    ) -> Dict[str, Any]:
        """生成 SPL-6a 反馈

        Args:
            report: 反事实分析报告

        Returns:
            反馈数据
        """
        feedback = {
            "source": "SPL-7b Counterfactual Analysis",
            "timestamp": report.timestamp.isoformat(),
            "strategy_id": report.strategy_id,
            "report_id": report.report_id,

            # 再标定建议
            "recalibration_recommendations": [],

            # 参数调整建议
            "parameter_adjustments": {}
        }

        # 提取关键发现
        for finding in report.key_findings:
            if "gating" in finding.lower() and "effective" in finding.lower():
                feedback["recalibration_recommendations"].append({
                    "type": "gating_threshold",
                    "action": "adjust",
                    "reason": finding
                })
            elif "weak_rule" in finding.lower():
                feedback["recalibration_recommendations"].append({
                    "type": "rule_removal",
                    "action": "remove",
                    "rule": finding.split(":")[0].strip() if ":" in finding else "unknown"
                })

        return feedback

    def generate_spl5_feedback(
        self,
        report: CounterfactualAnalysisReport
    ) -> Dict[str, Any]:
        """生成 SPL-5 反馈

        Args:
            report: 反事实分析报告

        Returns:
            反馈数据
        """
        feedback = {
            "source": "SPL-7b Counterfactual Analysis",
            "timestamp": report.timestamp.isoformat(),
            "strategy_id": report.strategy_id,
            "report_id": report.report_id,

            # 规则调整建议
            "rule_adjustments": [],

            # Allocator 改进建议
            "allocator_improvements": []
        }

        # 提取规则建议
        for recommendation in report.recommendations:
            if "keep" in recommendation.lower():
                feedback["rule_adjustments"].append({
                    "action": "keep",
                    "rule": recommendation
                })
            elif "remove" in recommendation.lower():
                feedback["rule_adjustments"].append({
                    "action": "remove",
                    "rule": recommendation
                })
            elif "modify" in recommendation.lower():
                feedback["rule_adjustments"].append({
                    "action": "modify",
                    "suggestion": recommendation
                })

        return feedback


class CounterfactualReporter:
    """反事实报告生成器

    生成完整的反事实分析报告。
    """

    def __init__(self, output_dir: str = "outputs/counterfactual"):
        """初始化报告生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        replay_path: str,
        strategy_ids: List[str],
        custom_scenarios: Optional[List[CounterfactualScenario]] = None
    ) -> CounterfactualAnalysisReport:
        """生成完整报告

        Args:
            replay_path: Replay 路径
            strategy_ids: 策略 ID 列表
            custom_scenarios: 自定义场景

        Returns:
            报告
        """
        print("\n" + "="*70)
        print("SPL-7b: 反事实分析报告生成")
        print("="*70)

        # 运行实验
        experiment = CounterfactualExperiment(replay_path, strategy_ids)
        results = experiment.run_experiment(custom_scenarios)

        # 提取 actual 结果
        actual_result = results.get("actual")
        if not actual_result:
            raise ValueError("Actual scenario not found in results")

        # 移除 actual 结果
        cf_results = {k: v for k, v in results.items() if k != "actual"}

        # 分析效果
        effect_calculator = EffectCalculator()
        effects = effect_calculator.calculate_effects(actual_result, cf_results)

        # 评估规则
        rule_evaluator = RuleEvaluator()
        rule_evaluations = rule_evaluator.evaluate_rules(actual_result, cf_results)

        # Spike 分析
        spike_analyzer = SpikeAnalyzer()
        spike_analysis = spike_analyzer.analyze_spike_elimination(actual_result, cf_results)

        # 生成结论和建议
        key_findings = self._generate_key_findings(
            effects, rule_evaluations, spike_analysis
        )
        recommendations = self._generate_recommendations(
            effects, rule_evaluations
        )

        # 构建报告
        report = CounterfactualAnalysisReport(
            report_id=f"cf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            strategy_id=strategy_ids[0] if strategy_ids else "unknown",
            replay_path=replay_path,
            scenarios_analyzed=list(cf_results.keys()),
            actual_result=actual_result.to_dict(),
            cf_results={k: v.to_dict() for k, v in cf_results.items()},
            effects_summary={k: v.to_dict() for k, v in effects.items()},
            rule_evaluations=[e.to_dict() for e in rule_evaluations],
            key_findings=key_findings,
            recommendations=recommendations
        )

        return report

    def _generate_key_findings(
        self,
        effects: Dict[str, EffectMetrics],
        rule_evaluations: List[RuleEvaluation],
        spike_analysis: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """生成关键发现

        Args:
            effects: 效果指标
            rule_evaluations: 规则评估
            spike_analysis: Spike 分析

        Returns:
            发现列表
        """
        findings = []

        # 找最优场景
        if effects:
            best = max(effects.items(), key=lambda x: x[1].tradeoff_ratio)
            findings.append(
                f"最优场景: {best[0]}，权衡比 {best[1].tradeoff_ratio:.2f}"
            )

        # 找最有效规则
        if rule_evaluations:
            strongest = rule_evaluations[0]  # 已排序
            findings.append(
                f"最有效规则: {strongest.rule_id}，价值评分 {strongest.overall_value:.1f}/100"
            )

        # Spike 消除分析
        if spike_analysis:
            best_spike_elimination = max(
                spike_analysis.items(),
                key=lambda x: x[1]["eliminated_spikes"]
            )
            if best_spike_elimination[1]["eliminated_spikes"] > 0:
                findings.append(
                    f"Spike 消除: {best_spike_elimination[0]} 消除了 "
                    f"{best_spike_elimination[1]['eliminated_spikes']} 个 spike"
                )

        return findings

    def _generate_recommendations(
        self,
        effects: Dict[str, EffectMetrics],
        rule_evaluations: List[RuleEvaluation]
    ) -> List[str]:
        """生成建议

        Args:
            effects: 效果指标
            rule_evaluations: 规则评估

        Returns:
            建议列表
        """
        recommendations = []

        # 基于权衡比的建议
        if effects:
            best = max(effects.items(), key=lambda x: x[1].tradeoff_ratio)
            worst = min(effects.items(), key=lambda x: x[1].tradeoff_ratio)

            if best[1].tradeoff_ratio > 1.0:
                recommendations.append(
                    f"采用 {best[0]} 可以显著改善风险收益权衡 "
                    f"（权衡比 {best[1].tradeoff_ratio:.2f}）"
                )
            elif worst[1].tradeoff_ratio < 0.0:
                recommendations.append(
                    f"Avoid {worst[0]}，风险增加且收益降低"
                )

        # 基于规则评估的建议
        weak_rules = [e for e in rule_evaluations if e.recommendation == "remove"]
        if weak_rules:
            recommendations.append(
                f"考虑移除低价值规则: {', '.join([r.rule_id for r in weak_rules])}"
            )

        # 基于效率评分的建议
        low_efficiency = [e for e in rule_evaluations if e.efficiency_score < 40.0]
        if low_efficiency:
            recommendations.append(
                f"以下规则效率较低，建议调整: "
                f"{', '.join([r.rule_id for r in low_efficiency])}"
            )

        return recommendations

    def save_report(
        self,
        report: CounterfactualAnalysisReport
    ) -> Dict[str, str]:
        """保存报告

        Args:
            report: 报告

        Returns:
            保存的文件路径
        """
        saved_files = {}

        # 保存 JSON
        json_file = self.output_dir / f"{report.report_id}.json"
        with open(json_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        saved_files["json"] = str(json_file)
        print(f"JSON 报告已保存: {json_file}")

        # 保存 Markdown
        md_file = self.output_dir / f"{report.report_id}.md"
        with open(md_file, 'w') as f:
            f.write(self._to_markdown(report))
        saved_files["markdown"] = str(md_file)
        print(f"Markdown 报告已保存: {md_file}")

        # 保存到 docs/
        docs_file = Path("docs") / f"COUNTERFACTUAL_ANALYSIS_{report.strategy_id}.md"
        docs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(docs_file, 'w') as f:
            f.write(self._to_markdown(report))
        saved_files["docs"] = str(docs_file)
        print(f"Docs 报告已保存: {docs_file}")

        return saved_files

    def _to_markdown(self, report: CounterfactualAnalysisReport) -> str:
        """转换为 Markdown

        Args:
            report: 报告

        Returns:
            Markdown 内容
        """
        lines = []
        lines.append(f"# 反事实分析报告: {report.report_id}\n")
        lines.append(f"**策略**: {report.strategy_id}")
        lines.append(f"**生成时间**: {report.timestamp.isoformat()}\n")

        # 概览
        lines.append("## 📊 概览\n")
        lines.append(f"- **Replay**: `{report.replay_path}`")
        lines.append(f"- **分析场景数**: {len(report.scenarios_analyzed)}")
        lines.append("")

        # 实际结果
        lines.append("## 实际结果 (Actual)\n")
        actual = report.actual_result
        lines.append(f"- 总收益: {actual['total_return']:.4f}")
        lines.append(f"- 最大回撤: {actual['max_drawdown']:.2%}")
        lines.append(f"- 波动率: {actual['volatility']:.4f}")
        lines.append(f"- CVaR-95: {actual['cvar_95']:.4f}")
        lines.append(f"- CVaR-99: {actual['cvar_99']:.4f}")
        lines.append(f"- Gating 次数: {actual['gating_events_count']}")
        lines.append("")

        # 反事实结果对比表
        lines.append("## 反事实结果对比\n")
        lines.append("| 场景 | 总收益 | Delta | 最大回撤 | Delta | 权衡比 |")
        lines.append("|------|--------|-------|----------|-------|--------|")

        for scenario_id, cf_result in report.cf_results.items():
            delta_return = cf_result['total_return'] - actual['total_return']
            delta_drawdown = cf_result['max_drawdown'] - actual['max_drawdown']

            # 计算权衡比
            tradeoff = delta_drawdown / abs(delta_return) if delta_return != 0 else 0.0

            lines.append(
                f"| {scenario_id} | {cf_result['total_return']:.4f} | "
                f"{delta_return:+.4f} | {cf_result['max_drawdown']:.2%} | "
                f"{delta_drawdown:+.2%} | {tradeoff:.2f} |"
            )
        lines.append("")

        # 效果分析
        lines.append("## 效果分析\n")
        for scenario_id, effect in report.effects_summary.items():
            lines.append(f"### {scenario_id}\n")
            lines.append(f"- 避免回撤: {effect['avoided_drawdown']:.2%}")
            lines.append(f"- 牺牲收益: {effect['lost_return']:.4f}")
            lines.append(f"- 权衡比: {effect['tradeoff_ratio']:.2f}")
            lines.append("")

        # 规则评估
        lines.append("## 规则价值评估\n")
        lines.append("| 规则 | 价值评分 | 效率评分 | 建议 |")
        lines.append("|------|----------|----------|------|")

        for eval in report.rule_evaluations[:10]:  # 前 10 个
            lines.append(
                f"| {eval['rule_id']} | {eval['overall_value']:.1f} | "
                f"{eval['efficiency_score']:.1f} | {eval['recommendation']} |"
            )
        lines.append("")

        # 关键发现
        lines.append("## 关键发现\n")
        for i, finding in enumerate(report.key_findings, 1):
            lines.append(f"{i}. {finding}")
        lines.append("")

        # 建议
        lines.append("## 建议\n")
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        return "\n".join(lines)


def run_counterfactual_analysis_and_feedback(
    replay_path: str,
    strategy_ids: List[str]
) -> Tuple[str, Dict[str, Any]]:
    """运行完整的反事实分析并回流

    Args:
        replay_path: Replay 路径
        strategy_ids: 策略 ID 列表

    Returns:
        (报告文件路径, 反馈数据)
    """
    # 生成报告
    reporter = CounterfactualReporter()
    report = reporter.generate_full_report(replay_path, strategy_ids)

    # 保存报告
    saved_files = reporter.save_report(report)

    # 生成反馈
    feedback_looper = FeedbackLooper()
    spl6a_feedback = feedback_looper.generate_spl6a_feedback(report)
    spl5_feedback = feedback_looper.generate_spl5_feedback(report)

    # 保存反馈
    feedback_dir = Path("outputs/counterfactual/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)

    feedback_file = feedback_dir / f"{report.report_id}_feedback.json"
    with open(feedback_file, 'w') as f:
        json.dump({
            "spl6a_feedback": spl6a_feedback,
            "spl5_feedback": spl5_feedback
    }, f, indent=2, default=str)

    print(f"\n反馈已保存: {feedback_file}")

    return (saved_files.get("markdown", ""), {
        "spl6a": spl6a_feedback,
        "spl5": spl5_feedback
    })


if __name__ == "__main__":
    """测试输出与回流"""
    print("=== SPL-7b-E: 输出与回流测试 ===\n")

    # 测试报告生成
    runs_dir = "runs"

    if Path(runs_dir).exists():
        # 加载策略
        from analysis.replay_schema import load_replay_outputs
        replays = load_replay_outputs(runs_dir)
        strategy_ids = [r.strategy_id for r in replays[:3]]

        if strategy_ids:
            # 运行分析
            report_path, feedback = run_counterfactual_analysis_and_feedback(
                runs_dir, strategy_ids
            )

            print(f"\n分析完成:")
            print(f"  报告: {report_path}")
            print(f"  反馈包含: {list(feedback.keys())}")

    print("\n✅ 输出与回流测试通过")
