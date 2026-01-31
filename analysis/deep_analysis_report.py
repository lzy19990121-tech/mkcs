"""
深度风险分析报告生成器

整合扰动测试、结构分析、风险包络、可执行规则等所有分析结果
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from analysis.replay_schema import ReplayOutput, load_replay_outputs
from analysis.perturbation_test import PerturbationTester, PerturbationResult
from analysis.structural_analysis import StructuralAnalyzer, StructuralAnalysisResult, RiskPatternType
from analysis.risk_envelope import RiskEnvelopeBuilder, RiskEnvelope, format_envelope_report
from analysis.actionable_rules import RiskRuleGenerator, RiskRuleset, format_ruleset_report
from analysis.stability_analysis import StabilityAnalyzer, StabilityReport


class DeepAnalysisReportGenerator:
    """深度风险分析报告生成器

    整合SPL-3b所有分析结果
    """

    def __init__(self, output_dir: str = "runs/deep_analysis"):
        """初始化生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化各分析器
        self.perturbation_tester = PerturbationTester()
        self.structural_analyzer = StructuralAnalyzer()
        self.envelope_builder = RiskEnvelopeBuilder()
        self.rule_generator = RiskRuleGenerator()
        self.stability_analyzer = StabilityAnalyzer()

    def generate_full_report(
        self,
        replay: ReplayOutput,
        window_lengths: List[str] = None
    ) -> str:
        """生成完整的深度分析报告

        Args:
            replay: 回测输出
            window_lengths: 窗口长度列表

        Returns:
            Markdown格式的完整报告
        """
        window_lengths = window_lengths or ["20d", "60d"]

        lines = []
        lines.append(self._generate_header(replay))

        # 对每个窗口长度生成分析
        for window in window_lengths:
            lines.append(f"\n{'='*80}\n")
            lines.append(f"# {window} 窗口 - 深度风险分析\n")
            lines.append(f"{'='*80}\n\n")

            # 1. 扰动测试
            lines.extend(self._generate_perturbation_section(replay, window))

            # 2. 结构分析
            lines.extend(self._generate_structure_section(replay, window))

            # 3. 风险包络
            lines.extend(self._generate_envelope_section(replay, window))

            # 4. 可执行规则
            lines.extend(self._generate_rules_section(replay, window))

        # 5. 总结与建议
        lines.extend(self._generate_summary_section(replay, window_lengths))

        # 6. 元数据
        lines.extend(self._generate_metadata_section(replay))

        return "\n".join(lines)

    def _generate_header(self, replay: ReplayOutput) -> str:
        """生成报告头部"""
        lines = []

        lines.append("╔════════════════════════════════════════════════════════════════╗\n")
        lines.append("║         SPL-3b 深度风险分析报告                                    ║\n")
        lines.append("║         深度扰动测试 + 结构分析 + 风险包络 + 可执行规则           ║\n")
        lines.append("╚════════════════════════════════════════════════════════════════╝\n")

        lines.append("**生成时间**: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        lines.append("**策略ID**: " + replay.strategy_id + "\n")
        lines.append("**运行ID**: " + replay.run_id + "\n")
        lines.append("**回测期间**: " + str(replay.start_date) + " 至 " + str(replay.end_date) + "\n")
        lines.append("**总收益**: " + f"{(float(replay.final_equity)/float(replay.initial_cash) - 1)*100:.2f}%\n")

        lines.append("\n**可追溯性**:\n")
        lines.append(f"- Git Commit: {replay.commit_hash}\n")
        lines.append(f"- Config Hash: {replay.config_hash}\n")

        lines.append("\n---\n")

        return "".join(lines)

    def _generate_perturbation_section(
        self,
        replay: ReplayOutput,
        window_length: str
    ) -> List[str]:
        """生成扰动测试部分"""
        lines = []

        lines.append("## 一、Worst-Case 稳定性检验（扰动测试）\n")

        # 执行扰动测试
        perturbation_results = self.perturbation_tester.test_perturbations(
            replay, window_length
        )

        # 分类稳定性
        stability_label = self.perturbation_tester.classify_stability(
            perturbation_results
        )

        lines.append(f"**稳定性标签**: {stability_label}\n")

        # 详细结果表格
        lines.append("\n### 扰动测试详情\n")
        lines.append("| 扰动类型 | 扰动值 | 原始收益 | 扰动后收益 | 差异 | 同一窗口 | 在Top-K |")
        lines.append("|----------|--------|----------|------------|------|----------|---------|")

        for r in perturbation_results:
            perturbed_return = f"{r.worst_window.total_return*100:.2f}%" if r.worst_window else "N/A"
            return_diff = f"{r.return_diff*100:+.2f}%"

            lines.append(
                f"| {r.perturbation_type.value} | "
                f"{r.perturbation_value:+.1%} | "
                f"{r.original_worst_window.total_return*100:.2f}% | "
                f"{perturbed_return} | "
                f"{return_diff} | "
                f"{'✓' if r.is_same_window else '✗'} | "
                f"{'✓' if r.is_in_top_k else '✗'} |"
            )

        lines.append("\n### 判断标准\n")
        lines.append("- **Stable**: 扰动后最坏窗口仍在同一时间区间（±2天）")
        lines.append("- **Weakly Stable**: 扰动后最坏窗口在原始Top-K中")
        lines.append("- **Fragile**: 扰动后最坏窗口不在Top-K中\n")

        return lines

    def _generate_structure_section(
        self,
        replay: ReplayOutput,
        window_length: str
    ) -> List[str]:
        """生成结构分析部分"""
        lines = []

        lines.append("## 二、Worst-Case 结构确认\n")

        # 执行结构分析
        structural_result = self.structural_analyzer.analyze_structure(
            replay, window_length, top_k=5
        )

        # 风险类型
        risk_type_name = {
            RiskPatternType.STRUCTURAL: "结构性风险 (Structural Risk Pattern)",
            RiskPatternType.SINGLE_OUTLIER: "单一异常 (Single-Outlier Risk)"
        }[structural_result.risk_pattern_type]

        lines.append(f"**风险类型**: {risk_type_name}\n")

        # 形态指标
        lines.append("\n### 形态指标\n")
        lines.append("| 指标 | 数值 | 说明 |")
        lines.append("|------|------|------|")
        lines.append(f"| 平均MDD | {structural_result.pattern_metrics.avg_mdd*100:.2f}% | Top-K平均 |")
        lines.append(f"| MDD标准差 | {structural_result.pattern_metrics.std_mdd*100:.2f}% | |")
        lines.append(f"| MDD变异系数 | {structural_result.pattern_metrics.mdd_cv:.3f} | <0.3为稳定 |")
        lines.append(f"| 形态相似度 | {structural_result.pattern_metrics.pattern_similarity:.3f} | >0.7为高度相似 |")
        lines.append(f"| 回撤曲线相关性 | {structural_result.drawdown_correlation:.3f} | |")
        lines.append(f"| 形态一致性 | {structural_result.shape_consistency:.3f} | |")

        # Top-K窗口详情
        lines.append("\n### Top-K 最坏窗口\n")
        lines.append("| 排名 | 窗口ID | 收益 | MDD | 回撤形态 |")
        lines.append("|------|--------|------|-----|----------|")

        for i, window in enumerate(structural_result.top_k_windows, 1):
            lines.append(
                f"| {i} | {window.window_id} | "
                f"{window.total_return*100:7.2f}% | "
                f"{window.max_drawdown*100:5.1f}% | "
                f"{window.drawdown_pattern} |"
            )

        return lines

    def _generate_envelope_section(
        self,
        replay: ReplayOutput,
        window_length: str
    ) -> List[str]:
        """生成风险包络部分"""
        lines = []

        lines.append("## 三、Worst-Case Envelope（风险边界）\n")

        # 构建风险包络
        envelope = self.envelope_builder.build_envelope(replay, window_length)

        # 使用格式化函数
        lines.append(format_envelope_report(envelope))

        return lines

    def _generate_rules_section(
        self,
        replay: ReplayOutput,
        window_length: str
    ) -> List[str]:
        """生成可执行规则部分"""
        lines = []

        lines.append("## 四、可执行风险规则\n")

        # 生成各项分析
        stability_report = self.stability_analyzer.analyze_replay(replay)
        envelope = self.envelope_builder.build_envelope(replay, window_length)
        structural_result = self.structural_analyzer.analyze_structure(replay, window_length)

        # 生成规则
        ruleset = self.rule_generator.generate_rules(
            replay, stability_report, envelope, structural_result
        )

        # 使用格式化函数
        lines.append(format_ruleset_report(ruleset))

        return lines

    def _generate_summary_section(
        self,
        replay: ReplayOutput,
        window_lengths: List[str]
    ) -> List[str]:
        """生成总结部分"""
        lines = []

        lines.append("## 五、综合结论\n")

        # 汇总所有窗口的稳定性
        lines.append("### 稳定性总评\n")
        lines.append("| 窗口长度 | 稳定性标签 | 风险类型 | 规则数量 |")
        lines.append("|----------|------------|----------|----------|")

        for window in window_lengths:
            perturbation_results = self.perturbation_tester.test_perturbations(
                replay, window
            )
            stability_label = self.perturbation_tester.classify_stability(
                perturbation_results
            )

            structural_result = self.structural_analyzer.analyze_structure(
                replay, window, top_k=5
            )

            # 生成规则
            stability_report = self.stability_analyzer.analyze_replay(replay)
            envelope = self.envelope_builder.build_envelope(replay, window)
            ruleset = self.rule_generator.generate_rules(
                replay, stability_report, envelope, structural_result
            )

            risk_type = structural_result.risk_pattern_type.value

            lines.append(
                f"| {window} | {stability_label} | {risk_type} | {len(ruleset.rules)} |"
            )

        lines.append("\n### 适用性评估\n")

        # 综合评估
        total_rules = 0
        for window in window_lengths:
            stability_report = self.stability_analyzer.analyze_replay(replay)
            envelope = self.envelope_builder.build_envelope(replay, window)
            structural_result = self.structural_analyzer.analyze_structure(replay, window)
            ruleset = self.rule_generator.generate_rules(
                replay, stability_report, envelope, structural_result
            )
            total_rules += len(ruleset.rules)

        if total_rules == 0:
            lines.append("✅ **策略表现良好，无需特殊风险控制**\n")
        elif total_rules <= 2:
            lines.append("⚠️ **策略存在一定风险，建议启用部分风控规则**\n")
        else:
            lines.append("🔴 **策略风险较高，必须启用完整风控机制**\n")

        return lines

    def _generate_metadata_section(self, replay: ReplayOutput) -> List[str]:
        """生成元数据部分"""
        lines = []

        lines.append("\n---\n")
        lines.append("## 六、元数据与可复现性\n")

        lines.append("### 分析配置\n")
        lines.append(f"- 扰动类型数: {len(self.perturbation_tester.perturbation_configs)}")
        lines.append(f"- 分析窗口: {self.perturbation_tester.perturbation_configs[0].epsilon} ε")
        lines.append(f"- 置信水平: {self.envelope_builder.confidence_level*100:.0f}%")

        lines.append("\n### 可复现性审计\n")
        lines.append("本报告可完全复现：\n")
        lines.append("```bash")
        lines.append(f"# 检出代码版本")
        lines.append(f"git checkout {replay.commit_hash}")
        lines.append("")
        lines.append(f"# 使用相同配置")
        lines.append(f"python -c \"")
        lines.append(f"from analysis import DeepAnalysisReportGenerator, load_replay_outputs")
        lines.append(f"replays = load_replay_outputs('runs')")
        lines.append(f"replay = [r for r in replays if r.run_id == '{replay.run_id}'][0]")
        lines.append(f"generator = DeepAnalysisReportGenerator()")
        lines.append(f"report = generator.generate_full_report(replay)")
        lines.append(f"print(report)")
        lines.append(f"\"")
        lines.append("```\n")

        lines.append("---\n")
        lines.append(f"*本报告由 MKCS SPL-3b 深度风险分析系统自动生成*\n")

        return lines

    def save_report(
        self,
        replay: ReplayOutput,
        window_lengths: List[str] = None
    ) -> str:
        """生成并保存报告

        Args:
            replay: 回测输出
            window_lengths: 窗口长度列表

        Returns:
            报告文件路径
        """
        # 生成报告
        report_content = self.generate_full_report(replay, window_lengths)

        # 保存文件
        filename = f"{replay.run_id}_deep_analysis_v3b.md"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return str(filepath)

    def save_analysis_data(
        self,
        replay: ReplayOutput,
        window_lengths: List[str] = None
    ) -> str:
        """保存分析数据（JSON格式）

        Args:
            replay: 回测输出
            window_lengths: 窗口长度列表

        Returns:
            数据文件路径
        """
        window_lengths = window_lengths or ["20d", "60d"]

        analysis_data = {
            "strategy_id": replay.strategy_id,
            "run_id": replay.run_id,
            "commit_hash": replay.commit_hash,
            "config_hash": replay.config_hash,
            "generated_at": datetime.now().isoformat(),
            "windows": {}
        }

        for window in window_lengths:
            # 扰动测试
            perturbation_results = self.perturbation_tester.test_perturbations(
                replay, window
            )

            # 结构分析
            structural_result = self.structural_analyzer.analyze_structure(
                replay, window, top_k=5
            )

            # 风险包络
            envelope = self.envelope_builder.build_envelope(replay, window)

            # 风险规则
            stability_report = self.stability_analyzer.analyze_replay(replay)
            ruleset = self.rule_generator.generate_rules(
                replay, stability_report, envelope, structural_result
            )

            # 稳定性标签
            stability_label = self.perturbation_tester.classify_stability(
                perturbation_results
            )

            analysis_data["windows"][window] = {
                "stability_label": stability_label,
                "risk_pattern_type": structural_result.risk_pattern_type.value,
                "pattern_metrics": {
                    "avg_mdd": structural_result.pattern_metrics.avg_mdd,
                    "std_mdd": structural_result.pattern_metrics.std_mdd,
                    "mdd_cv": structural_result.pattern_metrics.mdd_cv,
                    "pattern_similarity": structural_result.pattern_metrics.pattern_similarity,
                },
                "envelope": envelope.to_dict(),
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "rule_name": r.rule_name,
                        "rule_type": r.rule_type.value,
                        "trigger_metric": r.trigger_metric,
                        "trigger_threshold": r.trigger_threshold,
                        "description": r.description
                    }
                    for r in ruleset.rules
                ]
            }

        # 保存JSON
        filename = f"{replay.run_id}_deep_analysis_v3b.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)

        return str(filepath)


def generate_all_deep_reports(
    run_dir: str,
    output_dir: str = "runs/deep_analysis"
) -> Dict[str, Dict[str, str]]:
    """生成所有replay的深度分析报告

    Args:
        run_dir: runs目录路径
        output_dir: 输出目录

    Returns:
        {run_id: {"markdown": path, "json": path}}
    """
    replays = load_replay_outputs(run_dir)
    generator = DeepAnalysisReportGenerator(output_dir)

    all_paths = {}

    for replay in replays:
        # 生成Markdown报告
        md_path = generator.save_report(replay)

        # 生成JSON数据
        json_path = generator.save_analysis_data(replay)

        all_paths[replay.run_id] = {
            "markdown": md_path,
            "json": json_path
        }

        print(f"生成报告: {replay.run_id}")
        print(f"  Markdown: {md_path}")
        print(f"  JSON: {json_path}")

    return all_paths


if __name__ == "__main__":
    """测试代码"""
    print("=== DeepAnalysisReportGenerator 测试 ===\n")

    # 生成所有报告
    paths = generate_all_deep_reports("runs")

    print(f"\n共生成 {len(paths)} 份深度分析报告")

    # 显示第一个报告的内容摘要
    if paths:
        first_run_id = list(paths.keys())[0]
        md_path = paths[first_run_id]["markdown"]

        print(f"\n报告预览 ({md_path}):")
        print("=" * 80)

        with open(md_path) as f:
            content = f.read()
            # 显示前500字符
            print(content[:500])
            print("\n...")

        print("=" * 80)

    print("\n✓ 测试通过")
