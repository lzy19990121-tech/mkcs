"""
Risk Card 生成器

生成 Markdown 格式的风险分析报告
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from analysis.replay_schema import ReplayOutput, load_replay_outputs
from analysis.window_scanner import WindowScanner, WindowMetrics
from analysis.stability_analysis import StabilityAnalyzer, StabilityReport
from analysis.multi_strategy_comparison import (
    MultiStrategyComparator,
    StrategyWorstCaseSummary
)


@dataclass
class RiskCardConfig:
    """Risk Card 配置"""
    title: str = "风险分析报告"
    author: str = "MKCS Risk Analysis"
    version: str = "1.0.0"

    # 显示选项
    show_all_windows: bool = True
    top_k_worst: int = 5
    include_charts: bool = False

    # 输出格式
    output_format: str = "markdown"  # markdown / html / pdf


class RiskCardGenerator:
    """Risk Card 生成器

    生成最坏情况风险分析报告
    """

    def __init__(self, config: RiskCardConfig = None):
        """初始化生成器

        Args:
            config: 配置对象
        """
        self.config = config or RiskCardConfig()
        self.window_scanner = WindowScanner(top_k=self.config.top_k_worst)
        self.stability_analyzer = StabilityAnalyzer()
        self.multi_comparator = MultiStrategyComparator()

    def generate_for_replay(self, replay: ReplayOutput, output_path: str = None) -> str:
        """为单个replay生成Risk Card

        Args:
            replay: 回测输出
            output_path: 输出文件路径

        Returns:
            Markdown内容
        """
        # 分析最坏窗口
        worst_windows = self.window_scanner.find_worst_windows(replay, top_k=self.config.top_k_worst)

        # 分析稳定性
        stability = self.stability_analyzer.analyze_replay(replay)

        # 生成Markdown
        md = self._generate_single_replay_md(replay, worst_windows, stability)

        # 保存文件
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md)

        return md

    def generate_for_comparison(
        self,
        replays: List[ReplayOutput],
        output_path: str = None
    ) -> str:
        """生成多策略对比Risk Card

        Args:
            replays: 回测输出列表
            output_path: 输出文件路径

        Returns:
            Markdown内容
        """
        # 对比策略
        summaries = self.multi_comparator.compare_strategies(replays)

        # 生成Markdown
        md = self._generate_comparison_md(summaries)

        # 保存文件
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md)

        return md

    def _generate_single_replay_md(
        self,
        replay: ReplayOutput,
        worst_windows: List[WindowMetrics],
        stability: StabilityReport
    ) -> str:
        """生成单个replay的Markdown报告"""
        lines = []

        # 标题
        lines.append(f"# {self.config.title}\n")
        lines.append(f"**策略**: {replay.strategy_name}\n")
        lines.append(f"**运行ID**: {replay.run_id}\n")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")

        # 1. 基本信息
        lines.append("## 1. 基本信息\n")
        lines.append(f"- **策略ID**: {replay.strategy_id}\n")
        lines.append(f"- **回测期间**: {replay.start_date} 至 {replay.end_date}\n")
        lines.append(f"- **初始资金**: ${replay.initial_cash:,.2f}\n")
        lines.append(f"- **最终权益**: ${replay.final_equity:,.2f}\n")
        lines.append(f"- **总收益**: {(float(replay.final_equity) / float(replay.initial_cash) - 1) * 100:.2f}%\n")
        lines.append(f"- **Git Commit**: {replay.commit_hash}\n")
        lines.append(f"- **Config Hash**: {replay.config_hash}\n")
        lines.append("\n")

        # 2. 稳定性评分
        lines.append("## 2. 稳定性评分\n")
        lines.append(f"### 总分: {stability.stability_score:.1f} / 100\n\n")
        lines.append("| 指标 | 数值 |\n")
        lines.append("|------|------|\n")
        lines.append(f"| 收益方差 | {stability.return_variance * 100:.4f}% |\n")
        lines.append(f"| 收益极差 | {stability.return_range * 100:.2f}% |\n")
        lines.append(f"| MDD一致性 | {stability.mdd_consistency:.3f} |\n")
        lines.append(f"| 最差CVaR | {stability.worst_cvar:.3f} |\n")
        lines.append("\n")

        # 3. 最坏窗口详情
        lines.append("## 3. Top-K 最坏窗口\n\n")

        for i, window in enumerate(worst_windows, 1):
            lines.append(f"### #{i} - {window.window_length} 窗口\n\n")
            lines.append(f"- **窗口ID**: {window.window_id}\n")
            lines.append(f"- **时间范围**: {window.start_date.strftime('%Y-%m-%d')} 至 {window.end_date.strftime('%Y-%m-%d')}\n")
            lines.append(f"- **窗口收益**: {window.total_return * 100:.2f}%\n")
            lines.append(f"- **窗口盈亏**: ${window.total_pnl:,.2f}\n")

            lines.append("\n#### 风险指标\n")
            lines.append("| 指标 | 数值 | 说明 |\n")
            lines.append("|------|------|------|\n")
            lines.append(f"| 最大回撤 | {window.max_drawdown * 100:.2f}% | 窗口内最大权益回撤 |\n")
            lines.append(f"| 回撤持续 | {window.max_dd_duration} 天 | 最大回撤持续天数 |\n")
            lines.append(f"| 回撤恢复 | {window.max_dd_recovery_time} 天 | 回撤恢复时间 (-1表示未恢复) |\n")
            lines.append(f"| Ulcer指数 | {window.ulcer_index:.2f} | 回撤面积指标 |\n")
            lines.append(f"| 下行波动 | {window.downside_deviation:.4f} | 负收益波动率 |\n")
            lines.append(f"| 95% CVaR | {window.cvar_95:.3f} | 条件风险价值 |\n")
            lines.append(f"| 尾部均值 | {window.tail_mean:.4f} | 最差5%日均收益 |\n")

            lines.append("\n#### 尖刺风险\n")
            lines.append(f"- **最大单步亏损**: ${window.max_single_loss:,.2f}\n")
            lines.append(f"- **最长连亏**: {window.max_consecutive_losses} 天\n")
            if window.loss_distribution:
                lines.append(f"- **亏损分布**: ")
                for length, count in sorted(window.loss_distribution.items()):
                    lines.append(f"{length}天×{count}次 ")
                lines.append("\n")

            lines.append("\n#### 回撤形态\n")
            lines.append(f"- **形态**: {window.drawdown_pattern}\n")
            lines.append(f"- **是否未恢复**: {'是' if window.is_censored else '否'}\n")

            lines.append("\n---\n\n")

        # 4. 各窗口最坏情况汇总
        lines.append("## 4. 各窗口长度最坏情况汇总\n\n")
        lines.append("| 窗口长度 | 最坏收益 | 最大回撤 | CVaR | 回撤形态 |\n")
        lines.append("|----------|----------|----------|------|----------|\n")

        for window_len, worst in stability.worst_by_window.items():
            lines.append(
                f"| {window_len} | "
                f"{worst.total_return * 100:8.2f}% | "
                f"{worst.max_drawdown * 100:6.1f}% | "
                f"{worst.cvar_95:5.2f} | "
                f"{worst.drawdown_pattern} |\n"
            )
        lines.append("\n")

        # 5. 可复现性审计
        lines.append("## 5. 可复现性审计\n\n")
        lines.append("本报告可通过以下信息完全复现：\n\n")
        lines.append(f"```bash\n")
        lines.append(f"# 使用相同的配置和Git commit\n")
        lines.append(f"git checkout {replay.commit_hash}\n")
        lines.append(f"python -m agent.runner --config-hash {replay.config_hash}\n")
        lines.append(f"```\n\n")

        # 6. 配置详情
        lines.append("## 6. 配置详情\n\n")
        lines.append("```json\n")
        lines.append(json.dumps(replay.config, indent=2, ensure_ascii=False))
        lines.append("\n```\n")

        # 页脚
        lines.append("\n---\n")
        lines.append(f"*本报告由 {self.config.author} 自动生成 (v{self.config.version})*\n")

        return "".join(lines)

    def _generate_comparison_md(self, summaries: List[StrategyWorstCaseSummary]) -> str:
        """生成多策略对比的Markdown报告"""
        lines = []

        # 标题
        lines.append(f"# 多策略风险对比报告\n")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**对比策略数**: {len(summaries)}\n")
        lines.append("---\n")

        # 1. 总览
        lines.append("## 1. 稳定性排名\n\n")
        lines.append("| 排名 | 策略 | 稳定性评分 | 收益方差 | MDD一致性 | 最差CVaR |\n")
        lines.append("|------|------|------------|----------|-----------|---------|\n")

        for i, summary in enumerate(summaries, 1):
            if summary.stability_report:
                lines.append(
                    f"| {i} | "
                    f"{summary.strategy_name} | "
                    f"{summary.stability_report.stability_score:6.1f} | "
                    f"{summary.stability_report.return_variance*100:6.3f}% | "
                    f"{summary.stability_report.mdd_consistency:8.3f} | "
                    f"{summary.stability_report.worst_cvar:7.3f} |\n"
                )
        lines.append("\n")

        # 2. 各窗口对比
        lines.append("## 2. 各窗口最坏情况对比\n\n")

        for window in ["1d", "5d", "20d", "60d", "120d", "250d"]:
            lines.append(f"### {window} 窗口\n\n")

            # 找到该窗口下的所有有效策略
            valid = [s for s in summaries if getattr(s, f"worst_{window}") is not None]

            if not valid:
                lines.append("*无数据*\n\n")
                continue

            lines.append("| 策略 | 最坏收益 | 最大回撤 | CVaR | 回撤形态 |\n")
            lines.append("|------|----------|----------|------|----------|\n")

            for summary in valid:
                worst = getattr(summary, f"worst_{window}")
                lines.append(
                    f"| {summary.strategy_name} | "
                    f"{worst.total_return * 100:8.2f}% | "
                    f"{worst.max_drawdown * 100:6.1f}% | "
                    f"{worst.cvar_95:5.2f} | "
                    f"{worst.drawdown_pattern} |\n"
                )

            # 找到最优
            best = max(valid, key=lambda s: getattr(s, f"worst_{window}").total_return)
            lines.append(f"\n**最优策略**: {best.strategy_name} ({getattr(best, f'worst_{window}').total_return * 100:.2f}%)\n\n")

        # 3. 推荐策略
        lines.append("## 3. 推荐策略\n\n")

        most_stable = self.multi_comparator.find_most_stable(summaries)
        if most_stable:
            lines.append(f"### 最稳定策略\n\n")
            lines.append(f"**{most_stable.strategy_name}** (评分: {most_stable.stability_report.stability_score:.1f}/100)\n\n")
            lines.append(f"- 收益方差: {most_stable.stability_report.return_variance * 100:.4f}%\n")
            lines.append(f"- MDD一致性: {most_stable.stability_report.mdd_consistency:.3f}\n")
            lines.append(f"- 最差CVaR: {most_stable.stability_report.worst_cvar:.3f}\n\n")

        lines.append("\n---\n")
        lines.append(f"*本报告由 {self.config.author} 自动生成 (v{self.config.version})*\n")

        return "".join(lines)


def generate_risk_cards(
    run_dir: str,
    output_dir: str = "runs/risk_analysis"
) -> Dict[str, str]:
    """为所有replay生成Risk Card

    Args:
        run_dir: runs目录路径
        output_dir: 输出目录

    Returns:
        {run_id: output_path}
    """
    replays = load_replay_outputs(run_dir)
    generator = RiskCardGenerator()

    output_paths = {}

    # 为每个replay生成报告
    for replay in replays:
        output_path = f"{output_dir}/{replay.run_id}_risk_card.md"
        generator.generate_for_replay(replay, output_path)
        output_paths[replay.run_id] = output_path
        print(f"生成: {output_path}")

    # 生成对比报告
    comparison_path = f"{output_dir}/comparison_risk_card.md"
    generator.generate_for_comparison(replays, comparison_path)
    print(f"生成对比报告: {comparison_path}")

    return output_paths


if __name__ == "__main__":
    """测试代码"""
    print("=== RiskCardGenerator 测试 ===\n")

    # 生成所有Risk Card
    paths = generate_risk_cards("runs", "runs/risk_analysis")

    print(f"\n生成了 {len(paths)} 个报告")

    # 显示第一个报告的内容
    if paths:
        first_path = list(paths.values())[0]
        print(f"\n报告预览 ({first_path}):")
        print("=" * 60)
        with open(first_path) as f:
            content = f.read()
            print(content[:500] + "...")
        print("=" * 60)

    print("\n✓ 测试通过")
