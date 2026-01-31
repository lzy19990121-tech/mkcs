"""
Portfolio Risk Report Generator for SPL-4b

Generates comprehensive portfolio-level risk analysis reports.
"""

from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

from analysis.portfolio.portfolio_builder import Portfolio
from analysis.portfolio.portfolio_scanner import PortfolioWindowScanner, PortfolioWindowMetrics
from analysis.portfolio.synergy_analyzer import SynergyAnalyzer, SynergyRiskReport


class PortfolioRiskReportGenerator:
    """B4: Generate portfolio-level risk analysis report

    Creates comprehensive markdown reports covering:
    - Portfolio worst-case summary
    - Unsafe strategy combinations
    - Required gating/allocation rules
    """

    def __init__(self):
        """Initialize report generator"""
        self.window_scanner = PortfolioWindowScanner()
        self.synergy_analyzer = SynergyAnalyzer()

    def generate_report(
        self,
        portfolio: Portfolio,
        worst_windows: Dict[str, List[PortfolioWindowMetrics]],
        synergy_report: SynergyRiskReport,
        output_path: Path = None
    ) -> str:
        """Generate markdown report

        Args:
            portfolio: Portfolio to report on
            worst_windows: Worst windows by length
            synergy_report: Synergy risk report
            output_path: Optional path to save report

        Returns:
            Markdown report string
        """
        lines = []

        # Header
        lines.append(f"# Portfolio Risk Analysis Report\n")
        lines.append(f"**Generated**: {datetime.now().isoformat()}\n")
        lines.append(f"**Portfolio ID**: {portfolio.portfolio_id}\n")

        # Portfolio Summary
        lines.append("## Portfolio Summary\n")
        lines.append(f"- **Strategies**: {', '.join(portfolio.config.strategy_ids)}")
        lines.append(f"- **Weights**: {portfolio.config.weights}")
        lines.append(f"- **Period**: {portfolio.config.start_date} to {portfolio.config.end_date}")
        lines.append(f"- **Total Return**: {portfolio.total_return*100:.2f}%\n")

        # Worst-Case Summary
        lines.append("## Worst-Case Summary\n")
        lines.append("### Portfolio Worst Windows\n")

        for window_length, windows in worst_windows.items():
            lines.append(f"\n#### {window_length} Windows\n")

            if not windows:
                lines.append("*No worst windows found*\n")
                continue

            for i, window in enumerate(windows[:3], 1):  # Top 3
                lines.append(f"**#{i}**: {window.start_date} → {window.end_date}")
                lines.append(f"- Portfolio Return: {window.window_return*100:.2f}%")
                lines.append(f"- Max Drawdown: {window.max_drawdown*100:.2f}%")
                lines.append(f"- Volatility: {window.volatility*100:.2f}%")

                if window.worst_performers:
                    lines.append(f"- Worst Performers: {', '.join(window.worst_performers)}")

                lines.append(f"- Avg Correlation: {window.avg_correlation:.3f}\n")

        # Synergy Risks
        lines.append("## Synergy Risk Analysis\n")

        # Unsafe combinations
        lines.append("### Unsafe Strategy Combinations\n")
        if synergy_report.unsafe_combinations:
            lines.append(f"Found {len(synergy_report.unsafe_combinations)} unsafe pairs:\n")
            for s1, s2 in synergy_report.unsafe_combinations:
                lines.append(f"- **{s1}** + **{s2}**: Correlation spikes during stress")
            lines.append("")
        else:
            lines.append("*No unsafe combinations detected*\n")

        # Correlation dynamics
        lines.append("### Correlation Dynamics\n")
        lines.append(f"- **Baseline Correlation**: {synergy_report.avg_correlation_baseline:.3f}")
        lines.append(f"- **Stress Correlation**: {synergy_report.avg_correlation_stress:.3f}")
        lines.append(f"- **Correlation Increase**: {(synergy_report.avg_correlation_stress - synergy_report.avg_correlation_baseline)*100:+.1f}%\n")

        # Simultaneous tail losses
        lines.append("### Simultaneous Tail Loss Events\n")
        if synergy_report.simultaneous_tail_losses:
            lines.append(f"Found {len(synergy_report.simultaneous_tail_losses)} events:\n")

            for event in synergy_report.simultaneous_tail_losses[:5]:  # Top 5
                lines.append(f"**{event['window_id']}** ({event['start_date']} to {event['end_date']})")
                lines.append(f"- Strategies in tail loss: {event['count']}")
                lines.append(f"- Portfolio return: {event['portfolio_return']*100:.2f}%")

                strategies_list = [
                    f"{s['strategy_id']} ({s['contribution']*100:.2f}%)"
                    for s in event['strategies_in_tail_loss']
                ]
                lines.append(f"- Affected strategies: {', '.join(strategies_list)}\n")
        else:
            lines.append("*No simultaneous tail loss events detected*\n")

        # Risk budget breaches
        lines.append("### Risk Budget Breaches\n")
        if synergy_report.risk_budget_breaches:
            lines.append(f"Found {len(synergy_report.risk_budget_breaches)} breaches:\n")

            for breach in synergy_report.risk_budget_breaches[:3]:  # Top 3
                lines.append(f"**{breach['window_id']}**")
                lines.append(f"- Portfolio return: {breach['portfolio_return']*100:.2f}%")
                lines.append(f"- Risk budget: {breach['risk_budget']*100:.2f}%")
                lines.append(f"- Excess loss: {breach['excess_loss']*100:.2f}%\n")
        else:
            lines.append("*No risk budget breaches detected*\n")

        # Recommendations
        lines.append("## Recommendations\n")

        recommendations = self._generate_recommendations(
            portfolio,
            worst_windows,
            synergy_report
        )

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}\n")

        # Required gating/allocation rules
        lines.append("## Required Gating/Allocation Rules\n")

        rules = self._generate_rules(
            portfolio,
            worst_windows,
            synergy_report
        )

        if rules:
            for rule in rules:
                lines.append(f"- **{rule['type']}**: {rule['description']}")
                if 'implementation' in rule:
                    lines.append(f"  - Implementation: {rule['implementation']}\n")
                else:
                    lines.append("")
        else:
            lines.append("*No additional rules required*\n")

        # Summary
        lines.append("---\n")
        lines.append("*Generated by SPL-4b Portfolio Risk Analysis*\n")

        report = "\n".join(lines)

        # Save if path provided
        if output_path:
            output_path.write_text(report, encoding='utf-8')
            print(f"✓ Report saved: {output_path}")

        return report

    def _generate_recommendations(
        self,
        portfolio: Portfolio,
        worst_windows: Dict[str, List[PortfolioWindowMetrics]],
        synergy_report: SynergyRiskReport
    ) -> List[str]:
        """Generate portfolio recommendations

        Args:
            portfolio: Portfolio
            worst_windows: Worst windows
            synergy_report: Synergy report

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for unsafe combinations
        if synergy_report.unsafe_combinations:
            recommendations.append(
                f"⚠️  **Reduce allocation** to unsafe strategy pairs: "
                f"{', '.join([f'{s1}+{s2}' for s1, s2 in synergy_report.unsafe_combinations[:3]])}"
            )

        # Check correlation increase
        corr_increase = (
            synergy_report.avg_correlation_stress - synergy_report.avg_correlation_baseline
        )
        if corr_increase > 0.2:
            recommendations.append(
                f"⚠️  **Monitor correlation**: Correlation increases by "
                f"{corr_increase*100:.1f}% during stress periods. "
                f"Consider dynamic de-leveraging."
            )

        # Check simultaneous tail losses
        if synergy_report.simultaneous_tail_losses:
            worst_event = synergy_report.simultaneous_tail_losses[0]
            if worst_event['count'] >= 3:
                recommendations.append(
                    f"⚠️  **Concentration risk**: {worst_event['count']} strategies "
                    f"experience tail loss simultaneously. "
                    f"Consider diversifying strategy approaches."
                )

        # Check risk budget breaches
        if synergy_report.risk_budget_breaches:
            worst_breach = synergy_report.risk_budget_breaches[0]
            recommendations.append(
                f"⚠️  **Reduce position sizing**: Portfolio exceeded risk budget by "
                f"{abs(worst_breach['excess_loss'])*100:.2f}%. "
                f"Consider reducing gross exposure."
            )

        # Positive recommendations
        if not synergy_report.unsafe_combinations and not synergy_report.risk_budget_breaches:
            recommendations.append(
                "✅ **Portfolio construction is sound**: No significant synergy risks detected."
            )

        return recommendations

    def _generate_rules(
        self,
        portfolio: Portfolio,
        worst_windows: Dict[str, List[PortfolioWindowMetrics]],
        synergy_report: SynergyRiskReport
    ) -> List[Dict[str, str]]:
        """Generate required gating/allocation rules

        Args:
            portfolio: Portfolio
            worst_windows: Worst windows
            synergy_report: Synergy report

        Returns:
            List of rule dictionaries
        """
        rules = []

        # Correlation-based gating
        if synergy_report.avg_correlation_stress > 0.8:
            rules.append({
                "type": "Correlation Gating",
                "description": "Pause trading when avg correlation exceeds 0.8",
                "implementation": "Monitor rolling 20d correlation; pause if > 0.8"
            })

        # Pair-specific limits
        for s1, s2 in synergy_report.unsafe_combinations[:2]:
            rules.append({
                "type": "Pair Allocation Limit",
                "description": f"Cap combined allocation for {s1} + {s2}",
                "implementation": f"Sum({s1}, {s2}) <= 30% of portfolio"
            })

        # Tail loss circuit breaker
        if synergy_report.simultaneous_tail_losses:
            rules.append({
                "type": "Tail Loss Circuit Breaker",
                "description": "Reduce exposure when 2+ strategies in tail loss",
                "implementation": "Monitor 5d returns; reduce gross exposure by 50% if triggered"
            })

        # Drawdown gating
        worst_mdd = 0.0
        for windows in worst_windows.values():
            for window in windows:
                worst_mdd = max(worst_mdd, window.max_drawdown)

        if worst_mdd > 0.10:
            rules.append({
                "type": "Portfolio Drawdown Gate",
                "description": f"Pause trading when portfolio drawdown exceeds {worst_mdd*100:.0f}%",
                "implementation": "Monitor portfolio MDD; pause if breached"
            })

        return rules


if __name__ == "__main__":
    """Test report generator"""
    print("=== PortfolioRiskReportGenerator Test ===\n")

    generator = PortfolioRiskReportGenerator()

    print("Portfolio risk report generator initialized.")
    print("Use with PortfolioBuilder and SynergyAnalyzer to generate reports.")

    print("\n✓ Test passed")
