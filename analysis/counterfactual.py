"""
反事实分析模块 (Counterfactual Analysis)

用于分析"如果采用不同策略会有什么结果"：
- 比较不同配置下的交易表现
- 分析风控拒绝的交易如果成交会如何
- 生成对照实验报告
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


class CounterfactualType(Enum):
    """反事实分析类型"""
    REJECTED_TRADES = "rejected_trades"  # 分析被拒绝的交易
    ALTERNATIVE_CONFIG = "alternative_config"  # 使用不同配置
    WHAT_IF_SCENARIO = "what_if_scenario"  # 假设场景
    SMOOTHING_COMPARISON = "smoothing_comparison"  # 权重平滑对比


@dataclass
class CounterfactualResult:
    """反事实分析结果"""
    type: CounterfactualType
    description: str
    baseline_return: float
    counterfactual_return: float
    return_delta: float
    affected_trades: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "description": self.description,
            "baseline_return": self.baseline_return,
            "counterfactual_return": self.counterfactual_return,
            "return_delta": self.return_delta,
            "affected_trades": self.affected_trades,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class RejectedTradeAnalysis:
    """被拒绝交易分析"""
    original_return: float
    if_accepted_return: float
    rejected_trades: List[Dict[str, Any]]
    would_be_profitable: int
    would_be_loss: int
    avg_missed_profit: float
    avg_avoided_loss: float


class CounterfactualAnalyzer:
    """反事实分析器"""

    def __init__(self, storage_path: Optional[str] = None):
        """初始化分析器

        Args:
            storage_path: 结果存储路径
        """
        self.storage_path = Path(storage_path) if storage_path else Path("outputs/counterfactual")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.results: List[CounterfactualResult] = []

    def analyze_rejected_trades(
        self,
        backtest_result: Dict[str, Any],
        price_data: Dict[str, List[Tuple[datetime, float]]]
    ) -> RejectedTradeAnalysis:
        """分析如果被拒绝的交易被接受会怎样

        Args:
            backtest_result: 回测结果
            price_data: 价格数据 {symbol: [(timestamp, price), ...]}

        Returns:
            RejectedTradeAnalysis
        """
        rejected_trades = backtest_result.get("risk_rejects", [])
        original_return = backtest_result.get("summary", {}).get("total_return", 0.0)

        if not rejected_trades:
            return RejectedTradeAnalysis(
                original_return=original_return,
                if_accepted_return=original_return,
                rejected_trades=[],
                would_be_profitable=0,
                would_be_loss=0,
                avg_missed_profit=0.0,
                avg_avoided_loss=0.0
            )

        # 模拟被拒绝交易如果被接受的结果
        simulated_pnl = 0.0
        profitable = 0
        loss = 0
        missed_profits = []
        avoided_losses = []

        for reject in rejected_trades:
            symbol = reject.get("symbol")
            action = reject.get("action")  # BUY or SELL
            timestamp = reject.get("timestamp")
            confidence = reject.get("confidence", 0.5)

            # 获取价格数据
            if symbol not in price_data:
                continue

            # 简化处理：使用后续价格变化计算盈亏
            # 实际应该有更完整的模拟
            pnl = self._estimate_trade_pnl(reject, price_data[symbol])
            simulated_pnl += pnl

            if pnl > 0:
                profitable += 1
                missed_profits.append(pnl)
            else:
                loss += 1
                avoided_losses.append(abs(pnl))

        # 计算假设收益率
        initial_equity = backtest_result.get("summary", {}).get("initial_equity", 100000)
        if_accepted_return = original_return + (simulated_pnl / initial_equity)

        analysis = RejectedTradeAnalysis(
            original_return=original_return,
            if_accepted_return=if_accepted_return,
            rejected_trades=rejected_trades,
            would_be_profitable=profitable,
            would_be_loss=loss,
            avg_missed_profit=sum(missed_profits) / len(missed_profits) if missed_profits else 0.0,
            avg_avoided_loss=sum(avoided_losses) / len(avoided_losses) if avoided_losses else 0.0
        )

        # 记录结果
        result = CounterfactualResult(
            type=CounterfactualType.REJECTED_TRADES,
            description=f"分析 {len(rejected_trades)} 笔被拒绝的交易如果被接受的结果",
            baseline_return=original_return,
            counterfactual_return=if_accepted_return,
            return_delta=if_accepted_return - original_return,
            affected_trades=len(rejected_trades),
            metadata={
                "would_be_profitable": profitable,
                "would_be_loss": loss,
                "avg_missed_profit": analysis.avg_missed_profit,
                "avg_avoided_loss": analysis.avg_avoided_loss
            }
        )
        self.results.append(result)

        return analysis

    def _estimate_trade_pnl(
        self,
        reject: Dict[str, Any],
        price_data: List[Tuple[datetime, float]]
    ) -> float:
        """估算被拒绝交易的盈亏

        Args:
            reject: 拒绝记录
            price_data: 价格数据

        Returns:
            估算的盈亏
        """
        # 简化处理：随机生成一个小盈亏用于演示
        # 实际应该根据价格数据计算
        import random
        return random.uniform(-100, 100)

    def compare_weight_smoothing(
        self,
        no_smoothing_result: Dict[str, Any],
        with_smoothing_result: Dict[str, Any],
        lambda_value: float
    ) -> CounterfactualResult:
        """对比权重平滑的效果

        Args:
            no_smoothing_result: 无平滑的结果
            with_smoothing_result: 有平滑的结果
            lambda_value: 平滑参数

        Returns:
            CounterfactualResult
        """
        no_smooth_return = no_smoothing_result.get("summary", {}).get("total_return", 0.0)
        with_smooth_return = with_smoothing_result.get("summary", {}).get("total_return", 0.0)

        # 计算权重波动
        no_smooth_jitter = self._calculate_weight_jitter(no_smoothing_result)
        with_smooth_jitter = self._calculate_weight_jitter(with_smoothing_result)

        result = CounterfactualResult(
            type=CounterfactualType.SMOOTHING_COMPARISON,
            description=f"权重平滑效果对比 (λ={lambda_value})",
            baseline_return=no_smooth_return,
            counterfactual_return=with_smooth_return,
            return_delta=with_smooth_return - no_smooth_return,
            affected_trades=0,
            metadata={
                "lambda": lambda_value,
                "no_smoothing_jitter": no_smooth_jitter,
                "with_smoothing_jitter": with_smooth_jitter,
                "jitter_reduction": (no_smooth_jitter - with_smooth_jitter) / no_smooth_jitter if no_smooth_jitter > 0 else 0
            }
        )

        self.results.append(result)
        return result

    def _calculate_weight_jitter(self, result: Dict[str, Any]) -> float:
        """计算权重波动"""
        weight_history = result.get("weight_history", [])
        if len(weight_history) < 2:
            return 0.0

        import numpy as np
        jitters = []
        for i in range(1, len(weight_history)):
            prev = weight_history[i-1].get("weights", {})
            curr = weight_history[i].get("weights", {})

            # 计算权重变化
            changes = []
            for key in prev:
                if key in curr:
                    changes.append(abs(curr[key] - prev[key]))

            if changes:
                jitters.append(np.mean(changes))

        return np.mean(jitters) if jitters else 0.0

    def what_if_scenario(
        self,
        backtest_result: Dict[str, Any],
        scenario_config: Dict[str, Any]
    ) -> CounterfactualResult:
        """分析假设场景

        Args:
            backtest_result: 原始回测结果
            scenario_config: 场景配置
                - disable_strategies: 禁用的策略列表
                - adjust_risk_limits: 调整风险限制
                - change_commission: 修改手续费率

        Returns:
            CounterfactualResult
        """
        baseline_return = backtest_result.get("summary", {}).get("total_return", 0.0)
        description = scenario_config.get("description", "假设场景")

        # 简化处理：根据配置调整收益
        adjustment = scenario_config.get("return_adjustment", 0.0)
        scenario_return = baseline_return + adjustment

        result = CounterfactualResult(
            type=CounterfactualType.WHAT_IF_SCENARIO,
            description=description,
            baseline_return=baseline_return,
            counterfactual_return=scenario_return,
            return_delta=adjustment,
            affected_trades=scenario_config.get("affected_trades", 0),
            metadata=scenario_config
        )

        self.results.append(result)
        return result

    def generate_comparison_report(
        self,
        group_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """生成对照实验报告

        Args:
            group_results: 各组结果 {
                "Group A": {...},
                "Group B": {...},
                ...
            }

        Returns:
            报告内容 (Markdown)
        """
        lines = []
        lines.append("# 反事实分析对照实验报告")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().isoformat()}")
        lines.append("")

        # 概览表格
        lines.append("## 概览")
        lines.append("")
        lines.append("| 组别 | 收益率 | 夏普比率 | 最大回撤 | 交易次数 |")
        lines.append("|------|--------|----------|----------|----------|")

        for group_name, result in group_results.items():
            summary = result.get("summary", {})
            metrics = result.get("metrics", {})
            lines.append(
                f"| {group_name} | "
                f"{summary.get('total_return', 0)*100:.2f}% | "
                f"{metrics.get('sharpe_ratio', 0):.2f} | "
                f"{metrics.get('max_drawdown', 0)*100:.2f}% | "
                f"{summary.get('total_trades', 0)} |"
            )

        lines.append("")

        # 详细分析
        lines.append("## 详细分析")
        lines.append("")

        for group_name, result in group_results.items():
            lines.append(f"### {group_name}")
            lines.append("")
            summary = result.get("summary", {})

            lines.append(f"- **总收益率**: {summary.get('total_return', 0)*100:.2f}%")
            lines.append(f"- **交易次数**: {summary.get('total_trades', 0)}")
            lines.append(f"- **胜率**: {summary.get('win_rate', 0)*100:.1f}%")
            lines.append(f"- **最终权益**: ${summary.get('final_equity', 0):,.2f}")
            lines.append("")

        # 反事实结果
        lines.append("## 反事实分析结果")
        lines.append("")

        for result in self.results:
            lines.append(f"### {result.description}")
            lines.append("")
            lines.append(f"- **类型**: {result.type.value}")
            lines.append(f"- **基准收益**: {result.baseline_return*100:.2f}%")
            lines.append(f"- **反事实收益**: {result.counterfactual_return*100:.2f}%")
            lines.append(f"- **收益差**: {result.return_delta*100:+.2f}%")
            lines.append(f"- **影响交易数**: {result.affected_trades}")
            lines.append("")

            if result.metadata:
                lines.append("**详细数据**:")
                for key, value in result.metadata.items():
                    if isinstance(value, float):
                        lines.append(f"- {key}: {value:.4f}")
                    else:
                        lines.append(f"- {key}: {value}")
                lines.append("")

        # 结论
        lines.append("## 结论")
        lines.append("")

        # 找出最佳组别
        best_group = max(group_results.items(), key=lambda x: x[1].get("summary", {}).get("total_return", 0))
        lines.append(f"- **最佳收益组别**: {best_group[0]} ({best_group[1].get('summary', {}).get('total_return', 0)*100:.2f}%)")

        # 分析权重平滑效果
        smoothing_results = [r for r in self.results if r.type == CounterfactualType.SMOOTHING_COMPARISON]
        if smoothing_results:
            latest = smoothing_results[-1]
            jitter_reduction = latest.metadata.get("jitter_reduction", 0) * 100
            lines.append(f"- **权重平滑效果**: 波动降低 {jitter_reduction:.1f}%")
            lines.append(f"- **收益影响**: {latest.return_delta*100:+.2f}%")

        lines.append("")

        return "\n".join(lines)

    def save_report(self, report: str, filename: str = "counterfactual_report.md") -> Path:
        """保存报告

        Args:
            report: 报告内容
            filename: 文件名

        Returns:
            文件路径
        """
        output_path = self.storage_path / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return output_path

    def export_results_json(self, filename: str = "counterfactual_results.json") -> Path:
        """导出结果为 JSON

        Args:
            filename: 文件名

        Returns:
            文件路径
        """
        output_path = self.storage_path / filename
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return output_path


# 便捷函数
def analyze_rejected_trades_impact(
    backtest_result: Dict[str, Any],
    price_data: Dict[str, List[Tuple[datetime, float]]]
) -> Dict[str, Any]:
    """快速分析被拒绝交易的影响

    Args:
        backtest_result: 回测结果
        price_data: 价格数据

    Returns:
        分析结果字典
    """
    analyzer = CounterfactualAnalyzer()
    analysis = analyzer.analyze_rejected_trades(backtest_result, price_data)

    return {
        "original_return": analysis.original_return,
        "if_accepted_return": analysis.if_accepted_return,
        "return_delta": analysis.if_accepted_return - analysis.original_return,
        "rejected_count": len(analysis.rejected_trades),
        "would_be_profitable": analysis.would_be_profitable,
        "would_be_loss": analysis.would_be_loss,
        "avg_missed_profit": analysis.avg_missed_profit,
        "avg_avoided_loss": analysis.avg_avoided_loss
    }
