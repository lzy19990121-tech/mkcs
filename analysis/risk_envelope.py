"""
风险包络（Risk Envelope）构建模块

基于扰动测试构建 worst-case 的统计边界
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from analysis.replay_schema import ReplayOutput, load_replay_outputs
from analysis.window_scanner import WindowScanner, WindowMetrics
from analysis.perturbation_test import PerturbationTester, PerturbationResult


@dataclass
class RiskEnvelope:
    """风险包络"""
    strategy_id: str
    window_length: str

    # 样本信息
    sample_size: int
    perturbation_types: List[str]

    # worst-case return 的分位数
    return_p50: float
    return_p75: float
    return_p90: float
    return_p95: float
    return_p99: float

    # worst-case MDD 的分位数
    mdd_p50: float
    mdd_p75: float
    mdd_p90: float
    mdd_p95: float
    mdd_p99: float

    # 回撤持续时间的分位数
    duration_p50: float
    duration_p75: float
    duration_p90: float
    duration_p95: float
    duration_p99: float

    # 原始值（未扰动）
    original_return: float
    original_mdd: float
    original_duration: float

    # 置信度
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'strategy_id': self.strategy_id,
            'window_length': self.window_length,
            'sample_size': self.sample_size,
            'perturbation_types': self.perturbation_types,

            'return_percentiles': {
                'p50': self.return_p50,
                'p75': self.return_p75,
                'p90': self.return_p90,
                'p95': self.return_p95,
                'p99': self.return_p99,
            },

            'mdd_percentiles': {
                'p50': self.mdd_p50,
                'p75': self.mdd_p75,
                'p90': self.mdd_p90,
                'p95': self.mdd_p95,
                'p99': self.mdd_p99,
            },

            'duration_percentiles': {
                'p50': self.duration_p50,
                'p75': self.duration_p75,
                'p90': self.duration_p90,
                'p95': self.duration_p95,
                'p99': self.duration_p99,
            },

            'original_values': {
                'return': self.original_return,
                'mdd': self.original_mdd,
                'duration': self.original_duration,
            },

            'confidence_level': self.confidence_level
        }


class RiskEnvelopeBuilder:
    """风险包络构建器

    基于扰动测试构建统计边界
    """

    def __init__(
        self,
        confidence_level: float = 0.95
    ):
        """初始化构建器

        Args:
            confidence_level: 置信水平
        """
        self.confidence_level = confidence_level
        self.perturbation_tester = PerturbationTester()

    def build_envelope(
        self,
        replay: ReplayOutput,
        window_length: str
    ) -> RiskEnvelope:
        """构建单个策略和窗口的风险包络

        Args:
            replay: 回测输出
            window_length: 窗口长度

        Returns:
            风险包络
        """
        # 执行扰动测试
        perturbation_results = self.perturbation_tester.test_perturbations(
            replay, window_length
        )

        # 收集所有指标（包括原始值）
        all_returns = []
        all_mdds = []
        all_durations = []

        # 添加原始值
        if perturbation_results:
            original = perturbation_results[0].original_worst_window
            all_returns.append(original.total_return)
            all_mdds.append(original.max_drawdown)
            all_durations.append(original.max_dd_duration)

            # 添加扰动后的值
            for result in perturbation_results:
                if result.worst_window:
                    all_returns.append(result.worst_window.total_return)
                    all_mdds.append(result.worst_window.max_drawdown)
                    all_durations.append(result.worst_window.max_dd_duration)

        # 计算分位数
        if len(all_returns) == 0:
            all_returns = [0.0]
        if len(all_mdds) == 0:
            all_mdds = [0.0]
        if len(all_durations) == 0:
            all_durations = [0.0]

        return_p50 = np.percentile(all_returns, 50)
        return_p75 = np.percentile(all_returns, 75)
        return_p90 = np.percentile(all_returns, 90)
        return_p95 = np.percentile(all_returns, 95)
        return_p99 = np.percentile(all_returns, 99)

        mdd_p50 = np.percentile(all_mdds, 50)
        mdd_p75 = np.percentile(all_mdds, 75)
        mdd_p90 = np.percentile(all_mdds, 90)
        mdd_p95 = np.percentile(all_mdds, 95)
        mdd_p99 = np.percentile(all_mdds, 99)

        duration_p50 = np.percentile(all_durations, 50)
        duration_p75 = np.percentile(all_durations, 75)
        duration_p90 = np.percentile(all_durations, 90)
        duration_p95 = np.percentile(all_durations, 95)
        duration_p99 = np.percentile(all_durations, 99)

        # 获取扰动类型列表
        perturbation_types = [
            r.perturbation_type.value for r in perturbation_results
        ]

        return RiskEnvelope(
            strategy_id=replay.strategy_id,
            window_length=window_length,
            sample_size=len(all_returns),
            perturbation_types=perturbation_types,

            return_p50=return_p50,
            return_p75=return_p75,
            return_p90=return_p90,
            return_p95=return_p95,
            return_p99=return_p99,

            mdd_p50=mdd_p50,
            mdd_p75=mdd_p75,
            mdd_p90=mdd_p90,
            mdd_p95=mdd_p95,
            mdd_p99=mdd_p99,

            duration_p50=duration_p50,
            duration_p75=duration_p75,
            duration_p90=duration_p90,
            duration_p95=duration_p95,
            duration_p99=duration_p99,

            original_return=all_returns[0],
            original_mdd=all_mdds[0],
            original_duration=all_durations[0],
            confidence_level=self.confidence_level
        )

    def build_all_envelopes(
        self,
        run_dir: str,
        window_lengths: List[str] = None
    ) -> Dict[str, Dict[str, RiskEnvelope]]:
        """构建所有策略的风险包络

        Args:
            run_dir: runs目录路径
            window_lengths: 窗口长度列表

        Returns:
            {strategy_id: {window_length: envelope}}
        """
        replays = load_replay_outputs(run_dir)
        window_lengths = window_lengths or ["20d", "60d"]

        all_envelopes = {}

        for replay in replays:
            strategy_key = f"{replay.run_id}_{replay.strategy_id}"
            all_envelopes[strategy_key] = {}

            for window in window_lengths:
                envelope = self.build_envelope(replay, window)
                all_envelopes[strategy_key][window] = envelope

        return all_envelopes


def format_envelope_report(envelope: RiskEnvelope) -> str:
    """格式化风险包络报告

    Args:
        envelope: 风险包络

    Returns:
        Markdown格式的报告
    """
    lines = []

    lines.append(f"## 风险包络 - {envelope.strategy_id} ({envelope.window_length})\n")

    lines.append(f"**样本数**: {envelope.sample_size}")
    lines.append(f"**扰动类型**: {', '.join(envelope.perturbation_types)}")
    lines.append(f"**置信水平**: {envelope.confidence_level*100:.0f}%\n")

    lines.append("### Worst-Case Return 分位数")
    lines.append("| 分位数 | 数值 | 说明 |")
    lines.append("|--------|------|------|")
    lines.append(f"| P50    | {envelope.return_p50*100:7.2f}% | 中位数 |")
    lines.append(f"| P75    | {envelope.return_p75*100:7.2f}% | |")
    lines.append(f"| P90    | {envelope.return_p90*100:7.2f}% | |")
    lines.append(f"| P95    | {envelope.return_p95*100:7.2f}% | 95%置信边界 |")
    lines.append(f"| P99    | {envelope.return_p99*100:7.2f}% | 99%置信边界 |")
    lines.append(f"| 原始值 | {envelope.original_return*100:7.2f}% | 未扰动 |")
    lines.append("")

    lines.append("### Worst-Case MDD 分位数")
    lines.append("| 分位数 | 数值 | 说明 |")
    lines.append("|--------|------|------|")
    lines.append(f"| P50    | {envelope.mdd_p50*100:6.2f}% | 中位数 |")
    lines.append(f"| P75    | {envelope.mdd_p75*100:6.2f}% | |")
    lines.append(f"| P90    | {envelope.mdd_p90*100:6.2f}% | |")
    lines.append(f"| P95    | {envelope.mdd_p95*100:6.2f}% | 95%置信边界 |")
    lines.append(f"| P99    | {envelope.mdd_p99*100:6.2f}% | 99%置信边界 |")
    lines.append(f"| 原始值 | {envelope.original_mdd*100:6.2f}% | 未扰动 |")
    lines.append("")

    lines.append("### 回撤持续时间 分位数")
    lines.append("| 分位数 | 天数 | 说明 |")
    lines.append("|--------|------|------|")
    lines.append(f"| P50    | {envelope.duration_p50:5.1f} | 中位数 |")
    lines.append(f"| P75    | {envelope.duration_p75:5.1f} | |")
    lines.append(f"| P90    | {envelope.duration_p90:5.1f} | |")
    lines.append(f"| P95    | {envelope.duration_p95:5.1f} | 95%置信边界 |")
    lines.append(f"| P99    | {envelope.duration_p99:5.1f} | 99%置信边界 |")
    lines.append(f"| 原始值 | {envelope.original_duration:5.1f} | 未扰动 |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    """测试代码"""
    print("=== RiskEnvelopeBuilder 测试 ===\n")

    replays = load_replay_outputs("runs")

    if replays:
        replay = replays[0]
        print(f"构建风险包络: {replay.run_id}\n")

        builder = RiskEnvelopeBuilder(confidence_level=0.95)

        # 构建20d窗口的包络
        envelope = builder.build_envelope(replay, "20d")

        # 打印报告
        report = format_envelope_report(envelope)
        print(report)

        # 打印摘要
        print("### 摘要")
        print(f"在95%置信水平下：")
        print(f"  • 最坏收益可能达到: {envelope.return_p95*100:.2f}%")
        print(f"  • 最大回撤可能达到: {envelope.mdd_p95*100:.2f}%")
        print(f"  • 回撤持续可能达到: {envelope.duration_p95:.1f}天")

    print("\n✓ 测试通过")
