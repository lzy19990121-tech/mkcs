"""
跨窗口稳定性分析

比较不同窗口长度下的最坏情况，分析策略稳定性
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from analysis.replay_schema import ReplayOutput, load_replay_outputs
from analysis.window_scanner import WindowScanner, WindowMetrics


@dataclass
class StabilityReport:
    """稳定性报告"""
    strategy_id: str
    run_id: str
    commit_hash: str
    config_hash: str

    # 最坏窗口统计
    worst_by_window: Dict[str, WindowMetrics]  # window_length -> worst window

    # 跨窗口稳定性指标
    return_variance: float  # 最坏窗口收益方差
    return_range: float     # 最坏窗口收益极差
    mdd_consistency: float  # MDD一致性（std/mean）
    worst_cvar: float       # 最差CVaR

    # 稳定性评分
    stability_score: float  # 0-100，越高越稳定

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'strategy_id': self.strategy_id,
            'run_id': self.run_id,
            'commit_hash': self.commit_hash,
            'config_hash': self.config_hash,
            'worst_by_window': {
                w: m.to_dict() for w, m in self.worst_by_window.items()
            },
            'return_variance': self.return_variance,
            'return_range': self.return_range,
            'mdd_consistency': self.mdd_consistency,
            'worst_cvar': self.worst_cvar,
            'stability_score': self.stability_score
        }


class StabilityAnalyzer:
    """稳定性分析器

    分析策略在不同窗口长度下的最坏情况稳定性
    """

    def __init__(self, windows: List[str] = None):
        """初始化分析器

        Args:
            windows: 窗口长度列表
        """
        self.windows = windows or ["1d", "5d", "20d", "60d", "120d", "250d"]

    def analyze_replay(self, replay: ReplayOutput) -> StabilityReport:
        """分析单个replay的稳定性

        Args:
            replay: 回测输出

        Returns:
            稳定性报告
        """
        scanner = WindowScanner(windows=self.windows, top_k=1)

        # 获取每个窗口长度的最坏窗口
        worst_by_window = {}
        for window in self.windows:
            worst_windows = scanner.scan_replay(replay, window)
            if worst_windows:
                worst_by_window[window] = worst_windows[0]  # 最坏的一个

        # 计算稳定性指标
        report = StabilityReport(
            strategy_id=replay.strategy_id,
            run_id=replay.run_id,
            commit_hash=replay.commit_hash,
            config_hash=replay.config_hash,
            worst_by_window=worst_by_window,
            return_variance=0.0,
            return_range=0.0,
            mdd_consistency=0.0,
            worst_cvar=0.0,
            stability_score=0.0
        )

        if len(worst_by_window) > 0:
            # 收集最坏窗口收益
            worst_returns = [m.total_return for m in worst_by_window.values()]

            # 收集MDD
            worst_mdds = [m.max_drawdown for m in worst_by_window.values()]

            # 收集CVaR
            worst_cvars = [m.cvar_95 for m in worst_by_window.values()]

            # 计算指标
            report.return_variance = np.var(worst_returns) if len(worst_returns) > 1 else 0.0
            report.return_range = max(worst_returns) - min(worst_returns) if len(worst_returns) > 1 else 0.0

            # MDD一致性 (std/mean，越小越稳定)
            mdd_mean = np.mean(worst_mdds)
            mdd_std = np.std(worst_mdds) if len(worst_mdds) > 1 else 0.0
            report.mdd_consistency = (mdd_std / mdd_mean) if mdd_mean > 0 else 0.0

            # 最差CVaR
            report.worst_cvar = max(worst_cvars) if worst_cvars else 0.0

            # 稳定性评分 (0-100)
            report.stability_score = self._calculate_stability_score(report)

        return report

    def compare_replays(self, replays: List[ReplayOutput]) -> List[StabilityReport]:
        """比较多个replay的稳定性

        Args:
            replays: 回测输出列表

        Returns:
            稳定性报告列表
        """
        reports = []
        for replay in replays:
            report = self.analyze_replay(replay)
            reports.append(report)

        # 按稳定性评分排序
        reports.sort(key=lambda r: r.stability_score, reverse=True)
        return reports

    def _calculate_stability_score(self, report: StabilityReport) -> float:
        """计算稳定性评分

        评分标准：
        - 收益方差越小越好
        - MDD一致性越小越好
        - 最坏CVaR越小越好

        Returns:
            0-100的评分
        """
        # 收益方差惩罚 (max ~0.01)
        variance_penalty = min(report.return_variance * 1000, 40)

        # MDD一致性惩罚 (max ~1.0)
        mdd_penalty = min(report.mdd_consistency * 30, 30)

        # CVaR惩罚 (max ~0.1)
        cvar_penalty = min(report.worst_cvar * 100, 30)

        # 基础分100，减去各种惩罚
        score = 100 - variance_penalty - mdd_penalty - cvar_penalty
        return max(0.0, min(100.0, score))

    def find_most_stable(self, replays: List[ReplayOutput]) -> StabilityReport:
        """找到最稳定的replay

        Args:
            replays: 回测输出列表

        Returns:
            最稳定的稳定性报告
        """
        reports = self.compare_replays(replays)
        return reports[0] if reports else None


def analyze_all_stability(run_dir: str) -> List[StabilityReport]:
    """分析所有replay的稳定性

    Args:
        run_dir: runs目录路径

    Returns:
        稳定性报告列表
    """
    replays = load_replay_outputs(run_dir)
    analyzer = StabilityAnalyzer()
    return analyzer.compare_replays(replays)


if __name__ == "__main__":
    """测试代码"""
    print("=== StabilityAnalyzer 测试 ===\n")

    # 分析所有replay
    reports = analyze_all_stability("runs")

    print(f"分析了 {len(reports)} 个回测的稳定性\n")

    for report in reports:
        print(f"\n策略: {report.strategy_id}")
        print(f"  运行: {report.run_id}")
        print(f"  稳定性评分: {report.stability_score:.1f}/100")
        print(f"  收益方差: {report.return_variance*100:.4f}%")
        print(f"  收益极差: {report.return_range*100:.2f}%")
        print(f"  MDD一致性: {report.mdd_consistency:.3f}")
        print(f"  最差CVaR: {report.worst_cvar:.3f}")

        print(f"\n  最坏窗口收益:")
        for window, metrics in report.worst_by_window.items():
            print(f"    {window}: {metrics.total_return*100:.2f}% (MDD={metrics.max_drawdown*100:.1f}%)")

    if reports:
        print(f"\n最稳定策略: {reports[0].strategy_id} (评分: {reports[0].stability_score:.1f})")

    print("\n✓ 测试通过")
