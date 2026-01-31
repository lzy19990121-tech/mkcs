"""
多策略最坏情况对比

比较不同策略在最坏窗口下的表现
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from analysis.replay_schema import ReplayOutput, load_replay_outputs
from analysis.window_scanner import WindowScanner, WindowMetrics
from analysis.stability_analysis import StabilityAnalyzer, StabilityReport


@dataclass
class StrategyWorstCaseSummary:
    """策略最坏情况摘要"""
    strategy_id: str
    strategy_name: str
    commit_hash: str
    config_hash: str

    # 最坏窗口统计
    worst_1d: Optional[WindowMetrics] = None
    worst_5d: Optional[WindowMetrics] = None
    worst_20d: Optional[WindowMetrics] = None
    worst_60d: Optional[WindowMetrics] = None
    worst_120d: Optional[WindowMetrics] = None
    worst_250d: Optional[WindowMetrics] = None

    # 稳定性报告
    stability_report: Optional[StabilityReport] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'commit_hash': self.commit_hash,
            'config_hash': self.config_hash,
            'worst_1d': self.worst_1d.to_dict() if self.worst_1d else None,
            'worst_5d': self.worst_5d.to_dict() if self.worst_5d else None,
            'worst_20d': self.worst_20d.to_dict() if self.worst_20d else None,
            'worst_60d': self.worst_60d.to_dict() if self.worst_60d else None,
            'worst_120d': self.worst_120d.to_dict() if self.worst_120d else None,
            'worst_250d': self.worst_250d.to_dict() if self.worst_250d else None,
            'stability_score': self.stability_report.stability_score if self.stability_report else None,
            'return_variance': self.stability_report.return_variance if self.stability_report else None,
            'mdd_consistency': self.stability_report.mdd_consistency if self.stability_report else None,
            'worst_cvar': self.stability_report.worst_cvar if self.stability_report else None
        }


class MultiStrategyComparator:
    """多策略对比器

    比较不同策略的最坏窗口表现
    """

    def __init__(self):
        """初始化对比器"""
        self.window_scanner = WindowScanner(windows=["1d", "5d", "20d", "60d", "120d", "250d"], top_k=1)
        self.stability_analyzer = StabilityAnalyzer()

    def compare_strategies(
        self,
        replays: List[ReplayOutput]
    ) -> List[StrategyWorstCaseSummary]:
        """对比多个策略的最坏情况

        Args:
            replays: 回测输出列表

        Returns:
            策略最坏情况摘要列表
        """
        summaries = []

        for replay in replays:
            summary = self._analyze_replay(replay)
            summaries.append(summary)

        # 按稳定性评分排序
        summaries.sort(
            key=lambda s: s.stability_report.stability_score if s.stability_report else 0,
            reverse=True
        )

        return summaries

    def _analyze_replay(self, replay: ReplayOutput) -> StrategyWorstCaseSummary:
        """分析单个replay

        Args:
            replay: 回测输出

        Returns:
            策略最坏情况摘要
        """
        summary = StrategyWorstCaseSummary(
            strategy_id=replay.strategy_id,
            strategy_name=replay.strategy_name,
            commit_hash=replay.commit_hash,
            config_hash=replay.config_hash
        )

        # 找到各窗口长度的最坏窗口
        for window in ["1d", "5d", "20d", "60d", "120d", "250d"]:
            worst_windows = self.window_scanner.scan_replay(replay, window)
            if worst_windows:
                setattr(summary, f"worst_{window}", worst_windows[0])

        # 生成稳定性报告
        summary.stability_report = self.stability_analyzer.analyze_replay(replay)

        return summary

    def generate_comparison_table(
        self,
        summaries: List[StrategyWorstCaseSummary]
    ) -> pd.DataFrame:
        """生成对比表格

        Args:
            summaries: 策略摘要列表

        Returns:
            对比DataFrame
        """
        data = []

        for summary in summaries:
            row = {
                'strategy_id': summary.strategy_id,
                'strategy_name': summary.strategy_name,
                'stability_score': summary.stability_report.stability_score if summary.stability_report else None,
                'return_variance': summary.stability_report.return_variance if summary.stability_report else None,
                'mdd_consistency': summary.stability_report.mdd_consistency if summary.stability_report else None,
            }

            # 添加各窗口最坏收益
            for window in ["1d", "5d", "20d", "60d", "120d", "250d"]:
                worst = getattr(summary, f"worst_{window}")
                if worst:
                    row[f'{window}_return'] = worst.total_return
                    row[f'{window}_mdd'] = worst.max_drawdown
                    row[f'{window}_cvar'] = worst.cvar_95
                else:
                    row[f'{window}_return'] = None
                    row[f'{window}_mdd'] = None
                    row[f'{window}_cvar'] = None

            data.append(row)

        return pd.DataFrame(data)

    def find_best_for_window(
        self,
        summaries: List[StrategyWorstCaseSummary],
        window: str
    ) -> Optional[StrategyWorstCaseSummary]:
        """找到指定窗口的最优策略

        Args:
            summaries: 策略摘要列表
            window: 窗口长度 (如 "5d", "20d")

        Returns:
            最优策略摘要
        """
        valid = [s for s in summaries if getattr(s, f"worst_{window}") is not None]

        if not valid:
            return None

        # 按最坏窗口收益排序，返回最好的
        return max(valid, key=lambda s: getattr(s, f"worst_{window}").total_return)

    def find_most_stable(self, summaries: List[StrategyWorstCaseSummary]) -> Optional[StrategyWorstCaseSummary]:
        """找到最稳定的策略

        Args:
            summaries: 策略摘要列表

        Returns:
            最稳定策略摘要
        """
        valid = [s for s in summaries if s.stability_report is not None]

        if not valid:
            return None

        return max(valid, key=lambda s: s.stability_report.stability_score)


def compare_all_strategies(run_dir: str) -> List[StrategyWorstCaseSummary]:
    """对比所有策略的最坏情况

    Args:
        run_dir: runs目录路径

    Returns:
        策略摘要列表
    """
    replays = load_replay_outputs(run_dir)
    comparator = MultiStrategyComparator()
    return comparator.compare_strategies(replays)


if __name__ == "__main__":
    """测试代码"""
    print("=== MultiStrategyComparator 测试 ===\n")

    # 对比所有策略
    summaries = compare_all_strategies("runs")

    print(f"对比了 {len(summaries)} 个策略\n")

    # 生成对比表格
    comparator = MultiStrategyComparator()
    df = comparator.generate_comparison_table(summaries)

    print("对比表格:")
    print(df.to_string(index=False))

    # 找到各窗口的最优策略
    print(f"\n各窗口最优策略:")
    for window in ["1d", "5d", "20d"]:
        best = comparator.find_best_for_window(summaries, window)
        if best:
            worst = getattr(best, f"worst_{window}")
            print(f"  {window}: {best.strategy_name} (收益={worst.total_return*100:.2f}%, MDD={worst.max_drawdown*100:.1f}%)")

    # 找到最稳定策略
    most_stable = comparator.find_most_stable(summaries)
    if most_stable:
        print(f"\n最稳定策略: {most_stable.strategy_name} (评分={most_stable.stability_report.stability_score:.1f})")

    print("\n✓ 测试通过")
