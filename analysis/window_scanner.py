"""
时间窗口扫描器

实现滚动窗口扫描，找到最坏窗口和Top-K最坏窗口
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from decimal import Decimal
from collections import defaultdict, deque
from pathlib import Path
import warnings

from analysis.replay_schema import ReplayOutput, load_replay_outputs


@dataclass
class WindowMetrics:
    """窗口风险指标"""
    # 窗口信息
    window_id: str
    strategy_id: str
    window_length: str  # e.g., "5d", "20d"
    start_date: datetime
    end_date: datetime

    # 收益指标
    total_return: float
    total_pnl: float
    final_equity: float

    # 风险指标
    max_drawdown: float  # 最大回撤
    max_dd_duration: int  # 最大回撤持续天数
    max_dd_recovery_time: int  # 回撤恢复时间（天数）
    ulcer_index: float  # 溃疡指数

    # 波动指标
    downside_deviation: float  # 下行波动
    cvar_95: float  # 95% CVaR
    tail_mean: float  # 尾部均值（最差5%日均值）

    # 尖刺风险
    max_single_loss: float  # 最大单步亏损
    max_consecutive_losses: int  # 最长连续亏损次数
    loss_distribution: Dict[int, int]  # 亏损长度分布

    # 回撤形态
    drawdown_pattern: str  # 回撤形态（sharp/slow/recovery）
    is_censored: bool  # 是否未恢复

    # 元数据
    run_id: str
    commit_hash: str
    config_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'window_id': self.window_id,
            'strategy_id': self.strategy_id,
            'window_length': self.window_length,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_return': self.total_return,
            'total_pnl': self.total_pnl,
            'final_equity': self.final_equity,
            'max_drawdown': self.max_drawdown,
            'max_dd_duration': self.max_dd_duration,
            'max_dd_recovery_time': self.max_dd_recovery_time,
            'ulcer_index': self.ulcer_index,
            'downside_deviation': self.downside_deviation,
            'cvar_95': self.cvar_95,
            'tail_mean': self.tail_mean,
            'max_single_loss': self.max_single_loss,
            'max_consecutive_losses': self.max_consecutive_losses,
            'loss_distribution': self.loss_distribution,
            'drawdown_pattern': self.drawdown_pattern,
            'is_censored': self.is_censored,
            'run_id': self.run_id,
            'commit_hash': self.commit_hash,
            'config_hash': self.config_hash
        }


class WindowScanner:
    """时间窗口扫描器

    滚动扫描所有连续窗口，找到最坏窗口
    """

    def __init__(
        self,
        windows: List[str] = None,
        top_k: int = 5
    ):
        """初始化扫描器

        Args:
            windows: 窗口长度列表，如 ["1d", "5d", "20d", "60d"]
            top_k: 保留top-k个最坏窗口
        """
        self.windows = windows or ["1d", "5d", "20d", "60d", "120d", "250d"]
        self.top_k = top_k

    def parse_window_length(self, window_str: str) -> int:
        """解析窗口长度字符串

        Args:
            window_str: 窗口长度字符串，如 "5d", "20d"

        Returns:
            窗口天数
        """
        if window_str.endswith('d'):
            return int(window_str[:-1])
        elif window_str.endswith('m'):
            return int(window_str[:-1]) / 1440  # 分钟转天
        elif window_str.endswith('h'):
            return int(window_str[:-1]) / 24
        elif window_str.endswith('w'):
            return int(window_str[:-1]) * 5
        else:
            raise ValueError(f"Unknown window format: {window_str}")

    def scan_replay(
        self,
        replay: ReplayOutput,
        window_length: str
    ) -> List[WindowMetrics]:
        """扫描单个replay的所有窗口

        Args:
            replay: 回测输出
            window_length: 窗口长度（如 "5d"）

        Returns:
            所有窗口的指标列表
        """
        window_days = self.parse_window_length(window_length)
        df = replay.to_dataframe()

        if df.empty:
            return []

        df = df.sort_values('timestamp').reset_index(drop=True)

        metrics_list = []

        # 滚动窗口
        for i in range(len(df) - window_days + 1):
            window_df = df.iloc[i:i + window_days].copy()

            # 计算指标
            metrics = self._calculate_window_metrics(
                window_df,
                replay,
                window_length,
                i
            )
            metrics_list.append(metrics)

        # 按total_return排序，找到最坏窗口
        metrics_list.sort(key=lambda m: m.total_return)

        return metrics_list

    def find_worst_windows(
        self,
        replay: ReplayOutput,
        top_k: int = None
    ) -> List[WindowMetrics]:
        """找到最坏窗口

        Args:
            replay: 回测输出
            top_k: 保留top-k个最坏窗口

        Returns:
            最坏窗口列表（按收益排序）
        """
        k = top_k or self.top_k

        all_metrics = []

        for window in self.windows:
            window_metrics = self.scan_replay(replay, window)
            all_metrics.extend(window_metrics)

        # 按total_return排序，取最坏窗口
        all_metrics.sort(key=lambda m: m.total_return)
        return all_metrics[:k]

    def _calculate_window_metrics(
        self,
        df: pd.DataFrame,
        replay: ReplayOutput,
        window_length: str,
        start_idx: int
    ) -> WindowMetrics:
        """计算窗口指标

        Args:
            df: 窗口数据
            replay: 回测输出
            window_length: 窗口长度
            start_idx: 起始索引

        Returns:
            窗口指标
        """
        # 基本信息
        start_date = df['timestamp'].iloc[0]
        end_date = df['timestamp'].iloc[-1]

        # 收益指标
        total_pnl = df['step_pnl'].sum()
        initial_equity = float(replay.initial_cash)
        total_return = total_pnl / initial_equity
        final_equity = initial_equity + total_pnl

        # 计算最大回撤
        equity = df['equity'].values if 'equity' in df.columns else \
                  (replay.initial_cash + df['step_pnl'].cumsum().values)

        max_dd, max_dd_duration, max_dd_recovery_time = self._calculate_drawdown_metrics(
            equity, start_date, end_date
        )

        # 下行波动
        downside_dev = self._calculate_downside_deviation(df)
        cvar_95, tail_mean = self._calculate_cvar(df)

        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index(equity)

        # 尖刺风险
        max_single_loss, max_consecutive_losses, loss_dist = self._calculate_spike_risk(df)

        # 回撤形态
        drawdown_pattern, is_censored = self._classify_drawdown_pattern(
            equity, max_dd, max_dd_duration, max_dd_recovery_time
        )

        # 窗口ID
        window_id = f"{replay.run_id}_{window_length}_{start_idx}"

        return WindowMetrics(
            window_id=window_id,
            strategy_id=replay.strategy_id,
            window_length=window_length,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            total_pnl=float(total_pnl),
            final_equity=final_equity,
            max_drawdown=max_dd,
            max_dd_duration=max_dd_duration,
            max_dd_recovery_time=max_dd_recovery_time,
            ulcer_index=ulcer_index,
            downside_deviation=downside_dev,
            cvar_95=cvar_95,
            tail_mean=tail_mean,
            max_single_loss=max_single_loss,
            max_consecutive_losses=max_consecutive_losses,
            loss_distribution=loss_dist,
            drawdown_pattern=drawdown_pattern,
            is_censored=is_censored,
            run_id=replay.run_id,
            commit_hash=replay.commit_hash,
            config_hash=replay.config_hash
        )

    def _calculate_drawdown_metrics(
        self,
        equity: np.ndarray,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[float, int, int]:
        """计算回撤指标

        Returns:
            (最大回撤, 最大回撤持续天数, 回撤恢复时间)
        """
        if len(equity) == 0:
            return 0.0, 0, 0

        # 计算累计最大值
        cummax = np.maximum.accumulate(equity)
        drawdowns = cummax - equity

        # 最大回撤
        max_dd = drawdowns.max() / equity[0] if equity[0] > 0 else 0.0

        # 最大回撤持续时间
        dd_end_idx = np.argmax(drawdowns)
        dd_start_idx = np.argmax(equity[:dd_end_idx + 1]) if dd_end_idx > 0 else 0

        max_dd_duration = int(dd_end_idx - dd_start_idx)

        # 回撤恢复时间
        if dd_end_idx < len(equity) - 1:
            recovery_idx = np.where(equity[dd_end_idx + 1:] >= cummax[dd_end_idx])[0]
            if len(recovery_idx) > 0:
                recovery_time = int(recovery_idx[0] - dd_end_idx)
            else:
                recovery_time = int(len(equity) - 1 - dd_end_idx)
        else:
            recovery_time = -1  # 未恢复

        return max_dd, max_dd_duration, recovery_time

    def _calculate_downside_deviation(self, df: pd.DataFrame) -> float:
        """计算下行波动"""
        if 'step_pnl' not in df.columns:
            return 0.0

        returns = df['step_pnl']
        mean_return = returns.mean()

        # 只计算负收益
        negative_returns = returns[returns < mean_return]

        if len(negative_returns) == 0:
            return 0.0

        variance = ((negative_returns - mean_return) ** 2).sum() / len(negative_returns)
        return variance ** 0.5

    def _calculate_cvar(self, df: pd.DataFrame) -> Tuple[float, float]:
        """计算CVaR和尾部均值

        Returns:
            (cvar_95, tail_mean)
        """
        if 'step_pnl' not in df.columns:
            return 0.0, 0.0

        returns = df['step_pnl'].values
        if len(returns) == 0:
            return 0.0, 0.0

        # 95% CVaR
        var_95 = np.percentile(returns, 5)
        mean_return = returns.mean()

        if var_95 < 0:
            cvar_95 = abs(var_95) / mean_return if mean_return != 0 else 0.0
        else:
            cvar_95 = 0.0

        # 尾部均值（最差5%）
        tail_returns = np.partition(returns, max(0, len(returns) // 20))[:max(1, len(returns) // 20)]
        tail_mean = tail_returns.mean() if len(tail_returns) > 0 else 0.0

        return cvar_95, tail_mean

    def _calculate_ulcer_index(self, equity: np.ndarray) -> float:
        """计算Ulcer Index（回撤面积）"""
        if len(equity) == 0:
            return 0.0

        cummax = np.maximum.accumulate(equity)
        drawdowns = cummax - equity

        # 归一化到初始权益
        if cummax[0] > 0:
            normalized_dd = drawdowns / cummax[0]
        else:
            normalized_dd = drawdowns

        # 计算RMS（平方根均值）
        ulcer_index = np.sqrt((normalized_dd ** 2).mean())

        return ulcer_index * 100  # 转为百分比

    def _calculate_spike_risk(
        self,
        df: pd.DataFrame
    ) -> Tuple[float, int, Dict[int, int]]:
        """计算尖刺风险

        Returns:
            (最大单步亏损, 最长连续亏损次数, 亏损长度分布)
        """
        if 'step_pnl' not in df.columns:
            return 0.0, 0, {}

        returns = df['step_pnl'].values

        # 最大单步亏损
        max_loss = returns.min() if len(returns) > 0 else 0.0

        # 连续亏损
        is_loss = returns < 0
        max_consecutive = 0
        current = 0

        loss_dist = defaultdict(int)

        for is_l in is_loss:
            if is_l:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                if current > 0:
                    loss_dist[current] += 1
                current = 0

        return float(max_loss), max_consecutive, dict(loss_dist)

    def _classify_drawdown_pattern(
        self,
        equity: np.ndarray,
        max_dd: float,
        dd_duration: int,
        recovery_time: int
    ) -> Tuple[str, bool]:
        """分类回撤形态

        Returns:
            (形态类型, 是否未恢复)
        """
        is_censored = recovery_time < 0

        if is_censored:
            return "unrecovered", True
        elif dd_duration <= 2:
            return "sharp", False
        elif recovery_time <= 5:
            return "fast_recovery", False
        elif recovery_time <= 20:
            return "slow_recovery", False
        else:
            return "very_slow_recovery", False


def scan_all_replays(
    run_dir: str,
    window_lengths: List[str] = None,
    top_k: int = 5
) -> Dict[str, List[WindowMetrics]]:
    """扫描所有replay的所有窗口

    Args:
        run_dir: runs目录路径
        window_lengths: 窗口长度列表
        top_k: 每个窗口保留top-k个最坏窗口

    Returns:
        {window_length: [WindowMetrics]}
    """
    scanner = WindowScanner(windows=window_lengths, top_k=top_k)

    all_results = defaultdict(list)

    # 加载所有replay
    replays = load_replay_outputs(run_dir)

    for replay in replays:
        for window in scanner.windows:
            print(f"扫描: {replay.run_id} - {window}")
            window_metrics = scanner.scan_replay(replay, window)

            # 保留top-k最坏窗口
            worst_k = sorted(window_metrics, key=lambda m: m.total_return)[:top_k]
            all_results[window].extend(worst_k)

    return dict(all_results)


if __name__ == "__main__":
    """测试代码"""
    print("=== WindowScanner 测试 ===\n")

    # 扫描测试
    results = scan_all_replays("runs", window_lengths=["5d", "20d"], top_k=3)

    for window, metrics in results.items():
        print(f"\n窗口 {window}:")
        for m in metrics[:3]:  # 显示前3个
            print(f"  {m.window_id}: {m.total_return*100:.2f}% "
                  f"MDD={m.max_drawdown*100:.1f}% "
                  f"回撤={m.drawdown_pattern}")


