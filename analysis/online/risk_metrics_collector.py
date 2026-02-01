"""
SPL-7a-A: 运行态风险指标采集器

实时采集风险指标，与 SPL-4/5/6 口径完全一致。
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.online.risk_signal_schema import (
    RiskSignal,
    PortfolioRiskSignal,
    RollingReturnMetrics,
    DrawdownMetrics,
    SpikeMetrics,
    StabilityMetrics,
    RegimeFeatures,
    GatingEvent,
    AllocatorEvent
)
from analysis.regime_features import RegimeFeatures as RegimeFeaturesCalculator
from analysis.window_scanner import WindowScanner


@dataclass
class MetricsBuffer:
    """指标缓冲区"""
    returns: deque  # 收益序列
    prices: deque   # 价格序列
    timestamps: deque  # 时间戳序列

    def __init__(self, max_length: int = 252):  # 默认保留 1 年交易日
        self.returns = deque(maxlen=max_length)
        self.prices = deque(maxlen=max_length)
        self.timestamps = deque(maxlen=max_length)


class RiskMetricsCollector:
    """风险指标采集器

    负责采集、计算、聚合所有运行态风险指标。
    """

    def __init__(
        self,
        strategy_id: str,
        buffer_size: int = 252,
        update_frequency: str = "1d"
    ):
        """初始化采集器

        Args:
            strategy_id: 策略 ID
            buffer_size: 缓冲区大小（交易日）
            update_frequency: 更新频率（1d, 1h, 5m 等）
        """
        self.strategy_id = strategy_id
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency

        # 指标缓冲区
        self.buffer = MetricsBuffer(max_length=buffer_size)

        # 当前状态
        self.current_price: Optional[float] = None
        self.initial_price: Optional[float] = None
        self.current_position: float = 0.0

        # 事件缓冲
        self.recent_gating_events: List[GatingEvent] = []
        self.recent_allocator_events: List[AllocatorEvent] = []

        # 辅助工具
        self.window_scanner = WindowScanner()

    def update(
        self,
        price: float,
        timestamp: datetime,
        position: float = 0.0,
        regime_features: Optional[Dict[str, Any]] = None
    ) -> RiskSignal:
        """更新指标并生成风险信号

        Args:
            price: 当前价格
            timestamp: 时间戳
            position: 当前持仓
            regime_features: 市场状态特征（可选）

        Returns:
            RiskSignal
        """
        # 更新缓冲区
        self._update_buffer(price, timestamp, position)

        # 计算各项指标
        rolling_returns = self._calculate_rolling_returns()
        drawdown = self._calculate_drawdown_metrics()
        spike = self._calculate_spike_metrics()
        stability = self._calculate_stability_metrics()

        # 处理市场状态
        if regime_features:
            regime = RegimeFeatures(
                volatility_bucket=regime_features.get("volatility_bucket", "med"),
                trend_strength=regime_features.get("trend_strength", "weak"),
                liquidity_level=regime_features.get("liquidity_level", "high"),
                realized_volatility=regime_features.get("realized_volatility", 0.0),
                adx=regime_features.get("adx", 0.0),
                spread_cost=regime_features.get("spread_cost", 0.0)
            )
        else:
            regime = self._estimate_regime_features()

        # 构建风险信号
        signal = RiskSignal(
            timestamp=timestamp,
            strategy_id=self.strategy_id,
            rolling_returns=rolling_returns,
            drawdown=drawdown,
            spike=spike,
            stability=stability,
            regime=regime,
            gating_events=self.recent_gating_events.copy(),
            allocator_events=self.recent_allocator_events.copy()
        )

        return signal

    def _update_buffer(self, price: float, timestamp: datetime, position: float):
        """更新缓冲区"""
        # 初始化
        if self.initial_price is None:
            self.initial_price = price
            self.current_price = price
            self.buffer.prices.append(price)
            self.buffer.timestamps.append(timestamp)
            return

        # 计算收益
        price_return = (price - self.current_price) / self.current_price

        # 更新缓冲区
        self.buffer.returns.append(price_return)
        self.buffer.prices.append(price)
        self.buffer.timestamps.append(timestamp)

        # 更新状态
        self.current_price = price
        self.current_position = position

    def _calculate_rolling_returns(self) -> RollingReturnMetrics:
        """计算滚动收益"""
        returns = list(self.buffer.returns)

        if len(returns) == 0:
            return RollingReturnMetrics()

        # 滚动窗口收益
        windows = {
            "window_1d_return": 1,
            "window_5d_return": 5,
            "window_20d_return": 20,
            "window_60d_return": 60
        }

        rolling_returns = {}
        for key, window in windows.items():
            if len(returns) >= window:
                rolling_returns[key] = sum(returns[-window:])
            else:
                rolling_returns[key] = 0.0

        # 分位数收益
        if len(returns) >= 20:
            sorted_returns = sorted(returns)
            rolling_returns["percentile_5"] = np.percentile(sorted_returns, 5)
            rolling_returns["percentile_25"] = np.percentile(sorted_returns, 25)
            rolling_returns["percentile_75"] = np.percentile(sorted_returns, 75)
            rolling_returns["percentile_95"] = np.percentile(sorted_returns, 95)
        else:
            rolling_returns["percentile_5"] = 0.0
            rolling_returns["percentile_25"] = 0.0
            rolling_returns["percentile_75"] = 0.0
            rolling_returns["percentile_95"] = 0.0

        return RollingReturnMetrics(**rolling_returns)

    def _calculate_drawdown_metrics(self) -> DrawdownMetrics:
        """计算回撤指标"""
        prices = list(self.buffer.prices)

        if len(prices) < 2:
            return DrawdownMetrics()

        # 计算累计收益
        cumulative = np.array(prices) / self.initial_price - 1.0

        # 当前回撤
        running_max = np.maximum.accumulate(cumulative)
        current_dd = (cumulative[-1] - running_max[-1]) / (running_max[-1] + 1e-10)

        # 最大回撤
        max_dd = abs((cumulative - running_max).min())

        # 回撤持续时间
        dd_duration = 0
        if current_dd < 0:
            # 找到回撤开始点
            for i in range(len(cumulative) - 1, -1, -1):
                if cumulative[i] >= running_max[i]:
                    dd_duration = len(cumulative) - 1 - i
                    break

        # 平均回撤和回撤次数
        drawdowns = []
        in_drawdown = False
        for i in range(len(cumulative)):
            dd = (cumulative[i] - running_max[i]) / (running_max[i] + 1e-10)
            if dd < -0.01:  # 回撤超过 1%
                if not in_drawdown:
                    drawdowns.append(dd)
                    in_drawdown = True
            else:
                in_drawdown = False

        avg_dd = abs(np.mean(drawdowns)) if drawdowns else 0.0
        dd_count = len(drawdowns)
        severe_dd_count = sum(1 for dd in drawdowns if abs(dd) > 0.05)

        return DrawdownMetrics(
            current_drawdown=abs(current_dd),
            max_drawdown=max_dd,
            drawdown_duration=dd_duration,
            avg_drawdown=avg_dd,
            drawdown_count=dd_count,
            severe_drawdown_count=severe_dd_count
        )

    def _calculate_spike_metrics(self) -> SpikeMetrics:
        """计算尖刺指标"""
        returns = list(self.buffer.returns)

        if len(returns) == 0:
            return SpikeMetrics()

        # 最大单步亏损/收益
        max_loss = min(returns)
        max_gain = max(returns)

        # 连续亏损统计
        max_consecutive_losses = 0
        current_consecutive = 0

        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
            else:
                current_consecutive = 0

        # 当前连续亏损
        current_consecutive_losses = 0
        for ret in reversed(returns):
            if ret < 0:
                current_consecutive_losses += 1
            else:
                break

        # 损失聚类评分（近期负收益的聚集程度）
        recent_returns = returns[-20:] if len(returns) >= 20 else returns
        negative_ratio = sum(1 for r in recent_returns if r < 0) / len(recent_returns)
        loss_clustering_score = negative_ratio * 100

        # 近期尖刺次数（20天内超过 2 倍标准差的亏损）
        recent_returns = returns[-20:] if len(returns) >= 20 else returns
        std_return = np.std(recent_returns) if len(recent_returns) > 1 else 0.001
        spike_threshold = -2 * std_return
        recent_spike_count = sum(1 for r in recent_returns if r < spike_threshold)

        return SpikeMetrics(
            max_single_loss=max_loss,
            max_single_gain=max_gain,
            max_consecutive_losses=max_consecutive_losses,
            current_consecutive_losses=current_consecutive_losses,
            loss_clustering_score=loss_clustering_score,
            recent_spike_count=recent_spike_count
        )

    def _calculate_stability_metrics(self) -> StabilityMetrics:
        """计算稳定性指标"""
        returns = list(self.buffer.returns)

        if len(returns) == 0:
            return StabilityMetrics(stability_score=100.0)

        # 计算波动率（不同窗口）
        volatility_1d = np.std(returns[-1:]) if len(returns) >= 1 else 0.0
        volatility_5d = np.std(returns[-5:]) if len(returns) >= 5 else 0.0
        volatility_20d = np.std(returns[-20:]) if len(returns) >= 20 else 0.0

        return_std_1d = volatility_1d
        return_std_5d = volatility_5d
        return_std_20d = volatility_20d

        # 稳定性评分（基于波动率和回撤）
        # 高波动 + 高回撤 = 低稳定性
        volatility_score = min(100, 100 * volatility_20d / 0.02)  # 2% 波动率为基准
        drawdown = self._calculate_drawdown_metrics()
        drawdown_score = min(100, 100 * drawdown.max_drawdown / 0.10)  # 10% 回撤为基准

        stability_score = 100 - (volatility_score * 0.5 + drawdown_score * 0.5)
        stability_score = max(0, min(100, stability_score))

        return StabilityMetrics(
            stability_score=stability_score,
            volatility_1d=volatility_1d,
            volatility_5d=volatility_5d,
            volatility_20d=volatility_20d,
            return_std_1d=return_std_1d,
            return_std_5d=return_std_5d,
            return_std_20d=return_std_20d
        )

    def _estimate_regime_features(self) -> RegimeFeatures:
        """估计市场状态特征（简化版）"""
        returns = list(self.buffer.returns)

        if len(returns) < 20:
            return RegimeFeatures()

        # 实现波动率
        realized_vol = np.std(returns[-20:])

        # 波动率分桶
        if realized_vol < 0.01:
            vol_bucket = "low"
        elif realized_vol < 0.02:
            vol_bucket = "med"
        else:
            vol_bucket = "high"

        # 趋势强度（基于收益的符号一致性）
        recent_returns = returns[-20:]
        positive_days = sum(1 for r in recent_returns if r > 0)
        trend_strength = "strong" if positive_days > 12 else "weak"

        # 流动性（简化：基于价格变动幅度）
        avg_change = np.mean([abs(r) for r in recent_returns])
        liquidity = "high" if avg_change < 0.01 else "low"

        return RegimeFeatures(
            volatility_bucket=vol_bucket,
            trend_strength=trend_strength,
            liquidity_level=liquidity,
            realized_volatility=realized_vol,
            adx=0.0,  # 需要更复杂的计算
            spread_cost=0.0  # 需要实际价差数据
        )

    def add_gating_event(self, event: GatingEvent):
        """添加 gating 事件"""
        self.recent_gating_events.append(event)

        # 保持最近 10 个事件
        if len(self.recent_gating_events) > 10:
            self.recent_gating_events = self.recent_gating_events[-10:]

    def add_allocator_event(self, event: AllocatorEvent):
        """添加 allocator 事件"""
        self.recent_allocator_events.append(event)

        # 保持最近 10 个事件
        if len(self.recent_allocator_events) > 10:
            self.recent_allocator_events = self.recent_allocator_events[-10:]

    def get_latest_signal(self) -> Optional[RiskSignal]:
        """获取最新的风险信号"""
        if not self.buffer.timestamps:
            return None

        timestamp = self.buffer.timestamps[-1]
        price = self.buffer.prices[-1]

        return self.update(price, timestamp, self.current_position)


class PortfolioRiskCollector:
    """组合级风险采集器"""

    def __init__(self, strategy_ids: List[str]):
        """初始化组合采集器

        Args:
            strategy_ids: 策略 ID 列表
        """
        self.strategy_ids = strategy_ids
        self.collectors: Dict[str, RiskMetricsCollector] = {}

        # 为每个策略创建采集器
        for strategy_id in strategy_ids:
            self.collectors[strategy_id] = RiskMetricsCollector(strategy_id)

        # 组合权重
        self.current_weights: Dict[str, float] = {
            sid: 1.0 / len(strategy_ids) for sid in strategy_ids
        }

    def update_strategy(
        self,
        strategy_id: str,
        price: float,
        timestamp: datetime,
        position: float = 0.0
    ) -> RiskSignal:
        """更新单个策略的指标

        Args:
            strategy_id: 策略 ID
            price: 价格
            timestamp: 时间戳
            position: 持仓

        Returns:
            RiskSignal
        """
        if strategy_id not in self.collectors:
            self.collectors[strategy_id] = RiskMetricsCollector(strategy_id)

        collector = self.collectors[strategy_id]
        return collector.update(price, timestamp, position)

    def get_portfolio_signal(self) -> PortfolioRiskSignal:
        """获取组合级风险信号

        Returns:
            PortfolioRiskSignal
        """
        # 获取所有策略的最新信号
        strategy_signals = {}
        for strategy_id, collector in self.collectors.items():
            signal = collector.get_latest_signal()
            if signal:
                strategy_signals[strategy_id] = signal

        # 计算组合指标
        portfolio_signal = self._calculate_portfolio_metrics(strategy_signals)

        return portfolio_signal

    def _calculate_portfolio_metrics(
        self,
        strategy_signals: Dict[str, RiskSignal]
    ) -> PortfolioRiskSignal:
        """计算组合级指标

        Args:
            strategy_signals: 策略级信号字典

        Returns:
            PortfolioRiskSignal
        """
        if not strategy_signals:
            return PortfolioRiskSignal(timestamp=datetime.now())

        # 获取最新时间戳
        latest_timestamp = max(
            s.timestamp for s in strategy_signals.values()
        )

        # 计算组合收益（加权平均）
        portfolio_1d = 0.0
        portfolio_5d = 0.0
        portfolio_20d = 0.0

        for strategy_id, signal in strategy_signals.items():
            weight = self.current_weights.get(strategy_id, 0.0)
            portfolio_1d += signal.rolling_returns.window_1d_return * weight
            portfolio_5d += signal.rolling_returns.window_5d_return * weight
            portfolio_20d += signal.rolling_returns.window_20d_return * weight

        # 计算组合风险指标（简化）
        max_dd = max(
            (s.drawdown.max_drawdown for s in strategy_signals.values()),
            default=0.0
        )

        # 协同指标（需要多策略数据）
        # 简化：使用平均相关性
        avg_correlation = 0.5  # 占位
        tail_correlation = 0.6  # 占位
        co_crash_count = 0  # 占位

        # 计算 CVaR（简化）
        portfolio_cvar_95 = -0.05  # 占位
        portfolio_var = 0.02  # 占位

        return PortfolioRiskSignal(
            timestamp=latest_timestamp,
            portfolio_return_1d=portfolio_1d,
            portfolio_return_5d=portfolio_5d,
            portfolio_return_20d=portfolio_20d,
            portfolio_var=portfolio_var,
            portfolio_cvar_95=portfolio_cvar_95,
            portfolio_max_dd=max_dd,
            avg_correlation=avg_correlation,
            tail_correlation=tail_correlation,
            co_crash_count=co_crash_count,
            strategy_signals=strategy_signals
        )

    def update_weights(self, new_weights: Dict[str, float]):
        """更新组合权重

        Args:
            new_weights: 新权重字典
        """
        # 归一化
        total = sum(new_weights.values())
        if total > 0:
            self.current_weights = {
                k: v / total for k, v in new_weights.items()
            }
