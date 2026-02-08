"""
SPL-7a-D: Post-mortem 自动归因

当发生风险事件时自动生成 post-mortem 报告。
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.online.risk_signal_schema import RiskSignal, GatingEvent, AllocatorEvent
from analysis.online.risk_state_machine import RiskState, StateTransitionEvent
from analysis.replay_schema import ReplayOutput, load_replay_outputs


class PostMortemTriggerType(Enum):
    """Post-mortem 触发类型"""
    GATE_TRIGGERED = "gate_triggered"           # Gate 触发
    ENVELOPE_TOUCHED = "envelope_touched"       # Envelope 被触及
    SPIKE_ANOMALY = "spike_anomaly"             # 异常 spike
    CO_CRASH = "co_crash"                       # Co-crash
    STATE_CRITICAL = "state_critical"           # 状态转为严重
    MANUAL = "manual"                           # 手动触发


@dataclass
class PostMortemReport:
    """Post-mortem 报告"""
    report_id: str
    trigger_type: PostMortemTriggerType
    strategy_id: str

    # 时间信息
    trigger_time: datetime
    window_start: datetime
    window_end: datetime

    # 触发上下文
    trigger_event: Dict[str, Any]

    # 关键指标变化轨迹
    metrics_trajectory: List[Dict[str, float]]

    # 触发的规则/约束
    triggered_rules: List[Dict[str, Any]]
    binding_constraints: List[Dict[str, Any]]

    # 市场状态判断
    regime_at_trigger: Dict[str, Any]

    # 统计信息
    statistics: Dict[str, Any]

    # 对应的 replay 片段指针
    replay_pointers: List[Dict[str, Any]]

    # 归因分析
    root_cause_analysis: Dict[str, Any]

    # 建议
    recommendations: List[str]

    # 元信息
    generated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "report_id": self.report_id,
            "trigger_type": self.trigger_type.value,
            "strategy_id": self.strategy_id,
            "trigger_time": self.trigger_time.isoformat(),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "trigger_event": self.trigger_event,
            "metrics_trajectory": self.metrics_trajectory,
            "triggered_rules": self.triggered_rules,
            "binding_constraints": self.binding_constraints,
            "regime_at_trigger": self.regime_at_trigger,
            "statistics": self.statistics,
            "replay_pointers": self.replay_pointers,
            "root_cause_analysis": self.root_cause_analysis,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
            "version": self.version
        }

    def to_markdown(self) -> str:
        """转换为 Markdown 报告"""
        lines = []
        lines.append(f"# Post-mortem Report: {self.report_id}\n")
        lines.append(f"**Strategy**: {self.strategy_id}")
        lines.append(f"**Trigger Type**: {self.trigger_type.value}")
        lines.append(f"**Trigger Time**: {self.trigger_time.isoformat()}\n")

        lines.append("## 📊 Trigger Event\n")
        lines.append(f"```json")
        lines.append(json.dumps(self.trigger_event, indent=2))
        lines.append(f"```\n")

        lines.append("## 📈 Metrics Trajectory\n")
        lines.append("| Time | Return | Drawdown | Volatility | Spike |")
        lines.append("|------|--------|----------|------------|-------|")
        for point in self.metrics_trajectory:
            lines.append(
                f"| {point['time']} | {point.get('return', 0):.2%} | "
                f"{point.get('drawdown', 0):.2%} | {point.get('volatility', 0):.2%} | "
                f"{point.get('spike', 0)} |"
            )
        lines.append("")

        lines.append("## 🎯 Triggered Rules\n")
        for rule in self.triggered_rules:
            lines.append(f"- **{rule.get('rule_id', 'unknown')}**: {rule.get('description', '')}")
        lines.append("")

        lines.append("## 🌐 Market Regime at Trigger\n")
        lines.append(f"- Volatility Bucket: {self.regime_at_trigger.get('volatility_bucket', 'unknown')}")
        lines.append(f"- Trend Strength: {self.regime_at_trigger.get('trend_strength', 'unknown')}")
        lines.append(f"- Liquidity: {self.regime_at_trigger.get('liquidity_level', 'unknown')}")
        lines.append("")

        lines.append("## 🔍 Root Cause Analysis\n")
        lines.append(f"### Primary Cause\n")
        lines.append(f"{self.root_cause_analysis.get('primary_cause', 'Unknown')}\n")

        lines.append(f"### Contributing Factors\n")
        for factor in self.root_cause_analysis.get('contributing_factors', []):
            lines.append(f"- {factor}")
        lines.append("")

        lines.append("## 💡 Recommendations\n")
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        return "\n".join(lines)


@dataclass
class PostMortemConfig:
    """Post-mortem 配置"""
    # 时间窗口配置
    pre_trigger_window_seconds: int = 3600     # 触发前 1 小时
    post_trigger_window_seconds: int = 1800    # 触发后 30 分钟

    # 输出配置
    output_dir: str = "outputs/postmortems"
    save_markdown: bool = True
    save_json: bool = True

    # 指标采样频率
    sample_frequency_seconds: int = 60         # 每分钟

    # 最小报告间隔（防止重复生成）
    min_report_interval_seconds: int = 1800    # 30 分钟


class PostMortemGenerator:
    """Post-mortem 生成器

    自动分析风险事件并生成归因报告。
    """

    def __init__(
        self,
        strategy_id: str,
        config: Optional[PostMortemConfig] = None,
        replay_data_path: Optional[str] = None
    ):
        """初始化生成器

        Args:
            strategy_id: 策略 ID
            config: 配置
            replay_data_path: Replay 数据路径
        """
        self.strategy_id = strategy_id
        self.config = config or PostMortemConfig()
        self.replay_data_path = replay_data_path

        # 信号缓存（用于回溯历史）
        self.signal_history: List[RiskSignal] = []
        self.max_history_hours: int = 24

        # 最近生成的报告
        self.last_report_time: Dict[PostMortemTriggerType, datetime] = {}

    def add_signal(self, signal: RiskSignal):
        """添加信号到历史

        Args:
            signal: 风险信号
        """
        self.signal_history.append(signal)

        # 保持历史长度
        cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
        self.signal_history = [
            s for s in self.signal_history if s.timestamp >= cutoff_time
        ]

    def generate_from_gate_event(
        self,
        event: GatingEvent
    ) -> Optional[PostMortemReport]:
        """从 gating 事件生成 post-mortem

        Args:
            event: Gating 事件

        Returns:
            PostMortemReport
        """
        # 检查是否需要生成报告
        if event.action not in ["GATE", "DISABLE"]:
            return None

        if not self._check_report_cooldown(PostMortemTriggerType.GATE_TRIGGERED):
            return None

        # 提取上下文窗口
        window_signals = self._get_context_window(
            event.timestamp,
            self.config.pre_trigger_window_seconds,
            self.config.post_trigger_window_seconds
        )

        if not window_signals:
            return None

        # 生成报告
        report = PostMortemReport(
            report_id=f"pm_gate_{event.strategy_id}_{event.timestamp.strftime('%Y%m%d%H%M%S')}",
            trigger_type=PostMortemTriggerType.GATE_TRIGGERED,
            strategy_id=event.strategy_id,
            trigger_time=event.timestamp,
            window_start=window_signals[0].timestamp,
            window_end=window_signals[-1].timestamp,
            trigger_event=event.to_dict(),
            metrics_trajectory=self._extract_metrics_trajectory(window_signals),
            triggered_rules=[{
                "rule_id": event.rule_id,
                "description": f"Gating rule triggered: {event.reason}",
                "action": event.action,
                "threshold": event.threshold,
                "current_value": event.current_value
            }],
            binding_constraints=[],
            regime_at_trigger=event.regime_features.to_dict() if event.regime_features else {},
            statistics=self._calculate_statistics(window_signals),
            replay_pointers=self._find_replay_pointers(event.timestamp),
            root_cause_analysis=self._analyze_root_cause(event, window_signals),
            recommendations=self._generate_recommendations(event, window_signals)
        )

        # 保存报告
        self._save_report(report)

        self.last_report_time[PostMortemTriggerType.GATE_TRIGGERED] = datetime.now()

        return report

    def generate_from_state_transition(
        self,
        transition: StateTransitionEvent
    ) -> Optional[PostMortemReport]:
        """从状态转换生成 post-mortem

        Args:
            transition: 状态转换事件

        Returns:
            PostMortemReport
        """
        # 只对 CRITICAL 转换生成报告
        if transition.to_state != RiskState.CRITICAL:
            return None

        if not self._check_report_cooldown(PostMortemTriggerType.STATE_CRITICAL):
            return None

        # 提取上下文窗口
        window_signals = self._get_context_window(
            transition.timestamp,
            self.config.pre_trigger_window_seconds,
            self.config.post_trigger_window_seconds
        )

        if not window_signals:
            return None

        # 生成报告
        report = PostMortemReport(
            report_id=f"pm_state_{transition.strategy_id}_{transition.timestamp.strftime('%Y%m%d%H%M%S')}",
            trigger_type=PostMortemTriggerType.STATE_CRITICAL,
            strategy_id=transition.strategy_id,
            trigger_time=transition.timestamp,
            window_start=window_signals[0].timestamp,
            window_end=window_signals[-1].timestamp,
            trigger_event=transition.to_dict(),
            metrics_trajectory=self._extract_metrics_trajectory(window_signals),
            triggered_rules=[{
                "rule_id": transition.trigger_metric,
                "description": f"State transition: {transition.from_state.value} → {transition.to_state.value}",
                "threshold": transition.threshold,
                "current_value": transition.trigger_value
            }],
            binding_constraints=[],
            regime_at_trigger=transition.context.get("regime_features", {}),
            statistics=self._calculate_statistics(window_signals),
            replay_pointers=self._find_replay_pointers(transition.timestamp),
            root_cause_analysis=self._analyze_root_cause(transition, window_signals),
            recommendations=self._generate_recommendations(transition, window_signals)
        )

        self._save_report(report)
        self.last_report_time[PostMortemTriggerType.STATE_CRITICAL] = datetime.now()

        return report

    def _get_context_window(
        self,
        trigger_time: datetime,
        pre_window_seconds: int,
        post_window_seconds: int
    ) -> List[RiskSignal]:
        """获取上下文窗口内的信号

        Args:
            trigger_time: 触发时间
            pre_window_seconds: 触发前窗口
            post_window_seconds: 触发后窗口

        Returns:
            信号列表
        """
        window_start = trigger_time - timedelta(seconds=pre_window_seconds)
        window_end = trigger_time + timedelta(seconds=post_window_seconds)

        return [
            s for s in self.signal_history
            if window_start <= s.timestamp <= window_end
        ]

    def _extract_metrics_trajectory(
        self,
        signals: List[RiskSignal]
    ) -> List[Dict[str, float]]:
        """提取指标轨迹

        Args:
            signals: 信号列表

        Returns:
            指标轨迹
        """
        trajectory = []

        for signal in signals:
            # 计算累计收益（简化）
            cumulative_return = signal.rolling_returns.window_1d_return  # 占位

            point = {
                "time": signal.timestamp.isoformat(),
                "return": cumulative_return,
                "drawdown": signal.drawdown.current_drawdown,
                "volatility": signal.stability.volatility_20d,
                "spike": float(signal.spike.recent_spike_count),
                "stability_score": signal.stability.stability_score
            }
            trajectory.append(point)

        return trajectory

    def _calculate_statistics(
        self,
        signals: List[RiskSignal]
    ) -> Dict[str, Any]:
        """计算统计信息

        Args:
            signals: 信号列表

        Returns:
            统计信息
        """
        if not signals:
            return {}

        drawdowns = [s.drawdown.current_drawdown for s in signals]
        volatilities = [s.stability.volatility_20d for s in signals]
        spikes = [s.spike.recent_spike_count for s in signals]

        return {
            "max_drawdown": max(drawdowns),
            "avg_drawdown": np.mean(drawdowns),
            "max_volatility": max(volatilities),
            "avg_volatility": np.mean(volatilities),
            "total_spikes": sum(spikes),
            "duration_hours": (signals[-1].timestamp - signals[0].timestamp).total_seconds() / 3600
        }

    def _find_replay_pointers(
        self,
        trigger_time: datetime,
        trade_id: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """查找对应的 replay 片段指针

        Args:
            trigger_time: 触发时间
            trade_id: 交易 ID（可选）
            symbol: 交易品种（可选）

        Returns:
            Replay 指针列表
        """
        try:
            # 使用 ReplayLocator 查找数据
            locator = ReplayLocator(self.replay_data_path or "runs/")

            # 构建查找条件
            conditions = {}
            if trade_id:
                conditions["trade_id"] = trade_id
            if symbol:
                conditions["symbol"] = symbol
            conditions["timestamp"] = trigger_time

            # 查找对应的 replay 数据
            pointers = locator.find_replay_data(conditions)

            if pointers:
                return pointers

            # 如果没有找到，返回默认范围
            return [
                {
                    "replay_id": "unknown",
                    "segment_start": (trigger_time - timedelta(hours=1)).isoformat(),
                    "segment_end": (trigger_time + timedelta(minutes=30)).isoformat(),
                    "data_path": self.replay_data_path or "runs/",
                    "status": "fallback",
                    "reason": f"No replay data found for conditions: {conditions}"
                }
            ]

        except Exception as e:
            # 异常时返回降级结果
            return [
                {
                    "replay_id": "error",
                    "segment_start": (trigger_time - timedelta(hours=1)).isoformat(),
                    "segment_end": (trigger_time + timedelta(minutes=30)).isoformat(),
                    "data_path": self.replay_data_path or "runs/",
                    "status": "error",
                    "reason": f"Failed to locate replay data: {str(e)}"
                }
            ]

    def _analyze_root_cause(
        self,
        event: Any,
        signals: List[RiskSignal]
    ) -> Dict[str, Any]:
        """分析根本原因

        Args:
            event: 触发事件
            signals: 信号列表

        Returns:
            根本原因分析
        """
        # 分析主要因素
        factors = []

        # 检查回撤
        max_dd = max((s.drawdown.current_drawdown for s in signals), default=0)
        if max_dd > 0.05:
            factors.append(f"Severe drawdown ({max_dd:.1%})")

        # 检查波动率
        max_vol = max((s.stability.volatility_20d for s in signals), default=0)
        if max_vol > 0.03:
            factors.append(f"High volatility ({max_vol:.1%})")

        # 检查 spike
        total_spikes = sum((s.spike.recent_spike_count for s in signals))
        if total_spikes > 5:
            factors.append(f"Multiple spikes ({total_spikes} events)")

        # 检查市场状态
        if signals:
            latest_signal = signals[-1]
            if latest_signal.regime.volatility_bucket == "high":
                factors.append("High volatility regime")
            if latest_signal.regime.trend_strength == "strong":
                factors.append("Strong trend regime")

        # 判定主要成因
        primary_cause = factors[0] if factors else "Unknown"

        return {
            "primary_cause": primary_cause,
            "contributing_factors": factors,
            "confidence": "medium"  # low/medium/high
        }

    def _generate_recommendations(
        self,
        event: Any,
        signals: List[RiskSignal]
    ) -> List[str]:
        """生成建议

        Args:
            event: 触发事件
            signals: 信号列表

        Returns:
            建议列表
        """
        recommendations = []

        if not signals:
            return ["No data available for analysis"]

        latest_signal = signals[-1]

        # 基于 volatility 的建议
        if latest_signal.stability.volatility_20d > 0.03:
            recommendations.append("考虑收紧 gating 阈值以应对高波动环境")

        # 基于 drawdown 的建议
        if latest_signal.drawdown.current_drawdown > 0.05:
            recommendations.append("监控回撤恢复情况，必要时降低仓位")

        # 基于 spike 的建议
        if latest_signal.spike.recent_spike_count > 5:
            recommendations.append("检查市场状态，考虑暂时降低风险敞口")

        # 基于 regime 的建议
        if latest_signal.regime.volatility_bucket == "high":
            recommendations.append("当前高波动环境，建议启用保守策略")

        # 通用建议
        recommendations.append("继续监控风险指标，准备应急响应")

        return recommendations

    def _check_report_cooldown(
        self,
        trigger_type: PostMortemTriggerType
    ) -> bool:
        """检查报告冷却时间

        Args:
            trigger_type: 触发类型

        Returns:
            True（可以生成）或 False（冷却中）
        """
        if trigger_type not in self.last_report_time:
            return True

        elapsed = (datetime.now() - self.last_report_time[trigger_type]).total_seconds()
        return elapsed >= self.config.min_report_interval_seconds

    def _save_report(self, report: PostMortemReport):
        """保存报告

        Args:
            report: Post-mortem 报告
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 JSON
        if self.config.save_json:
            json_file = output_dir / f"{report.report_id}.json"
            with open(json_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            print(f"Post-mortem JSON saved: {json_file}")

        # 保存 Markdown
        if self.config.save_markdown:
            md_file = output_dir / f"{report.report_id}.md"
            with open(md_file, 'w') as f:
                f.write(report.to_markdown())
            print(f"Post-mortem Markdown saved: {md_file}")


class ReplayLocator:
    """Replay 数据定位器

    根据交易 ID、时间戳、品种等条件查找对应的 replay 数据片段。
    """

    def __init__(self, runs_dir: str = "runs/"):
        """初始化定位器

        Args:
            runs_dir: runs 目录路径
        """
        self.runs_dir = Path(runs_dir)
        self._replay_cache: Dict[str, ReplayOutput] = {}

    def find_replay_data(
        self,
        conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """查找 replay 数据

        Args:
            conditions: 查找条件
                - trade_id: 交易 ID
                - timestamp: 时间戳（datetime）
                - symbol: 交易品种
                - strategy_id: 策略 ID
                - run_id: 运行 ID

        Returns:
            找到的 replay 指针列表
        """
        pointers = []

        # 按 trade_id 查找
        if "trade_id" in conditions:
            trade_id = conditions["trade_id"]
            pointers.extend(self._find_by_trade_id(trade_id))

        # 按 timestamp 和 symbol 查找
        elif "timestamp" in conditions:
            timestamp = conditions["timestamp"]
            symbol = conditions.get("symbol")
            pointers.extend(self._find_by_time(timestamp, symbol))

        # 按 run_id 查找
        elif "run_id" in conditions:
            run_id = conditions["run_id"]
            pointers.extend(self._find_by_run_id(run_id))

        return pointers

    def _find_by_trade_id(self, trade_id: str) -> List[Dict[str, Any]]:
        """按交易 ID 查找

        Args:
            trade_id: 交易 ID

        Returns:
            replay 指针列表
        """
        pointers = []

        # 扫描所有 run 目录
        for replay in self._load_all_replays():
            # 查找匹配的 trade
            for trade in replay.trades:
                if trade.trade_id == trade_id:
                    pointers.append({
                        "replay_id": replay.run_id,
                        "strategy_id": replay.strategy_id,
                        "strategy_name": replay.strategy_name,
                        "trade_id": trade.trade_id,
                        "trade_timestamp": trade.timestamp.isoformat(),
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "price": float(trade.price),
                        "quantity": trade.quantity,
                        "segment_start": (trade.timestamp - timedelta(hours=1)).isoformat(),
                        "segment_end": (trade.timestamp + timedelta(minutes=30)).isoformat(),
                        "data_path": str(self.runs_dir / replay.run_id),
                        "signal_id": trade.signal_id,
                        "status": "found"
                    })

        return pointers

    def _find_by_time(
        self,
        timestamp: datetime,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """按时间查找

        Args:
            timestamp: 时间戳
            symbol: 交易品种（可选）

        Returns:
            replay 指针列表
        """
        pointers = []

        # 扫描所有 run 目录
        for replay in self._load_all_replays():
            # 检查时间范围是否覆盖
            if not (replay.start_date <= timestamp.date() <= replay.end_date):
                continue

            # 查找该时间附近的 steps
            nearby_steps = [
                s for s in replay.steps
                if abs((s.timestamp - timestamp).total_seconds()) < 3600  # 1小时内
            ]

            if nearby_steps:
                # 找到最近的 step
                nearest = min(nearby_steps, key=lambda s: abs((s.timestamp - timestamp).total_seconds()))

                # 查找该时间的交易
                relevant_trades = []
                for trade in replay.trades:
                    if abs((trade.timestamp - timestamp).total_seconds()) < 3600:
                        if symbol is None or trade.symbol == symbol:
                            relevant_trades.append({
                                "trade_id": trade.trade_id,
                                "symbol": trade.symbol,
                                "side": trade.side,
                                "price": float(trade.price),
                                "quantity": trade.quantity
                            })

                pointers.append({
                    "replay_id": replay.run_id,
                    "strategy_id": replay.strategy_id,
                    "strategy_name": replay.strategy_name,
                    "query_timestamp": timestamp.isoformat(),
                    "nearest_step_time": nearest.timestamp.isoformat(),
                    "segment_start": (timestamp - timedelta(hours=1)).isoformat(),
                    "segment_end": (timestamp + timedelta(minutes=30)).isoformat(),
                    "data_path": str(self.runs_dir / replay.run_id),
                    "relevant_trades": relevant_trades,
                    "status": "found"
                })

        return pointers

    def _find_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        """按运行 ID 查找

        Args:
            run_id: 运行 ID

        Returns:
            replay 指针列表
        """
        replay_path = self.runs_dir / run_id

        if not replay_path.exists():
            return [{
                "replay_id": run_id,
                "status": "not_found",
                "reason": f"Run directory not found: {replay_path}"
            }]

        try:
            replay = ReplayOutput.from_directory(replay_path)

            return [{
                "replay_id": replay.run_id,
                "strategy_id": replay.strategy_id,
                "strategy_name": replay.strategy_name,
                "start_date": replay.start_date.isoformat(),
                "end_date": replay.end_date.isoformat(),
                "data_path": str(replay_path),
                "total_trades": len(replay.trades),
                "total_steps": len(replay.steps),
                "status": "found"
            }]

        except Exception as e:
            return [{
                "replay_id": run_id,
                "status": "error",
                "reason": f"Failed to load replay: {str(e)}"
            }]

    def _load_all_replays(self) -> List[ReplayOutput]:
        """加载所有 replay 数据

        Returns:
            ReplayOutput 列表
        """
        if not self._replay_cache:
            try:
                self._replay_cache = {
                    r.run_id: r for r in load_replay_outputs(str(self.runs_dir))
                }
            except Exception as e:
                print(f"Warning: Failed to load replays: {e}")

        return list(self._replay_cache.values())

    def get_signal_context(
        self,
        run_id: str,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """获取信号上下文

        查找指定 run 中指定时间点的信号状态。

        Args:
            run_id: 运行 ID
            timestamp: 时间戳

        Returns:
            信号上下文信息
        """
        # 查找 replay
        pointers = self._find_by_run_id(run_id)

        if not pointers or pointers[0]["status"] != "found":
            return {
                "status": "not_found",
                "reason": f"Replay not found for run_id: {run_id}"
            }

        replay = self._replay_cache.get(run_id)
        if not replay:
            return {
                "status": "error",
                "reason": "Replay not in cache"
            }

        # 查找最近的 step
        nearest_step = None
        min_diff = float('inf')

        for step in replay.steps:
            diff = abs((step.timestamp - timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                nearest_step = step

        if nearest_step is None:
            return {
                "status": "not_found",
                "reason": "No steps found in replay"
            }

        # 查找相关交易
        nearby_trades = [
            t for t in replay.trades
            if abs((t.timestamp - timestamp).total_seconds()) < 3600
        ]

        return {
            "status": "found",
            "step": {
                "timestamp": nearest_step.timestamp.isoformat(),
                "step_pnl": float(nearest_step.step_pnl),
                "equity": float(nearest_step.equity),
                "signal_state": nearest_step.signal_state
            },
            "nearby_trades": [
                {
                    "trade_id": t.trade_id,
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "side": t.side,
                    "price": float(t.price),
                    "quantity": t.quantity
                }
                for t in nearby_trades
            ],
            "config": {
                "cost_model": nearest_step.cost_model,
                "slippage": nearest_step.slippage
            }
        }

    def clear_cache(self) -> None:
        """清除缓存"""
        self._replay_cache.clear()
