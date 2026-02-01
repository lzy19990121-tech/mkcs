"""
SPL-7a: Risk Monitor - 集成到 Live Runner

在每个交易周期生成 risk snapshot + 事件
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.online.risk_metrics_collector import RiskMetricsCollector
from analysis.online.risk_state_machine import RiskStateMachine
from analysis.online.trend_detector import TrendDetector
from analysis.online.alerting import AlertingManager, Alert
from analysis.online.postmortem_generator import PostMortemGenerator
from analysis.online.risk_event_store import RiskEventStore
from analysis.online.risk_signal_schema import (
    RiskSignal,
    PortfolioRiskSignal,
    GatingEvent,
    AllocatorEvent,
    RegimeFeatures,
    RiskLevel
)
from core.models import Bar, Position, Trade

logger = logging.getLogger(__name__)


class HookPhase(Enum):
    """Hook 阶段"""
    PRE_DECISION = "pre_decision"       # 决策前：更新市场特征
    POST_DECISION = "post_decision"     # 决策后：记录 gating/allocator
    POST_FILL = "post_fill"             # 成交后：更新 PnL/DD/spike


@dataclass
class RiskSnapshot:
    """风险快照（每个周期输出）"""
    timestamp: datetime
    strategy_id: str
    phase: HookPhase

    # 市场特征
    price: float
    regime_features: Dict[str, Any]

    # 风险指标
    rolling_returns: Dict[str, float]
    drawdown: Dict[str, float]
    spike: Dict[str, Any]
    stability: Dict[str, float]

    # 状态
    risk_state: str
    gating_status: str
    allocator_status: str

    # 事件
    alerts_count: int = 0
    gate_triggered: bool = False
    allocator_rebalanced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "phase": self.phase.value,
            "price": self.price,
            "regime_features": self.regime_features,
            "rolling_returns": self.rolling_returns,
            "drawdown": self.drawdown,
            "spike": self.spike,
            "stability": self.stability,
            "risk_state": self.risk_state,
            "gating_status": self.gating_status,
            "allocator_status": self.allocator_status,
            "alerts_count": self.alerts_count,
            "gate_triggered": self.gate_triggered,
            "allocator_rebalanced": self.allocator_rebalanced
        }


class RiskMonitor:
    """风险监控器（SPL-7a 集成版）

    在 Live Trader 的主循环中插入，每个周期产生 risk snapshot + 事件
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        config_path: Optional[str] = None,
        output_dir: str = "data/live_monitoring"
    ):
        """初始化风险监控器

        Args:
            strategy_id: 策略 ID
            symbols: 交易标的列表
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ========== SPL-7a 组件 ==========
        logger.info(f"[RiskMonitor] 初始化 SPL-7a 组件 for {strategy_id}...")

        # 1. 指标采集器（每个 symbol 一个）
        self.collectors: Dict[str, RiskMetricsCollector] = {}
        for symbol in symbols:
            collector_id = f"{strategy_id}_{symbol}"
            self.collectors[symbol] = RiskMetricsCollector(
                strategy_id=collector_id
            )

        # 2. 状态机（每个 symbol 一个）
        self.state_machines: Dict[str, RiskStateMachine] = {}
        for symbol in symbols:
            sm_id = f"{strategy_id}_{symbol}"
            self.state_machines[symbol] = RiskStateMachine(
                strategy_id=sm_id
            )

        # 3. 趋势检测器
        self.trend_detectors: Dict[str, TrendDetector] = {}
        for symbol in symbols:
            detector_id = f"{strategy_id}_{symbol}"
            self.trend_detectors[symbol] = TrendDetector(
                strategy_id=detector_id
            )

        # 4. 告警管理器
        self.alerting_manager = AlertingManager(config_path=config_path)

        # 5. Post-mortem 生成器
        self.postmortem_generator = PostMortemGenerator(
            strategy_id=strategy_id
        )

        # 6. 事件存储
        self.event_store = RiskEventStore(
            db_path=str(self.output_dir / "risk_events.db")
        )

        # ========== 状态缓存 ==========
        self._current_signals: Dict[str, RiskSignal] = {}
        self._current_states: Dict[str, RiskState] = {}
        self._current_trends: Dict[str, Dict] = {}
        self._pending_gating_events: List[GatingEvent] = []
        self._pending_allocator_events: List[AllocatorEvent] = []

        # ========== 统计 ==========
        self.stats = {
            "snapshots_generated": 0,
            "alerts_generated": 0,
            "gates_triggered": 0,
            "allocator_actions": 0,
            "state_transitions": 0
        }

        logger.info(f"[RiskMonitor] SPL-7a 初始化完成")

    # ========== Hook 1: Pre-Decision ==========

    def pre_decision_hook(
        self,
        symbol: str,
        bar: Bar,
        position: Optional[Position],
        context: Dict[str, Any]
    ) -> RiskSignal:
        """Pre-Decision Hook: 更新市场特征

        在生成交易信号前调用，更新：
        - 市场数据（价格、成交量）
        - Regime 特征（vol/ADX/cost proxy）
        - 基础风险指标

        Args:
            symbol: 标的
            bar: 当前 K 线
            position: 当前持仓
            context: 上下文信息

        Returns:
            当前风险信号
        """
        collector = self.collectors.get(symbol)
        if not collector:
            logger.warning(f"[RiskMonitor] No collector for {symbol}")
            return None

        # 更新风险指标
        signal = collector.update(
            price=float(bar.close),
            timestamp=bar.timestamp,
            position=float(position.quantity) if position else 0.0
        )

        # 缓存当前信号
        self._current_signals[symbol] = signal

        # 生成快照
        snapshot = self._generate_snapshot(
            symbol=symbol,
            phase=HookPhase.PRE_DECISION,
            signal=signal
        )

        # 存储 snapshot
        self._store_snapshot(snapshot)

        return signal

    # ========== Hook 2: Post-Decision ==========

    def post_decision_hook(
        self,
        symbol: str,
        gating_result: Optional[Dict[str, Any]],
        allocator_result: Optional[Dict[str, Any]]
    ):
        """Post-Decision Hook: 记录 gating/allocator 决策

        在风控检查后调用，记录：
        - Gating 结果（是否触发、规则、阈值）
        - Allocator 结果（是否 rebalance、权重变化）

        Args:
            symbol: 标的
            gating_result: Gating 结果 {"approved": bool, "rule": str, ...}
            allocator_result: Allocator 结果 {"rebalanced": bool, "weights": dict, ...}
        """
        signal = self._current_signals.get(symbol)
        if not signal:
            return

        # 记录 gating 事件
        if gating_result:
            gating_event = self._create_gating_event(symbol, gating_result)
            if gating_event:
                self._pending_gating_events.append(gating_event)
                signal.gating_events.append(gating_event)

                if gating_event.action in ["GATE", "DISABLE"]:
                    self.stats["gates_triggered"] += 1

        # 记录 allocator 事件
        if allocator_result:
            allocator_event = self._create_allocator_event(symbol, allocator_result)
            if allocator_event:
                self._pending_allocator_events.append(allocator_event)
                signal.allocator_events.append(allocator_event)

                if allocator_event.action == "rebalance":
                    self.stats["allocator_actions"] += 1

        # 生成快照
        snapshot = self._generate_snapshot(
            symbol=symbol,
            phase=HookPhase.POST_DECISION,
            signal=signal
        )

        self._store_snapshot(snapshot)

    # ========== Hook 3: Post-Fill ==========

    def post_fill_hook(
        self,
        symbol: str,
        trade: Optional[Trade],
        current_price: float
    ):
        """Post-Fill Hook: 更新 PnL/DD/spike 并触发告警

        在成交后（或周期结束时）调用：
        - 更新收益指标
        - 计算回撤
        - 检测 spike
        - 触发告警判定

        Args:
            symbol: 标的
            trade: 成交记录（如有）
            current_price: 当前价格
        """
        collector = self.collectors.get(symbol)
        state_machine = self.state_machines.get(symbol)
        trend_detector = self.trend_detectors.get(symbol)

        if not all([collector, state_machine, trend_detector]):
            return

        # 获取当前信号
        signal = self._current_signals.get(symbol)
        if not signal:
            return

        # 更新状态机
        transition = state_machine.update_state(signal)
        if transition:
            self.stats["state_transitions"] += 1

            # 存储状态转换事件
            self.event_store.store_event(
                event_type="STATE_TRANSITION",
                data=transition.to_dict()
            )

        # 趋势检测
        trend_alerts = trend_detector.update(signal)
        # 将 TrendAlert 对象转换为简单的字典
        trends = {}
        for alert in trend_alerts:
            trends[alert.metric_name] = {
                "direction": alert.direction.value,
                "slope": alert.slope,
                "r_squared": alert.r_squared,
                "trend_type": alert.trend_type.value
            }
        self._current_trends[symbol] = trends

        # 告警判定
        current_state = state_machine.get_current_state()
        alerts = self.alerting_manager.process_risk_update(
            signal=signal,
            state=current_state,
            trends=trends,
            state_transition=transition
        )

        # 处理告警
        for alert in alerts:
            self.stats["alerts_generated"] += 1

            # 存储告警事件
            self.event_store.store_event(
                event_type="ALERT",
                data=alert.to_dict()
            )

            # 检查是否需要生成 post-mortem
            if alert.severity.value == "critical":
                self._generate_postmortem_for_alert(symbol, alert, signal)

        # 生成快照
        snapshot = self._generate_snapshot(
            symbol=symbol,
            phase=HookPhase.POST_FILL,
            signal=signal
        )

        self._store_snapshot(snapshot)

    # ========== 内部方法 ==========

    def _generate_snapshot(
        self,
        symbol: str,
        phase: HookPhase,
        signal: RiskSignal
    ) -> RiskSnapshot:
        """生成风险快照"""
        state_machine = self.state_machines.get(symbol)
        current_state = state_machine.get_current_state() if state_machine else RiskState.NORMAL

        snapshot = RiskSnapshot(
            timestamp=datetime.now(),
            strategy_id=self.strategy_id,
            phase=phase,
            price=float(signal.rolling_returns.window_1d_return) if signal.rolling_returns else 0,
            regime_features=signal.regime.to_dict() if signal.regime else {},
            rolling_returns=signal.rolling_returns.to_dict() if signal.rolling_returns else {},
            drawdown=signal.drawdown.to_dict() if signal.drawdown else {},
            spike=signal.spike.to_dict() if signal.spike else {},
            stability=signal.stability.to_dict() if signal.stability else {},
            risk_state=current_state.value,
            gating_status="gated" if signal.gating_events else "active",
            allocator_status="rebalanced" if signal.allocator_events else "hold",
            alerts_count=len([a for a in signal.gating_events]),  # 简化
            gate_triggered=len(signal.gating_events) > 0,
            allocator_rebalanced=len(signal.allocator_events) > 0
        )

        self.stats["snapshots_generated"] += 1
        return snapshot

    def _store_snapshot(self, snapshot: RiskSnapshot):
        """存储快照到文件和数据库"""
        # 写到 JSONL 文件（便于流式读取）
        snapshot_file = self.output_dir / "snapshots.jsonl"
        with open(snapshot_file, "a") as f:
            import json
            f.write(json.dumps(snapshot.to_dict()) + "\n")

        # 每 10 个快照打印一次
        if self.stats["snapshots_generated"] % 10 == 0:
            logger.info(
                f"[RiskMonitor] 已生成 {self.stats['snapshots_generated']} 个快照, "
                f"告警 {self.stats['alerts_generated']}, "
                f"Gating {self.stats['gates_triggered']}"
            )

    def _create_gating_event(
        self,
        symbol: str,
        gating_result: Dict[str, Any]
    ) -> Optional[GatingEvent]:
        """创建 Gating 事件"""
        if not gating_result:
            return None

        approved = gating_result.get("approved", True)
        reason = gating_result.get("risk_reason", "")

        event = GatingEvent(
            event_id=f"gating_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            strategy_id=self.strategy_id,
            action="GATE" if not approved else "ALLOW",
            rule_id="risk_manager",
            threshold=None,
            current_value=None,
            regime_features=None,
            reason=reason
        )

        return event

    def _create_allocator_event(
        self,
        symbol: str,
        allocator_result: Dict[str, Any]
    ) -> Optional[AllocatorEvent]:
        """创建 Allocator 事件"""
        if not allocator_result:
            return None

        event = AllocatorEvent(
            event_id=f"allocator_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            allocator_type="rules",
            action="rebalance" if allocator_result.get("rebalanced") else "hold",
            old_weights=allocator_result.get("old_weights", {}),
            new_weights=allocator_result.get("new_weights", {}),
            weight_changes={},
            reason=allocator_result.get("reason", ""),
            constraints_hit=allocator_result.get("constraints_hit", [])
        )

        return event

    def _generate_postmortem_for_alert(
        self,
        symbol: str,
        alert: Alert,
        signal: RiskSignal
    ):
        """为严重告警生成 post-mortem"""
        try:
            # 将 alert 转换为 gating event 格式（兼容）
            from analysis.online.postmortem_generator import GatingEvent as PMGatingEvent

            pm_event = PMGatingEvent(
                event_id=alert.alert_id,
                timestamp=alert.timestamp,
                strategy_id=self.strategy_id,
                action="ALERT",
                rule_id=alert.rule_id,
                threshold=alert.threshold,
                current_value=alert.current_value,
                reason=alert.message
            )

            # 生成 post-mortem
            report = self.postmortem_generator.generate_from_gate_event(pm_event)

            # 保存报告
            report_file = self.output_dir / f"postmortem_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, "w") as f:
                f.write(report.markdown_report)

            logger.info(f"[RiskMonitor] Post-mortem 已生成: {report_file}")

        except Exception as e:
            logger.error(f"[RiskMonitor] 生成 post-mortem 失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "current_states": {
                symbol: state.value
                for symbol, state in self._current_states.items()
            },
            "pending_gating_events": len(self._pending_gating_events),
            "pending_allocator_events": len(self._pending_allocator_events)
        }

    def shutdown(self):
        """关闭监控器"""
        logger.info("[RiskMonitor] 关闭中...")
        logger.info(f"[RiskMonitor] 统计: {self.get_stats()}")
        logger.info("[RiskMonitor] 已关闭")


if __name__ == "__main__":
    """测试 RiskMonitor"""
    print("=== RiskMonitor 测试 ===\n")

    from datetime import timedelta

    # 创建监控器
    monitor = RiskMonitor(
        strategy_id="test_strategy",
        symbols=["AAPL", "MSFT"],
        output_dir="data/test_monitoring"
    )

    # 模拟 pre-decision hook
    now = datetime.now()
    bar = Bar(
        symbol="AAPL",
        timestamp=now,
        open=Decimal("175.0"),
        high=Decimal("176.0"),
        low=Decimal("174.0"),
        close=Decimal("175.5"),
        volume=1000000,
        interval="1m"
    )

    signal = monitor.pre_decision_hook(
        symbol="AAPL",
        bar=bar,
        position=None,
        context={}
    )

    print(f"✓ Pre-decision hook 完成")
    print(f"  Risk state: {signal is not None}")

    # 模拟 post-decision hook
    monitor.post_decision_hook(
        symbol="AAPL",
        gating_result={"approved": False, "risk_reason": "测试 gating"},
        allocator_result=None
    )

    print(f"✓ Post-decision hook 完成")

    # 模拟 post-fill hook
    monitor.post_fill_hook(
        symbol="AAPL",
        trade=None,
        current_price=175.5
    )

    print(f"✓ Post-fill hook 完成")

    # 打印统计
    stats = monitor.get_stats()
    print(f"\n统计: {stats}")

    monitor.shutdown()

    print("\n✅ RiskMonitor 测试通过")
