"""
SPL-7a-B: 风险状态机与趋势判定

定义风险状态机（NORMAL / WARNING / CRITICAL）并检测风险趋势。
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from collections import deque
import numpy as np

from analysis.online.risk_signal_schema import RiskSignal, PortfolioRiskSignal


class RiskState(Enum):
    """风险状态"""
    NORMAL = "NORMAL"               # 正常：风险在可控范围内
    WARNING = "WARNING"             # 警告：接近 envelope 或风险上升趋势
    CRITICAL = "CRITICAL"           # 严重：触发 gate 或接近 hard limit


class StateTransition(Enum):
    """状态转换类型"""
    UPGRADE = "upgrade"      # 状态升级（NORMAL → WARNING → CRITICAL）
    DOWNGRADE = "downgrade"  # 状态降级（CRITICAL → WARNING → NORMAL���
    HOLD = "hold"            # 状态保持


@dataclass
class StateTransitionEvent:
    """状态转换事件"""
    timestamp: datetime
    from_state: RiskState
    to_state: RiskState
    transition_type: StateTransition

    # 触发原因
    trigger_metric: str
    trigger_value: float
    threshold: float

    # 上下文
    strategy_id: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "transition_type": self.transition_type.value,
            "trigger_metric": self.trigger_metric,
            "trigger_value": self.trigger_value,
            "threshold": self.threshold,
            "strategy_id": self.strategy_id,
            "context": self.context
        }


@dataclass
class StateMachineConfig:
    """状态机配置"""
    # WARNING 阈值（接近 envelope）
    warning_envelope_usage: float = 0.7      # envelope 利用率 70%
    warning_spike_growth: float = 0.5        # spike 指标增长 50%
    warning_gating_frequency: float = 0.2    # gating 频率 20%
    warning_volatility: float = 0.025        # 波动率 2.5%

    # CRITICAL 阈值（触发 gate 或 hard limit）
    critical_envelope_usage: float = 0.9     # envelope 利用率 90%
    critical_spike_growth: float = 1.0       # spike 指标增长 100%
    critical_gating_frequency: float = 0.4   # gating 频率 40%
    critical_volatility: float = 0.04        # 波动率 4%

    # 状态保持时间（防止抖动）
    min_state_duration_seconds: int = 300    # 最小状态持续时间 5 分钟

    # 趋势窗口
    trend_window_seconds: int = 3600         # 趋势窗口 1 小时

    # 评分权重
    envelope_weight: float = 0.4
    spike_weight: float = 0.3
    gating_weight: float = 0.2
    volatility_weight: float = 0.1


class RiskStateMachine:
    """风险状态机

    根据风险信号自动判定当前风险状态。
    """

    def __init__(
        self,
        strategy_id: str,
        config: Optional[StateMachineConfig] = None
    ):
        """初始化状态机

        Args:
            strategy_id: 策略 ID
            config: 状态机配置
        """
        self.strategy_id = strategy_id
        self.config = config or StateMachineConfig()

        # 当前状态
        self.current_state: RiskState = RiskState.NORMAL
        self.state_since: datetime = datetime.now()

        # 历史状态
        self.state_history: deque = deque(maxlen=100)
        self.transition_history: List[StateTransitionEvent] = []

        # 趋势数据
        self.trend_data: Dict[str, deque] = {
            "envelope_usage": deque(maxlen=60),  # 60 个数据点
            "spike_growth": deque(maxlen=60),
            "gating_frequency": deque(maxlen=60),
            "volatility": deque(maxlen=60)
        }

        # 最近的 gating 事件
        self.recent_gating_events: deque = deque(maxlen=100)

    def update_state(self, signal: RiskSignal) -> Optional[StateTransitionEvent]:
        """更新状态

        Args:
            signal: 风险信号

        Returns:
            StateTransitionEvent（如果状态发生转换）
        """
        # 更新趋势数据
        self._update_trend_data(signal)

        # 记录 gating 事件
        for event in signal.gating_events:
            self.recent_gating_events.append(event)

        # 计算风险评分
        risk_score = self._calculate_risk_score(signal)

        # 判定新状态
        new_state = self._determine_state(risk_score, signal)

        # 检查是否需要转换状态
        transition = None
        if new_state != self.current_state:
            # 检查最小持续时间
            time_in_state = (datetime.now() - self.state_since).total_seconds()
            if time_in_state >= self.config.min_state_duration_seconds:
                # 执行状态转换
                transition = self._transition_to(new_state, signal, risk_score)

        # 记录状态历史
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "state": self.current_state.value,
            "risk_score": risk_score
        })

        return transition

    def _update_trend_data(self, signal: RiskSignal):
        """更新趋势数据"""
        # Envelope 利用率（current_drawdown / max_drawdown_limit）
        envelope_limit = 0.10  # 10% envelope limit
        envelope_usage = min(1.0, signal.drawdown.current_drawdown / envelope_limit)
        self.trend_data["envelope_usage"].append(envelope_usage)

        # Spike 增长率（近期 spike count 相对于历史平均）
        historical_spike_avg = 5.0  # 占位
        spike_growth = (signal.spike.recent_spike_count - historical_spike_avg) / max(historical_spike_avg, 1)
        self.trend_data["spike_growth"].append(spike_growth)

        # Gating 频率（近期 gating 事件数 / 时间窗口）
        recent_gating_count = len(signal.gating_events)
        gating_frequency = min(1.0, recent_gating_count / 10.0)  # 归一化到 [0, 1]
        self.trend_data["gating_frequency"].append(gating_frequency)

        # 波动率
        self.trend_data["volatility"].append(signal.stability.volatility_20d)

    def _calculate_risk_score(self, signal: RiskSignal) -> float:
        """计算风险评分（0-100）

        Args:
            signal: 风险信号

        Returns:
            风险评分（越高越危险）
        """
        score = 0.0

        # 1. Envelope 利用率 (0-40 分)
        envelope_usage = self.trend_data["envelope_usage"][-1] if self.trend_data["envelope_usage"] else 0.0
        envelope_score = envelope_usage * 40 * self.config.envelope_weight
        score += envelope_score

        # 2. Spike 增长率 (0-30 分)
        spike_growth = self.trend_data["spike_growth"][-1] if self.trend_data["spike_growth"] else 0.0
        spike_growth_normalized = max(0, min(1, spike_growth))  # 归一化到 [0, 1]
        spike_score = spike_growth_normalized * 30 * self.config.spike_weight
        score += spike_score

        # 3. Gating 频率 (0-20 分)
        gating_frequency = self.trend_data["gating_frequency"][-1] if self.trend_data["gating_frequency"] else 0.0
        gating_score = gating_frequency * 20 * self.config.gating_weight
        score += gating_score

        # 4. 波动率 (0-10 分)
        volatility = signal.stability.volatility_20d
        volatility_normalized = min(1.0, volatility / 0.05)  # 5% 波动率为最大
        volatility_score = volatility_normalized * 10 * self.config.volatility_weight
        score += volatility_score

        return min(100, score)

    def _determine_state(self, risk_score: float, signal: RiskSignal) -> RiskState:
        """根据风险评分判定状态

        Args:
            risk_score: 风险评分
            signal: 风险信号

        Returns:
            RiskState
        """
        # CRITICAL: 评分 > 70 或有硬性触发条件
        if risk_score > 70:
            return RiskState.CRITICAL

        # 检查硬性触发条件
        if (signal.drawdown.current_drawdown > 0.08 or  # 8% 回撤
            signal.spike.current_consecutive_losses > 5 or  # 连续 5 次亏损
            len([e for e in signal.gating_events if e.action in ["GATE", "DISABLE"]]) > 0):
            return RiskState.CRITICAL

        # WARNING: 评分 > 40 或有警告条件
        if risk_score > 40:
            return RiskState.WARNING

        # 检查警告条件
        if (signal.drawdown.current_drawdown > 0.05 or  # 5% 回撤
            signal.spike.recent_spike_count > 3 or  # 近期 3 次尖刺
            signal.stability.stability_score < 60):
            return RiskState.WARNING

        # NORMAL: 其他情况
        return RiskState.NORMAL

    def _transition_to(
        self,
        new_state: RiskState,
        signal: RiskSignal,
        risk_score: float
    ) -> StateTransitionEvent:
        """转换到新状态

        Args:
            new_state: 新状态
            signal: 风险信号
            risk_score: 风险评分

        Returns:
            StateTransitionEvent
        """
        old_state = self.current_state

        # 确定转换类型
        if new_state.value > old_state.value:
            # 定义顺序: NORMAL=0, WARNING=1, CRITICAL=2
            state_order = {RiskState.NORMAL: 0, RiskState.WARNING: 1, RiskState.CRITICAL: 2}
            if state_order[new_state] > state_order[old_state]:
                transition_type = StateTransition.UPGRADE
            else:
                transition_type = StateTransition.DOWNGRADE
        else:
            transition_type = StateTransition.DOWNGRADE

        # 确定触发指标
        trigger_metric = "risk_score"
        trigger_value = risk_score
        threshold = 50.0  # 中间阈值

        # 找到具体触发的指标
        envelope_usage = self.trend_data["envelope_usage"][-1] if self.trend_data["envelope_usage"] else 0.0
        if envelope_usage > self.config.warning_envelope_usage:
            trigger_metric = "envelope_usage"
            trigger_value = envelope_usage
            threshold = self.config.warning_envelope_usage

        # 创建转换事件
        transition = StateTransitionEvent(
            timestamp=datetime.now(),
            from_state=old_state,
            to_state=new_state,
            transition_type=transition_type,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            threshold=threshold,
            strategy_id=self.strategy_id,
            context={
                "risk_score": risk_score,
                "signal": signal.to_dict()
            }
        )

        # 更新状态
        self.current_state = new_state
        self.state_since = datetime.now()
        self.transition_history.append(transition)

        return transition

    def get_current_state(self) -> RiskState:
        """获取当前状态"""
        return self.current_state

    def get_state_duration(self) -> timedelta:
        """获取当前状态持续时长"""
        return datetime.now() - self.state_since

    def get_transition_history(
        self,
        limit: int = 10
    ) -> List[StateTransitionEvent]:
        """获取状态转换历史

        Args:
            limit: 返回最近的 N 次转换

        Returns:
            转换事件列表
        """
        return self.transition_history[-limit:]


class PortfolioRiskStateMachine:
    """组合级风险状态机"""

    def __init__(self, strategy_ids: List[str]):
        """初始化组合状态机

        Args:
            strategy_ids: 策略 ID 列表
        """
        self.strategy_ids = strategy_ids
        self.state_machines: Dict[str, RiskStateMachine] = {}

        # 为每个策略创建状态机
        for strategy_id in strategy_ids:
            self.state_machines[strategy_id] = RiskStateMachine(strategy_id)

        # 组合级状态（综合所有策略）
        self.portfolio_state: RiskState = RiskState.NORMAL

    def update_strategy_state(
        self,
        strategy_id: str,
        signal: RiskSignal
    ) -> Optional[StateTransitionEvent]:
        """更新单个策略的状态

        Args:
            strategy_id: 策略 ID
            signal: 风险信号

        Returns:
            StateTransitionEvent（如果发生转换）
        """
        if strategy_id not in self.state_machines:
            self.state_machines[strategy_id] = RiskStateMachine(strategy_id)

        machine = self.state_machines[strategy_id]
        return machine.update_state(signal)

    def get_portfolio_state(self) -> RiskState:
        """获取组合级状态

        策略：
        - 任一策略 CRITICAL → 组合 CRITICAL
        - 任一策略 WARNING → 组合 WARNING
        - 全部 NORMAL → 组合 NORMAL
        """
        states = [machine.get_current_state() for machine in self.state_machines.values()]

        if RiskState.CRITICAL in states:
            return RiskState.CRITICAL
        elif RiskState.WARNING in states:
            return RiskState.WARNING
        else:
            return RiskState.NORMAL

    def get_all_states(self) -> Dict[str, RiskState]:
        """获取所有策略的状态"""
        return {
            strategy_id: machine.get_current_state()
            for strategy_id, machine in self.state_machines.items()
        }
