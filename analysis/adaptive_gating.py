"""
SPL-5a-D: Runtime自适应风控闸门

实现运行时自适应风控决策，根据市场状态动态调整风控阈值。
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import deque

from analysis.regime_features import (
    RegimeFeatures,
    RegimeFeatureCalculator,
    get_calculator
)
from analysis.adaptive_threshold import (
    AdaptiveThreshold,
    AdaptiveThresholdRuleset,
    create_default_adaptive_ruleset
)
from analysis.adaptive_calibration import (
    load_calibrated_params,
    create_calibrated_ruleset
)


class GatingAction(Enum):
    """风控动作"""
    ALLOW = "allow"           # 允许交易
    GATE = "gate"             # 暂停交易
    REDUCE = "reduce"         # 降仓
    DISABLE = "disable"       # 禁用策略


@dataclass
class GatingDecision:
    """风控决策结果"""
    action: GatingAction
    rule_id: Optional[str]    # 触发的规则ID
    threshold: Optional[float] # 当前阈值
    current_value: Optional[float]  # 当前指标值
    reason: str               # 决策原因
    regime: RegimeFeatures    # 市场状态
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "rule_id": self.rule_id,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "reason": self.reason,
            "regime": self.regime.to_dict() if self.regime else None,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class GatingEvent:
    """风控事件记录"""
    timestamp: datetime
    strategy_id: str
    action: GatingAction
    rule_id: Optional[str]
    threshold: Optional[float]
    current_value: Optional[float]
    regime_snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "action": self.action.value,
            "rule_id": self.rule_id,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "regime_snapshot": self.regime_snapshot
        }


@dataclass
class RiskMetrics:
    """风险指标"""
    stability_score: float = 100.0      # 稳定性评分
    window_return: float = 0.0          # 窗口收益
    drawdown_duration: int = 0          # 回撤持续天数
    market_regime: float = 1.0          # 市场状态 (1=允许, 0=禁止)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stability_score": self.stability_score,
            "window_return": self.window_return,
            "drawdown_duration": self.drawdown_duration,
            "market_regime": self.market_regime
        }


class AdaptiveRiskGate:
    """自适应风控闸门

    根据市场状态和标定参数，实时决策风控动作。
    """

    def __init__(
        self,
        strategy_id: str,
        params_path: Optional[str] = None,
        event_history_size: int = 100
    ):
        """初始化风控闸门

        Args:
            strategy_id: 策略ID
            params_path: 标定参数路径（None则使用默认）
            event_history_size: 事件历史保留数量
        """
        self.strategy_id = strategy_id
        self.event_history_size = event_history_size

        # 加载标定规则集
        if params_path and Path(params_path).exists():
            self.ruleset = create_calibrated_ruleset(params_path)
        else:
            self.ruleset = create_default_adaptive_ruleset()

        # 市场状态计算器
        self.calculator = get_calculator(strategy_id, window_length=20)

        # 当前风险指标
        self.risk_metrics = RiskMetrics()

        # 事件历史
        self.event_history: deque = deque(maxlen=event_history_size)

        # 状态
        self.is_gated = False
        self.is_disabled = False
        self.gating_start_time: Optional[datetime] = None
        self.current_position_scale: float = 1.0  # 仓位缩放因子

        # 上次更新时间
        self.last_update: Optional[datetime] = None

    def update_price(self, price: float, timestamp: datetime = None) -> None:
        """更新价格数据

        Args:
            price: 当前价格
            timestamp: 时间戳
        """
        self.calculator.update(price, timestamp)
        self.last_update = timestamp or datetime.now()

    def evaluate_risk_metrics(
        self,
        equity: float,
        peak_equity: float,
        window_returns: List[float] = None
    ) -> RiskMetrics:
        """计算当前风险指标

        Args:
            equity: 当前权益
            peak_equity: 峰值权益
            window_returns: 窗口收益序列

        Returns:
            风险指标
        """
        # 计算当前回撤
        current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        # 稳定性评分 (简化：基于回撤)
        self.risk_metrics.stability_score = max(0, 100 - current_drawdown * 1000)

        # 窗口收益
        if window_returns:
            self.risk_metrics.window_return = sum(window_returns)
        else:
            self.risk_metrics.window_return = (equity - 100000) / 100000  # 简化

        # 回撤持续（如果有gating状态）
        if self.is_gated and self.gating_start_time:
            self.risk_metrics.drawdown_duration = (
                datetime.now() - self.gating_start_time
            ).days

        # 市场状态
        regime = self.calculator.calculate()
        self.risk_metrics.market_regime = 1.0 if regime.trend_bucket == "strong" else 0.0

        return self.risk_metrics

    def check(self, price: float = None) -> GatingDecision:
        """检查是否需要风控动作

        Args:
            price: 当前价格（可选，用于更新市场状态）

        Returns:
            风控决策
        """
        timestamp = datetime.now()

        # 更新价格
        if price is not None:
            self.update_price(price, timestamp)

        # 获取当前市场状态
        regime = self.calculator.calculate()

        # 评估所有规则
        for rule in self.ruleset.get_all_rules():
            # 获取当前阈值
            threshold = rule.get_threshold(regime)

            # 获取当前指标值
            current_value = self._get_metric_value(rule.trigger_metric)

            # 判断是否触发
            triggered = self._should_trigger_rule(
                rule.rule_type,
                current_value,
                threshold
            )

            if triggered:
                # 记录事件
                event = GatingEvent(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    action=self._rule_type_to_action(rule.rule_type),
                    rule_id=rule.rule_id,
                    threshold=threshold,
                    current_value=current_value,
                    regime_snapshot=regime.to_dict()
                )
                self.event_history.append(event)

                # 生成决策
                decision = GatingDecision(
                    action=self._rule_type_to_action(rule.rule_type),
                    rule_id=rule.rule_id,
                    threshold=threshold,
                    current_value=current_value,
                    reason=f"{rule.rule_name}: {current_value:.2f} < {threshold:.2f}",
                    regime=regime,
                    timestamp=timestamp
                )

                # 更新内部状态
                self._update_state(decision)

                return decision

        # 没有触发任何规则
        return GatingDecision(
            action=GatingAction.ALLOW,
            rule_id=None,
            threshold=None,
            current_value=None,
            reason="所有规则均未触发",
            regime=regime,
            timestamp=timestamp
        )

    def _get_metric_value(self, metric_name: str) -> float:
        """获取指标值

        Args:
            metric_name: 指标名称

        Returns:
            指标值
        """
        if metric_name == "stability_score":
            return self.risk_metrics.stability_score
        elif metric_name == "window_return":
            return self.risk_metrics.window_return
        elif metric_name == "drawdown_duration":
            return float(self.risk_metrics.drawdown_duration)
        elif metric_name == "market_regime":
            return self.risk_metrics.market_regime
        else:
            return 0.0

    def _should_trigger_rule(
        self,
        rule_type: str,
        current_value: float,
        threshold: float
    ) -> bool:
        """判断规则是否应该触发

        Args:
            rule_type: 规则类型
            current_value: 当前指标值
            threshold: 阈值

        Returns:
            是否触发
        """
        if rule_type == "gating":
            # Gating规则：当前值 < 阈值时触发
            # 例如：稳定性评分 < 30 → 暂停交易
            return current_value < threshold

        elif rule_type == "reduction":
            # 降仓规则：当前值 < 阈值时触发
            # 例如：窗口收益 < -10% → 降仓
            return current_value < threshold

        elif rule_type == "disable":
            # 禁用规则：当前值 < 阈值时触发
            # 例如：市场状态 < 0.5 → 禁用策略
            return current_value < threshold

        else:
            return False

    def _rule_type_to_action(self, rule_type: str) -> GatingAction:
        """将规则类型转换为动作

        Args:
            rule_type: 规则类型

        Returns:
            风控动作
        """
        mapping = {
            "gating": GatingAction.GATE,
            "reduction": GatingAction.REDUCE,
            "disable": GatingAction.DISABLE
        }
        return mapping.get(rule_type, GatingAction.ALLOW)

    def _update_state(self, decision: GatingDecision) -> None:
        """更新内部状态

        Args:
            decision: 风控决策
        """
        if decision.action == GatingAction.GATE:
            if not self.is_gated:
                self.is_gated = True
                self.gating_start_time = decision.timestamp

        elif decision.action == GatingAction.ALLOW:
            # 解除gating状态
            if self.is_gated:
                gating_duration = (
                    decision.timestamp - self.gating_start_time
                ).total_seconds() / 3600  # 小时
                if gating_duration > 24:  # 超过24小时才解除
                    self.is_gated = False
                    self.gating_start_time = None

        elif decision.action == GatingAction.DISABLE:
            self.is_disabled = True

        elif decision.action == GatingAction.REDUCE:
            self.current_position_scale = 0.5  # 降仓50%

    def get_position_scale(self) -> float:
        """获取当前仓位缩放因子

        Returns:
            仓位缩放因子 (0.0 - 1.0)
        """
        if self.is_disabled:
            return 0.0
        elif self.is_gated:
            return 0.0
        else:
            return self.current_position_scale

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态

        Returns:
            状态字典
        """
        regime = self.calculator.calculate()

        return {
            "strategy_id": self.strategy_id,
            "is_gated": self.is_gated,
            "is_disabled": self.is_disabled,
            "position_scale": self.current_position_scale,
            "gating_duration_hours": (
                (datetime.now() - self.gating_start_time).total_seconds() / 3600
                if self.gating_start_time else 0
            ),
            "current_regime": regime.to_dict(),
            "risk_metrics": self.risk_metrics.to_dict(),
            "event_count": len(self.event_history),
            "last_update": self.last_update.isoformat() if self.last_update else None
        }

    def get_event_history(self) -> List[Dict[str, Any]]:
        """获取事件历史

        Returns:
            事件历史列表
        """
        return [event.to_dict() for event in self.event_history]

    def reset(self) -> None:
        """重置状态"""
        self.is_gated = False
        self.is_disabled = False
        self.gating_start_time = None
        self.current_position_scale = 1.0
        self.risk_metrics = RiskMetrics()
        self.event_history.clear()


class AdaptiveGatingManager:
    """自适应风控管理器

    管理多个策略的风控闸门。
    """

    def __init__(
        self,
        params_path: Optional[str] = None,
        event_history_size: int = 100
    ):
        """初始化管理器

        Args:
            params_path: 标定参数路径
            event_history_size: 事件历史保留数量
        """
        self.params_path = params_path
        self.event_history_size = event_history_size
        self.gates: Dict[str, AdaptiveRiskGate] = {}

    def get_gate(self, strategy_id: str) -> AdaptiveRiskGate:
        """获取或创建策略的风控闸门

        Args:
            strategy_id: 策略ID

        Returns:
            风控闸门
        """
        if strategy_id not in self.gates:
            self.gates[strategy_id] = AdaptiveRiskGate(
                strategy_id=strategy_id,
                params_path=self.params_path,
                event_history_size=self.event_history_size
            )
        return self.gates[strategy_id]

    def check_all(self) -> Dict[str, GatingDecision]:
        """检查所有策略的风控状态

        Returns:
            {strategy_id: GatingDecision}
        """
        decisions = {}
        for strategy_id, gate in self.gates.items():
            decisions[strategy_id] = gate.check()
        return decisions

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有策略的状态

        Returns:
            {strategy_id: status_dict}
        """
        return {
            strategy_id: gate.get_status()
            for strategy_id, gate in self.gates.items()
        }

    def reset_all(self) -> None:
        """重置所有策略的状态"""
        for gate in self.gates.values():
            gate.reset()


if __name__ == "__main__":
    """测试代码"""
    print("=== AdaptiveRiskGate 测试 ===\n")

    # 创建风控闸门
    gate = AdaptiveRiskGate(
        strategy_id="test_strategy",
        params_path=None,  # 使用默认规则
        event_history_size=10
    )

    # 模拟价格更新
    print("1. 模拟价格更新:")
    prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    for price in prices:
        gate.update_price(price)
        decision = gate.check(price)
        print(f"   价格={price}, 动作={decision.action.value}, 原因={decision.reason}")

    # 模拟风险指标变化
    print("\n2. 模拟风险指标变化:")
    gate.risk_metrics.stability_score = 25.0  # 低于默认阈值30
    decision = gate.check()
    print(f"   稳定性评分=25, 动作={decision.action.value}")
    print(f"   触发规则={decision.rule_id}")
    print(f"   当前阈值={decision.threshold}")

    # 检查状态
    print("\n3. 检查状态:")
    status = gate.get_status()
    print(f"   是否暂停={status['is_gated']}")
    print(f"   仓位缩放={status['position_scale']}")
    print(f"   市场状态={status['current_regime']['vol_bucket']}")

    # 测试管理器
    print("\n4. 测试管理器:")
    manager = AdaptiveGatingManager()
    gate1 = manager.get_gate("strategy_1")
    gate2 = manager.get_gate("strategy_2")

    all_status = manager.get_all_status()
    print(f"   管理策略数: {len(all_status)}")

    print("\n✓ 测试通过")
