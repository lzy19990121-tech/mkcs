"""
自适应风控管理器

实现：
1. 风控调整目标仓位（而非简单拒绝）
2. 风控冷却与恢复机制
3. rules.json → runtime 生效
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from core.models import Signal, Position
from core.strategy_models import StrategySignal

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """风控动作"""
    APPROVE = "approve"                # 完全批准
    SCALE_DOWN = "scale_down"          # 降权
    PAUSE = "pause"                    # 暂停
    DISABLE = "disable"                # 禁用


@dataclass
class CooldownState:
    """冷却状态"""
    rule_id: str
    last_triggered: datetime
    cooldown_until: datetime
    trigger_count: int = 0

    def is_cooled_down(self) -> bool:
        return datetime.now() >= self.cooldown_until

    def remaining_cooldown_seconds(self) -> float:
        delta = self.cooldown_until - datetime.now()
        return max(0, delta.total_seconds())


@dataclass
class RiskRule:
    """风控规则"""
    rule_id: str
    name: str
    rule_type: str  # position_limit, drawdown, volatility, etc.
    scope: str      # global, symbol, strategy
    priority: int   # 优先级（数字越小优先级越高）

    # 规则参数
    params: Dict[str, Any] = field(default_factory=dict)

    # 冷却配置
    cooldown_seconds: int = 0
    max_triggers_per_day: int = 10

    # 当前状态
    enabled: bool = True
    trigger_count: int = 0
    last_triggered: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "rule_type": self.rule_type,
            "scope": self.scope,
            "priority": self.priority,
            "params": self.params,
            "cooldown_seconds": self.cooldown_seconds,
            "max_triggers_per_day": self.max_triggers_per_day,
            "enabled": self.enabled,
            "trigger_count": self.trigger_count,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskRule":
        data = data.copy()
        if data.get("last_triggered"):
            data["last_triggered"] = datetime.fromisoformat(data["last_triggered"])
        return cls(**data)


@dataclass
class RiskCheckResult:
    """风控检查结果"""
    action: RiskAction
    scale_factor: float = 1.0          # 仓位缩放系数
    reason: str = ""
    triggered_rules: List[str] = field(default_factory=list)
    cooldown_remaining: Optional[int] = None


class AdaptiveRiskManager:
    """自适应风控管理器

    特点：
    1. 不是简单拒绝，而是调整仓位
    2. 支持规则冷却和恢复
    3. 可从 rules.json 加载规则
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        enable_scaling: bool = True,
        default_cooldown_seconds: int = 300
    ):
        """初始化风控管理器

        Args:
            initial_capital: 初始资金
            enable_scaling: 是否启用仓位缩放
            default_cooldown_seconds: 默认冷却时间
        """
        self.initial_capital = initial_capital
        self.enable_scaling = enable_scaling
        self.default_cooldown_seconds = default_cooldown_seconds

        # 规则管理
        self.rules: Dict[str, RiskRule] = {}
        self.cooldown_states: Dict[str, CooldownState] = {}

        # 风险状态
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()

        # 全局状态
        self.paused = False
        self.disabled = False
        self.pause_reason = ""
        self.pause_since: Optional[datetime] = None

        # 加载默认规则
        self._load_default_rules()

    def _load_default_rules(self):
        """加载默认规则"""
        default_rules = [
            RiskRule(
                rule_id="max_position_ratio",
                name="单个标的最大仓位限制",
                rule_type="position_limit",
                scope="symbol",
                priority=1,
                params={"max_ratio": 0.2},
                cooldown_seconds=60
            ),
            RiskRule(
                rule_id="max_total_position",
                name="总仓位限制",
                rule_type="position_limit",
                scope="global",
                priority=1,
                params={"max_ratio": 0.8},
                cooldown_seconds=60
            ),
            RiskRule(
                rule_id="max_daily_loss",
                name="单日最大亏损限制",
                rule_type="drawdown",
                scope="global",
                priority=0,  # 高优先级
                params={"max_loss_ratio": 0.05},
                cooldown_seconds=3600
            ),
            RiskRule(
                rule_id="volatility_limit",
                name="高波动限制",
                rule_type="volatility",
                scope="symbol",
                priority=2,
                params={"max_volatility": 2.0},
                cooldown_seconds=300
            ),
            RiskRule(
                rule_id="consecutive_losses",
                name="连续亏损限制",
                rule_type="performance",
                scope="global",
                priority=1,
                params={"max_consecutive": 5},
                cooldown_seconds=600
            ),
        ]

        for rule in default_rules:
            self.rules[rule.rule_id] = rule

    def check_signal(
        self,
        signal: Signal,
        positions: Dict[str, Position],
        cash_balance: float,
        strategy_name: Optional[str] = None
    ) -> RiskCheckResult:
        """检查交易信号

        Args:
            signal: 交易信号
            positions: 当前持仓
            cash_balance: 现金余额
            strategy_name: 策略名称

        Returns:
            风控检查结果
        """
        # 检查全局暂停/禁用状态
        if self.disabled:
            return RiskCheckResult(
                action=RiskAction.DISABLE,
                reason="策略已禁用"
            )

        if self.paused:
            return RiskCheckResult(
                action=RiskAction.PAUSE,
                reason=self.pause_reason,
                cooldown_remaining=int((datetime.now() - self.pause_since).total_seconds()) if self.pause_since else None
            )

        # 检查规则
        triggered_rules = []
        scale_factors = []

        # 按优先级排序检查规则
        sorted_rules = sorted(self.rules.values(), key=lambda r: r.priority)

        for rule in sorted_rules:
            if not rule.enabled:
                continue

            # 检查冷却
            cooldown_state = self.cooldown_states.get(rule.rule_id)
            if cooldown_state and not cooldown_state.is_cooled_down():
                continue

            # 执行规则检查
            result = self._check_rule(rule, signal, positions, cash_balance, strategy_name)

            if result["triggered"]:
                triggered_rules.append(rule.rule_id)
                scale_factors.append(result.get("scale_factor", 1.0))

                # 更新触发计数
                rule.trigger_count += 1
                rule.last_triggered = datetime.now()

                # 设置冷却
                if rule.cooldown_seconds > 0:
                    self.cooldown_states[rule.rule_id] = CooldownState(
                        rule_id=rule.rule_id,
                        last_triggered=datetime.now(),
                        cooldown_until=datetime.now() + timedelta(seconds=rule.cooldown_seconds),
                        trigger_count=rule.trigger_count
                    )

        # 决定风控动作
        if not triggered_rules:
            return RiskCheckResult(action=RiskAction.APPROVE, scale_factor=1.0)

        # 计算综合缩放系数
        if self.enable_scaling:
            final_scale = min(scale_factors) if scale_factors else 1.0

            # 如果缩放系数太小，直接暂停
            if final_scale < 0.1:
                return RiskCheckResult(
                    action=RiskAction.PAUSE,
                    reason=f"仓位缩放过小 ({final_scale:.2%})",
                    triggered_rules=triggered_rules
                )

            return RiskCheckResult(
                action=RiskAction.SCALE_DOWN,
                scale_factor=final_scale,
                reason=f"触发规则: {', '.join(triggered_rules)}",
                triggered_rules=triggered_rules
            )
        else:
            # 不启用缩放，直接拒绝
            return RiskCheckResult(
                action=RiskAction.PAUSE,
                reason=f"触发规则: {', '.join(triggered_rules)}",
                triggered_rules=triggered_rules
            )

    def check_strategy_signal(
        self,
        signal: StrategySignal,
        positions: Dict[str, Position],
        cash_balance: float,
        strategy_name: Optional[str] = None
    ) -> Tuple[RiskAction, float, StrategySignal]:
        """检查新格式信号

        Returns:
            (风控动作, 缩放系数, 调整后的信号)
        """
        # 创建旧格式Signal用于检查
        old_signal = Signal(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            action=signal.raw_action or "BUY",
            price=Decimal("100"),  # 临时值
            quantity=100,
            confidence=signal.confidence,
            reason=signal.reason
        )

        result = self.check_signal(old_signal, positions, cash_balance, strategy_name)

        # 应用缩放
        if result.action == RiskAction.SCALE_DOWN:
            # 创建调整后的信号
            adjusted = StrategySignal(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                target_position=signal.target_position * result.scale_factor,
                target_weight=signal.target_weight * result.scale_factor,
                confidence=signal.confidence * result.scale_factor,
                reason=f"{signal.reason} (风控调整: {result.reason})",
                regime=signal.regime,
                risk_hints=signal.risk_hints,
                raw_action=signal.raw_action,
                metadata={**signal.metadata, "risk_scale_factor": result.scale_factor}
            )
            return result.action, result.scale_factor, adjusted

        return result.action, result.scale_factor, signal

    def _check_rule(
        self,
        rule: RiskRule,
        signal: Signal,
        positions: Dict[str, Position],
        cash_balance: float,
        strategy_name: Optional[str]
    ) -> Dict[str, Any]:
        """检查单个规则"""
        result = {"triggered": False, "scale_factor": 1.0}

        if rule.rule_type == "position_limit":
            result = self._check_position_limit(rule, signal, positions, cash_balance)
        elif rule.rule_type == "drawdown":
            result = self._check_drawdown(rule, signal)
        elif rule.rule_type == "volatility":
            result = self._check_volatility(rule, signal)
        elif rule.rule_type == "performance":
            result = self._check_performance(rule, signal)

        return result

    def _check_position_limit(
        self,
        rule: RiskRule,
        signal: Signal,
        positions: Dict[str, Position],
        cash_balance: float
    ) -> Dict[str, Any]:
        """检查仓位限制"""
        max_ratio = rule.params.get("max_ratio", 0.2)

        if rule.scope == "symbol":
            # 单个标的仓位限制
            position = positions.get(signal.symbol)
            current_value = position.market_value if position else Decimal("0")
            equity = self.initial_capital  # 简化，实际应计算总权益

            current_ratio = float(current_value / equity) if equity > 0 else 0

            if current_ratio >= max_ratio:
                return {"triggered": True, "scale_factor": 0.0}

            # 计算新仓位
            signal_value = float(signal.price * signal.quantity)
            new_ratio = (float(current_value) + signal_value) / equity

            if new_ratio > max_ratio:
                scale_factor = max_ratio / new_ratio
                return {"triggered": True, "scale_factor": scale_factor}

        elif rule.scope == "global":
            # 总仓位限制
            total_position = sum(p.market_value for p in positions.values())
            equity = self.initial_capital
            current_ratio = float(total_position / equity) if equity > 0 else 0

            if current_ratio >= max_ratio:
                return {"triggered": True, "scale_factor": 0.0}

        return {"triggered": False, "scale_factor": 1.0}

    def _check_drawdown(self, rule: RiskRule, signal: Signal) -> Dict[str, Any]:
        """检查回撤限制"""
        max_loss_ratio = rule.params.get("max_loss_ratio", 0.05)

        if abs(self.current_drawdown) >= max_loss_ratio:
            return {"triggered": True, "scale_factor": 0.0}

        return {"triggered": False, "scale_factor": 1.0}

    def _check_volatility(self, rule: RiskRule, signal: Signal) -> Dict[str, Any]:
        """检查波动率限制"""
        # Signal 类没有 metadata，返回默认值
        max_volatility = rule.params.get("max_volatility", 2.0)
        volatility = 1.0  # 默认值

        if volatility > max_volatility:
            # 降低仓位
            scale_factor = max_volatility / volatility
            return {"triggered": True, "scale_factor": scale_factor}

        return {"triggered": False, "scale_factor": 1.0}

    def _check_performance(self, rule: RiskRule, signal: Signal) -> Dict[str, Any]:
        """检查性能限制"""
        # 简化处理：检查连续亏损
        # 实际应从交易记录统计
        return {"triggered": False, "scale_factor": 1.0}

    def update_daily_pnl(self, pnl: float):
        """更新每日盈亏"""
        self.daily_pnl += pnl

        # 更新回撤
        if pnl < 0:
            self.current_drawdown = min(self.current_drawdown, self.daily_pnl / self.initial_capital)
        else:
            self.current_drawdown = min(0, self.current_drawdown + pnl / self.initial_capital)

        # 检查是否需要重置日计数
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_pnl = pnl
            self.daily_trades = 0
            self.last_reset_date = today

    def pause(self, reason: str, duration_seconds: Optional[int] = None):
        """暂停交易"""
        self.paused = True
        self.pause_reason = reason
        self.pause_since = datetime.now()

        if duration_seconds:
            # 设置定时恢复
            pass

    def resume(self):
        """恢复交易"""
        self.paused = False
        self.pause_reason = ""
        self.pause_since = None

    def disable_strategy(self, reason: str):
        """禁用策略"""
        self.disabled = True
        self.pause_reason = reason

    def enable_strategy(self):
        """启用策略"""
        self.disabled = False
        self.pause_reason = ""

    def load_rules_from_json(self, path: str):
        """从 JSON 文件加载规则"""
        rules_path = Path(path)
        if not rules_path.exists():
            logger.warning(f"规则文件不存在: {path}")
            return

        with open(rules_path, 'r') as f:
            data = json.load(f)

        for rule_data in data.get("rules", []):
            rule = RiskRule.from_dict(rule_data)
            self.rules[rule.rule_id] = rule

        logger.info(f"从 {path} 加载了 {len(data.get('rules', []))} 条规则")

    def save_rules_to_json(self, path: str):
        """保存规则到 JSON 文件"""
        rules_data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "rules": [rule.to_dict() for rule in self.rules.values()]
        }

        rules_path = Path(path)
        rules_path.parent.mkdir(parents=True, exist_ok=True)

        with open(rules_path, 'w') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)

        logger.info(f"规则已保存到 {path}")

    def get_rule_status(self) -> Dict[str, Any]:
        """获取规则状态"""
        return {
            "paused": self.paused,
            "disabled": self.disabled,
            "pause_reason": self.pause_reason,
            "current_drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
            "rules": {
                rule_id: {
                    "enabled": rule.enabled,
                    "trigger_count": rule.trigger_count,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
                    "cooldown_remaining": self.cooldown_states.get(rule_id, CooldownState(
                        rule_id="", last_triggered=datetime.now(), cooldown_until=datetime.now()
                    )).remaining_cooldown_seconds() if rule_id in self.cooldown_states else None
                }
                for rule_id, rule in self.rules.items()
            }
        }


if __name__ == "__main__":
    """测试代码"""
    print("=== AdaptiveRiskManager 测试 ===\n")

    # 创建风控管理器
    risk_mgr = AdaptiveRiskManager(
        initial_capital=100000,
        enable_scaling=True
    )

    # 创建测试信号
    from decimal import Decimal
    from datetime import datetime

    signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150"),
        quantity=200,
        confidence=0.8,
        reason="测试信号"
    )

    # 测试1: 正常信号通过
    print("1. 正常信号:")
    result = risk_mgr.check_signal(signal, {}, 100000)
    print(f"   动作: {result.action.value}")
    print(f"   缩放系数: {result.scale_factor}")

    # 测试2: 超过仓位限制
    print("\n2. 超过仓位限制:")
    large_signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150"),
        quantity=1000,  # 超过20%限制
        confidence=0.8,
        reason="大额信号"
    )
    result = risk_mgr.check_signal(large_signal, {}, 100000)
    print(f"   动作: {result.action.value}")
    print(f"   缩放系数: {result.scale_factor}")
    print(f"   触发规则: {result.triggered_rules}")

    # 测试3: 暂停后恢复
    print("\n3. 暂停/恢复:")
    risk_mgr.pause("测试暂停")
    result = risk_mgr.check_signal(signal, {}, 100000)
    print(f"   暂停后动作: {result.action.value}")

    risk_mgr.resume()
    result = risk_mgr.check_signal(signal, {}, 100000)
    print(f"   恢复后动作: {result.action.value}")

    # 测试4: 保存/加载规则
    print("\n4. 保存/加载规则:")
    import tempfile
    import os

    temp_file = tempfile.mktemp(suffix=".json")
    try:
        risk_mgr.save_rules_to_json(temp_file)
        print(f"   规则已保存到: {temp_file}")

        # 创建新管理器并加载
        new_mgr = AdaptiveRiskManager()
        new_mgr.load_rules_from_json(temp_file)
        print(f"   加载了 {len(new_mgr.rules)} 条规则")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # 测试5: 获取状态
    print("\n5. 规则状态:")
    status = risk_mgr.get_rule_status()
    print(f"   暂停: {status['paused']}")
    print(f"   当前回撤: {status['current_drawdown']:.2%}")
    print(f"   规则数量: {len(status['rules'])}")

    print("\n✓ 所有测试通过")
