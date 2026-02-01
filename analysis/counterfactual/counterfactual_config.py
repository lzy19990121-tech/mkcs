"""
SPL-7b-A: 反事实输入定义

定义可切换的反事实维度和统一接口。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pathlib import Path
import json
import yaml

import sys
import os
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.replay_schema import ReplayOutput


class CounterfactualDimension(Enum):
    """反事实维度"""
    GATING_THRESHOLD = "gating_threshold"       # Gating 阈值
    RULE_ENABLED = "rule_enabled"               # 规则开关
    ALLOCATOR_TYPE = "allocator_type"           # Allocator 类型
    PORTFOLIO_COMPOSITION = "portfolio_composition"  # 组合成分
    RISK_BUDGET = "risk_budget"                 # 风险预算


@dataclass
class GatingThresholdConfig:
    """Gating 阈值配置"""
    baseline: float              # 基线阈值
    earlier: float               # 更早（更保守）
    later: float                # 更晚（更宽松）
    stronger: float             # 更强（更保守）
    weaker: float               # 更弱（更宽松）

    def to_dict(self) -> Dict[str, float]:
        return {
            "baseline": self.baseline,
            "earlier": self.earlier,
            "later": self.later,
            "stronger": self.stronger,
            "weaker": self.weaker
        }


@dataclass
class RuleConfig:
    """规则配置"""
    rule_id: str
    enabled: bool
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "enabled": self.enabled,
            "parameters": self.parameters
        }


@dataclass
class AllocatorConfig:
    """Allocator 配置"""
    allocator_type: str  # "rules", "optimizer_v2", "equal_weight"

    # 参数
    rules_config: Optional[Dict[str, Any]] = None
    optimizer_config: Optional[Dict[str, Any]] = None

    # 组合权重（如果固定）
    fixed_weights: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocator_type": self.allocator_type,
            "rules_config": self.rules_config,
            "optimizer_config": self.optimizer_config,
            "fixed_weights": self.fixed_weights
        }


@dataclass
class PortfolioComposition:
    """组合成分配置"""
    strategy_ids: List[str]
    weights: Dict[str, float]  # strategy_id -> weight

    # 排除的策略
    excluded_strategies: List[str] = field(default_factory=list)

    # 额外加入的策略（如果有）
    additional_strategies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_ids": self.strategy_ids,
            "weights": self.weights,
            "excluded_strategies": self.excluded_strategies,
            "additional_strategies": self.additional_strategies
        }


@dataclass
class CounterfactualScenario:
    """反事实场景定义"""
    scenario_id: str
    name: str
    description: str

    # 维度配置
    gating_thresholds: Optional[Dict[str, GatingThresholdConfig]] = None
    rule_configs: Optional[Dict[str, RuleConfig]] = None
    allocator_config: Optional[AllocatorConfig] = None
    portfolio_composition: Optional[PortfolioComposition] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "gating_thresholds": {
                k: v.to_dict() for k, v in (self.gating_thresholds or {}).items()
            },
            "rule_configs": {
                k: v.to_dict() for k, v in (self.rule_configs or {}).items()
            },
            "allocator_config": self.allocator_config.to_dict() if self.allocator_config else None,
            "portfolio_composition": self.portfolio_composition.to_dict() if self.portfolio_composition else None
        }


@dataclass
class CounterfactualInput:
    """反事实输入（统一接口）

    输入：replay + decision config
    输出：PnL + risk metrics + events
    """
    # Replay 数据
    replay: Union[ReplayOutput, str]  # ReplayOutput 对象或文件路径

    # 场景配置
    scenario: CounterfactualScenario

    # 输入验证
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "replay": str(self.replay) if isinstance(self.replay, ReplayOutput) else self.replay,
            "scenario": self.scenario.to_dict(),
            "validated": self.validated,
            "validation_errors": self.validation_errors
        }


@dataclass
class CounterfactualResult:
    """反事实运行结果

    输出：PnL + risk metrics + events
    """
    scenario_id: str
    strategy_id: str

    # 收益指标
    total_return: float
    daily_returns: List[float]

    # 风险指标
    max_drawdown: float
    volatility: float
    cvar_95: float
    cvar_99: float

    # 事件统计
    gating_events_count: int
    allocator_rebalance_count: int
    state_transitions: List[Dict[str, Any]]

    # 执行信息
    execution_time: float
    success: bool
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "scenario_id": self.scenario_id,
            "strategy_id": self.strategy_id,
            "total_return": self.total_return,
            "daily_returns": self.daily_returns,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "gating_events_count": self.gating_events_count,
            "allocator_rebalance_count": self.allocator_rebalance_count,
            "state_transitions": self.state_transitions,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message
        }


class CounterfactualScenarioLibrary:
    """反事实场景库

    预定义常用的反事实场景。
    """

    @staticmethod
    def create_actual_scenario() -> CounterfactualScenario:
        """创建 Actual 场景（真实发生）"""
        return CounterfactualScenario(
            scenario_id="actual",
            name="Actual (真实发生)",
            description="实际运行的场景，作为对照基线"
        )

    @staticmethod
    def create_earlier_gating_scenario() -> CounterfactualScenario:
        """创建更早 gating 场景"""
        return CounterfactualScenario(
            scenario_id="cf_earlier_gating",
            name="CF-A: 更早 Gating",
            description="更保守的 gating 阈值，更早触发风控",
            gating_thresholds={
                "stability_score": GatingThresholdConfig(
                    baseline=50.0,
                    earlier=70.0,   # 提高到 70（更容易触发）
                    later=50.0,
                    stronger=80.0,
                    weaker=40.0
                )
            }
        )

    @staticmethod
    def create_later_gating_scenario() -> CounterfactualScenario:
        """创建更晚 gating 场景"""
        return CounterfactualScenario(
            scenario_id="cf_later_gating",
            name="CF-B: 更晚 Gating",
            description="更宽松的 gating 阈值，更晚触发风控",
            gating_thresholds={
                "stability_score": GatingThresholdConfig(
                    baseline=50.0,
                    earlier=70.0,
                    later=30.0,    # 降低到 30（更难触发）
                    stronger=80.0,
                    weaker=20.0
                )
            }
        )

    @staticmethod
    def create_no_gating_scenario() -> CounterfactualScenario:
        """创建无 gating 场景"""
        return CounterfactualScenario(
            scenario_id="cf_no_gating",
            name="CF-C: 无 Gating",
            description="完全禁用 gating",
            rule_configs={
                "stability_gating": RuleConfig(
                    rule_id="stability_gating",
                    enabled=False
                ),
                "volatility_gating": RuleConfig(
                    rule_id="volatility_gating",
                    enabled=False
                )
            }
        )

    @staticmethod
    def create_optimizer_scenario() -> CounterfactualScenario:
        """创建 Optimizer 场景"""
        return CounterfactualScenario(
            scenario_id="cf_optimizer",
            name="CF-D: Optimizer Allocator",
            description="使用 SPL-6b 优化器分配",
            allocator_config=AllocatorConfig(
                allocator_type="optimizer_v2",
                optimizer_config={
                    "method": "quadratic_programming",
                    "risk_budget": {
                        "cvar_95": -0.10,
                        "max_drawdown": 0.12
                    }
                }
            )
        )

    @staticmethod
    def create_equal_weight_scenario() -> CounterfactualScenario:
        """创建等权重场景"""
        return CounterfactualScenario(
            scenario_id="cf_equal_weight",
            name="CF-E: 等权重组合",
            description="所有策略等权重分配",
            allocator_config=AllocatorConfig(
                allocator_type="equal_weight",
                fixed_weights={}  # 运行时计算
            )
        )

    @staticmethod
    def create_exclude_strategy_scenario(excluded_strategy: str) -> CounterfactualScenario:
        """创建排除策略场景"""
        return CounterfactualScenario(
            scenario_id=f"cf_exclude_{excluded_strategy}",
            name=f"CF-F: 排除 {excluded_strategy}",
            description=f"从组合中排除 {excluded_strategy}",
            portfolio_composition=PortfolioComposition(
                strategy_ids=[],
                excluded_strategies=[excluded_strategy]
            )
        )

    @classmethod
    def get_all_scenarios(cls, strategy_ids: List[str]) -> List[CounterfactualScenario]:
        """获取所有预定义场景

        Args:
            strategy_ids: 策略 ID 列表

        Returns:
            场景列表
        """
        scenarios = [
            cls.create_actual_scenario(),
            cls.create_earlier_gating_scenario(),
            cls.create_later_gating_scenario(),
            cls.create_no_gating_scenario(),
            cls.create_optimizer_scenario(),
            cls.create_equal_weight_scenario()
        ]

        # 为每个策略生成排除场景
        for strategy_id in strategy_ids:
            scenarios.append(cls.create_exclude_strategy_scenario(strategy_id))

        return scenarios


class CounterfactualConfigValidator:
    """反事实配置验证器"""

    @staticmethod
    def validate_input(input_config: CounterfactualInput) -> bool:
        """验证反事实输入

        Args:
            input_config: 反事实输入

        Returns:
            是否有效
        """
        errors = []

        # 验证 replay
        replay_path = input_config.replay
        if isinstance(replay_path, str):
            if not Path(replay_path).exists():
                errors.append(f"Replay 文件不存在: {replay_path}")

        # 验证场景
        scenario = input_config.scenario

        # 检查至少有一个维度被修改
        has_modification = (
            scenario.gating_thresholds is not None or
            scenario.rule_configs is not None or
            scenario.allocator_config is not None or
            scenario.portfolio_composition is not None
        )

        if scenario.scenario_id != "actual" and not has_modification:
            errors.append("反事实场景必须至少修改一个维度")

        # 验证 allocator 配置
        if scenario.allocator_config:
            allocator_type = scenario.allocator_config.allocator_type
            if allocator_type not in ["rules", "optimizer_v2", "equal_weight"]:
                errors.append(f"不支持的 allocator 类型: {allocator_type}")

        # 更新验证状态
        input_config.validated = len(errors) == 0
        input_config.validation_errors = errors

        return input_config.validated


def load_scenario_from_yaml(
    yaml_path: str,
    scenario_id: str
) -> CounterfactualScenario:
    """从 YAML 加载场景

    Args:
        yaml_path: YAML 文件路径
        scenario_id: 场景 ID

    Returns:
        CounterfactualScenario
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    scenarios_config = config.get("scenarios", [])
    scenario_data = None

    for s in scenarios_config:
        if s.get("scenario_id") == scenario_id:
            scenario_data = s
            break

    if not scenario_data:
        raise ValueError(f"场景不存在: {scenario_id}")

    # 构建场景对象
    return CounterfactualScenario(
        scenario_id=scenario_data["scenario_id"],
        name=scenario_data["name"],
        description=scenario_data["description"],
        gating_thresholds=scenario_data.get("gating_thresholds"),
        rule_configs=scenario_data.get("rule_configs"),
        allocator_config=scenario_data.get("allocator_config"),
        portfolio_composition=scenario_data.get("portfolio_composition")
    )


def save_scenario_to_yaml(
    scenario: CounterfactualScenario,
    yaml_path: str
):
    """保存场景到 YAML

    Args:
        scenario: 场景定义
        yaml_path: YAML 文件路径
    """
    # 加载现有配置（如果存在）
    existing_scenarios = []
    if Path(yaml_path).exists():
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
            existing_scenarios = config.get("scenarios", [])

    # 添加新场景
    existing_scenarios.append(scenario.to_dict())

    # 保存
    with open(yaml_path, 'w') as f:
        yaml.dump({"scenarios": existing_scenarios}, f, default_flow_style=False)


if __name__ == "__main__":
    """测试反事实输入定义"""
    print("=== SPL-7b-A: 反事实输入定义测试 ===\n")

    # 获取所有场景
    strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]
    scenarios = CounterfactualScenarioLibrary.get_all_scenarios(strategy_ids)

    print(f"预定义场景数量: {len(scenarios)}\n")
    for scenario in scenarios:
        print(f"- {scenario.scenario_id}: {scenario.name}")

    print("\n✅ 反事实输入定义测试通过")
