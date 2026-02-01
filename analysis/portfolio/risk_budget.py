"""
SPL-5b-A: 组合风险预算定义

定义组合层面的风险预算约束，用于动态分配策略权重。
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from enum import Enum


class BudgetUnit(Enum):
    """预算单位"""
    TAIL_LOSS_CONTRIBUTION = "tail_loss_contribution"  # 尾部损失贡献
    RISK_WEIGHT = "risk_weight"  # 风险权重
    VAR_CONTRIBUTION = "var_contribution"  # VaR贡献


@dataclass
class PortfolioRiskBudget:
    """组合风险预算

    定义组合层面的��约束，所有策略分配必须满足这些约束。
    """

    # 组合级硬约束
    budget_return_p95: float     # P95最坏收益阈值（负数，如-0.10表示-10%）
    budget_mdd_p95: float        # P95最大回撤阈值（正数，如0.15表示15%）
    budget_duration_p95: int     # P95回撤持续阈值（天数，如30天）

    # 策略级预算单位
    budget_unit: BudgetUnit = BudgetUnit.TAIL_LOSS_CONTRIBUTION

    # 预算分配选项
    min_weight_per_strategy: float = 0.05    # 单策略最小权重（5%）
    max_weight_per_strategy: float = 0.50    # 单策略最大权重（50%）
    min_strategies_count: int = 2            # 最少策略数量

    # 协同风险约束
    max_correlation_threshold: float = 0.85  # 允许的最大相关性
    max_simultaneous_tail_losses: int = 2    # 允许同时尾部亏损的最大策略数

    # 元数据
    version: str = "v1.0"
    commit_hash: str = ""
    created_at: Optional[datetime] = None
    description: str = ""

    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.now()

    def validate(self) -> tuple[bool, List[str]]:
        """验证预算合理性

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 检查收益预算
        if self.budget_return_p95 >= 0:
            errors.append(f"budget_return_p95应为负数，当前: {self.budget_return_p95}")

        if self.budget_return_p95 < -0.50:
            errors.append(f"budget_return_p95过于宽松（<-50%）: {self.budget_return_p95}")

        # 检查MDD预算
        if self.budget_mdd_p95 <= 0:
            errors.append(f"budget_mdd_p95应为正数，当前: {self.budget_mdd_p95}")

        if self.budget_mdd_p95 > 0.50:
            errors.append(f"budget_mdd_p95过于宽松（>50%）: {self.budget_mdd_p95}")

        # 检查持续期预算
        if self.budget_duration_p95 <= 0:
            errors.append(f"budget_duration_p95应为正数，当前: {self.budget_duration_p95}")

        if self.budget_duration_p95 > 180:
            errors.append(f"budget_duration_p95过于宽松（>180天）: {self.budget_duration_p95}")

        # 检查权重范围
        if self.min_weight_per_strategy < 0 or self.min_weight_per_strategy > 0.2:
            errors.append(f"min_weight_per_strategy应在[0, 0.2]范围，当前: {self.min_weight_per_strategy}")

        if self.max_weight_per_strategy <= self.min_weight_per_strategy:
            errors.append(f"max_weight_per_strategy应>=min_weight_per_strategy")

        if self.max_weight_per_strategy > 1.0:
            errors.append(f"max_weight_per_strategy应<=1.0，当前: {self.max_weight_per_strategy}")

        # 检查策略数量
        if self.min_strategies_count < 1:
            errors.append(f"min_strategies_count应>=1，当前: {self.min_strategies_count}")

        # 检查相关性阈值
        if not (0 <= self.max_correlation_threshold <= 1):
            errors.append(f"max_correlation_threshold应在[0, 1]范围，当前: {self.max_correlation_threshold}")

        # 检查同时亏损策略数
        if self.max_simultaneous_tail_losses < 1:
            errors.append(f"max_simultaneous_tail_losses应>=1，当前: {self.max_simultaneous_tail_losses}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换BudgetUnit枚举
        data["budget_unit"] = self.budget_unit.value
        # 转换datetime
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioRiskBudget":
        """从字典创建"""
        # 转换BudgetUnit
        if isinstance(data.get("budget_unit"), str):
            data["budget_unit"] = BudgetUnit(data["budget_unit"])

        # 转换datetime
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return cls(**data)

    def save(self, filepath: str) -> None:
        """保存到文件

        Args:
            filepath: 文件路径
        """
        is_valid, errors = self.validate()
        if not is_valid:
            raise ValueError(f"预算验证失败: {errors}")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> "PortfolioRiskBudget":
        """从文件加载

        Args:
            filepath: 文件路径

        Returns:
            PortfolioRiskBudget
        """
        with open(filepath) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def get_summary(self) -> Dict[str, Any]:
        """获取预算摘要"""
        return {
            "version": self.version,
            "constraints": {
                "return_p95": f"{self.budget_return_p95*100:.1f}%",
                "mdd_p95": f"{self.budget_mdd_p95*100:.1f}%",
                "duration_p95": f"{self.budget_duration_p95}天"
            },
            "allocation": {
                "min_weight": f"{self.min_weight_per_strategy*100:.1f}%",
                "max_weight": f"{self.max_weight_per_strategy*100:.1f}%",
                "min_strategies": self.min_strategies_count
            },
            "synergy": {
                "max_correlation": f"{self.max_correlation_threshold:.2f}",
                "max_simultaneous_losses": self.max_simultaneous_tail_losses
            }
        }


@dataclass
class StrategyBudget:
    """单个策略的预算分配"""

    strategy_id: str
    allocated_weight: float        # 分配的权重
    risk_score: float              # 风险评分（0-100）
    tail_loss_contribution: float  # 尾部损失贡献

    # 约束
    weight_cap: Optional[float] = None  # 仓位上限
    disabled: bool = False             # 是否禁用

    # 历史统计
    budget_hit_count: int = 0          # 预算触发次数
    last_adjustment: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        if self.last_adjustment:
            data["last_adjustment"] = self.last_adjustment.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyBudget":
        """从字典创建"""
        if isinstance(data.get("last_adjustment"), str):
            data["last_adjustment"] = datetime.fromisoformat(data["last_adjustment"])
        return cls(**data)


@dataclass
class BudgetAllocation:
    """预算分配结果"""

    total_budget: PortfolioRiskBudget  # 总预算
    strategy_budgets: Dict[str, StrategyBudget]  # {strategy_id: StrategyBudget}

    # 分配统计
    total_weight: float = 0.0
    active_strategies: int = 0
    disabled_strategies: int = 0

    # 时间戳
    allocated_at: datetime = field(default_factory=datetime.now)

    def calculate_totals(self) -> None:
        """计算总计"""
        self.total_weight = sum(
            b.allocated_weight for b in self.strategy_budgets.values()
            if not b.disabled
        )
        self.active_strategies = sum(
            1 for b in self.strategy_budgets.values()
            if not b.disabled
        )
        self.disabled_strategies = sum(
            1 for b in self.strategy_budgets.values()
            if b.disabled
        )

    def validate_against_budget(self) -> tuple[bool, List[str]]:
        """验证分配是否满足预算约束

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 检查权重和
        if abs(self.total_weight - 1.0) > 0.01:
            errors.append(f"权重和应≈1.0，当前: {self.total_weight:.3f}")

        # 检查策略数量
        if self.active_strategies < self.total_budget.min_strategies_count:
            errors.append(
                f"活跃策略数({self.active_strategies}) < 最小要求({self.total_budget.min_strategies_count})"
            )

        # 检查单个策略权重
        for strat_id, budget in self.strategy_budgets.items():
            if budget.disabled:
                continue

            if budget.allocated_weight < self.total_budget.min_weight_per_strategy:
                errors.append(
                    f"{strat_id}: 权重({budget.allocated_weight:.3f}) < 最小值({self.total_budget.min_weight_per_strategy})"
                )

            if budget.weight_cap and budget.allocated_weight > budget.weight_cap:
                errors.append(
                    f"{strat_id}: 权重({budget.allocated_weight:.3f}) > 上限({budget.weight_cap})"
                )

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_budget": self.total_budget.to_dict(),
            "strategy_budgets": {
                k: v.to_dict() for k, v in self.strategy_budgets.items()
            },
            "total_weight": self.total_weight,
            "active_strategies": self.active_strategies,
            "disabled_strategies": self.disabled_strategies,
            "allocated_at": self.allocated_at.isoformat()
        }


def create_conservative_budget() -> PortfolioRiskBudget:
    """创建保守型预算配置

    适用于风险厌恶场景。
    """
    return PortfolioRiskBudget(
        budget_return_p95=-0.05,     # P95收益-5%
        budget_mdd_p95=0.08,         # P95回撤8%
        budget_duration_p95=20,      # P95持续20天
        budget_unit=BudgetUnit.TAIL_LOSS_CONTRIBUTION,
        min_weight_per_strategy=0.10,  # 最小10%
        max_weight_per_strategy=0.40,  # 最大40%
        min_strategies_count=2,
        max_correlation_threshold=0.80,
        max_simultaneous_tail_losses=2,
        description="保守型：严格的风险控制"
    )


def create_moderate_budget() -> PortfolioRiskBudget:
    """创建中等预算配置

    适用于平衡风险和收益的场景。
    """
    return PortfolioRiskBudget(
        budget_return_p95=-0.10,     # P95收益-10%
        budget_mdd_p95=0.15,         # P95回撤15%
        budget_duration_p95=30,      # P95持续30天
        budget_unit=BudgetUnit.TAIL_LOSS_CONTRIBUTION,
        min_weight_per_strategy=0.05,  # 最小5%
        max_weight_per_strategy=0.50,  # 最大50%
        min_strategies_count=2,
        max_correlation_threshold=0.85,
        max_simultaneous_tail_losses=2,
        description="中等型：平衡风险和收益"
    )


def create_aggressive_budget() -> PortfolioRiskBudget:
    """创建激进型预算配置

    适用于追求更高收益的场景。
    """
    return PortfolioRiskBudget(
        budget_return_p95=-0.15,     # P95收益-15%
        budget_mdd_p95=0.25,         # P95回撤25%
        budget_duration_p95=45,      # P95持续45天
        budget_unit=BudgetUnit.TAIL_LOSS_CONTRIBUTION,
        min_weight_per_strategy=0.03,  # 最小3%
        max_weight_per_strategy=0.60,  # 最大60%
        min_strategies_count=2,
        max_correlation_threshold=0.90,
        max_simultaneous_tail_losses=3,
        description="激进型：追求更高收益"
    )


if __name__ == "__main__":
    """测试代码"""
    print("=== PortfolioRiskBudget 测试 ===\n")

    # 测试1: 创建预算
    print("1. 创建中等预算:")
    budget = create_moderate_budget()
    is_valid, errors = budget.validate()

    print(f"   验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    if errors:
        for error in errors:
            print(f"   - {error}")

    print(f"\n   预算摘要:")
    summary = budget.get_summary()
    for category, items in summary.items():
        print(f"   {category}:")
        if isinstance(items, dict):
            for key, value in items.items():
                print(f"     {key}: {value}")
        else:
            print(f"     {items}")

    # 测试2: 保存和加载
    print("\n2. 保存和加载:")
    test_file = "/tmp/test_budget.json"
    budget.save(test_file)
    print(f"   ✓ 已保存到 {test_file}")

    loaded_budget = PortfolioRiskBudget.load(test_file)
    print(f"   ✓ 已加载，version={loaded_budget.version}")

    # 测试3: 预算分配
    print("\n3. 预算分配:")
    allocation = BudgetAllocation(
        total_budget=budget,
        strategy_budgets={
            "strategy_1": StrategyBudget(
                strategy_id="strategy_1",
                allocated_weight=0.60,
                risk_score=50.0,
                tail_loss_contribution=-0.05
            ),
            "strategy_2": StrategyBudget(
                strategy_id="strategy_2",
                allocated_weight=0.40,
                risk_score=40.0,
                tail_loss_contribution=-0.03
            )
        }
    )
    allocation.calculate_totals()

    print(f"   总权重: {allocation.total_weight:.2f}")
    print(f"   活跃策略: {allocation.active_strategies}")
    print(f"   禁用策略: {allocation.disabled_strategies}")

    is_valid, errors = allocation.validate_against_budget()
    print(f"   验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    if errors:
        for error in errors:
            print(f"   - {error}")

    print("\n✓ 测试完成")
