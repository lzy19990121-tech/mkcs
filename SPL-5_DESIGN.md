# SPL-5 技术设计文档

**版本**: v1.0
**日期**: 2026-02-01
**状态**: 设计阶段

---

## 概述

SPL-5 系列是对 SPL-4 的重大升级，引入自适应风控和组合预算分配：

- **SPL-5a**: Adaptive Risk Gating - 固定阈值 → 自适应阈值
- **SPL-5b**: Risk Budget Allocation - 固定权重 → 预算化分配

---

# SPL-5a: Adaptive Risk Gating

## 目标

将 SPL-4 的固定阈值 gating 升级为随市场状态/风险状态自适应的 gating：

| 指标 | SPL-4 | SPL-5a |
|------|-------|--------|
| 阈值类型 | 固定常数 | 随状态变化的函数 |
| 误杀率 | 较高 | 降低 |
| 避险时效 | 滞后 | 提前 |
| 风险约束 | 满足4c回归 | 仍满足 |

---

## A. 自适应输入特征

### A.1 特征定义

定义3类用于 runtime 的状态特征：

| 特征类 | 指标 | 计算方式 | 用途 |
|--------|------|----------|------|
| **波动状态** | realized_vol | 过去20日收益标准差 | 高波动→保守阈值 |
| **趋势状态** | ADX | 平均趋向指数 | 震荡→禁用策略 |
| **流动性状态** | spread_proxy | bid-ask spread估计 | 低流动性→降仓 |

### A.2 特征计算

```python
@dataclass
class RegimeFeatures:
    """市场状态特征"""

    # 波动状态
    realized_vol: float      # 过去20日收益波动率
    vol_bucket: str          # "low", "med", "high"

    # 趋势状态
    adx: float               # ADX值
    trend_bucket: str        # "weak", "strong"

    # 流动性状态
    spread_proxy: float      # 价差代理
    cost_bucket: str         # "low", "high"

    @classmethod
    def calculate(cls, ctx: RunContext, window: int = 20) -> "RegimeFeatures":
        """从运行时数据计算特征"""
        # 实现细节...
        pass
```

### A.3 离散分桶

每个特征离散化为2-4个桶：

| 特征 | 桶数 | 边界 | 示例 |
|------|------|------|------|
| realized_vol | 3 | [0.01, 0.02, 0.05] | low/med/high |
| ADX | 2 | [25] | weak/strong |
| spread_proxy | 2 | [0.001] | low/high |

**分桶原则**：
- 桶数≤4（防过拟合）
- 边界基于数据分位数
- 单调性约束（vol↑ → threshold↓）

---

## B. 阈值函数设计

### B.1 函数形式

将 SPL-4 的每条规则阈值改为分段常数函数：

```python
# SPL-4: 固定阈值
THRESHOLD_STABILITY = 30.0

# SPL-5a: 自适应阈值
def T_stability(regime: RegimeFeatures) -> float:
    """稳定性评分阈值随波动变化"""
    if regime.vol_bucket == "low":
        return 20.0  # 低波动时允许更低的稳定性
    elif regime.vol_bucket == "med":
        return 30.0  # 中等波动保持原阈值
    else:  # high
        return 40.0  # 高波动时要求更高稳定性
```

### B.2 阈值函数表

| 规则 | SPL-4阈值 | SPL-5a函数（示例） |
|------|-----------|-------------------|
| stability_gating | 30.0 | T_stability(vol) ∈ {20, 30, 40} |
| return_reduction | -0.10 | T_return(vol) ∈ {-0.08, -0.10, -0.15} |
| regime_disable | ADX<25 | T_disable(trend) ∈ {weak→禁用} |
| duration_gating | 10.0 | T_duration(vol) ∈ {8, 10, 15} |

### B.3 复杂度约束

**防过拟合规则**：
1. 仅允许分段常数/分段线性
2. 桶数≤4
3. 单调性约束（vol↑ → threshold↑）
4. 总参数数≤原规则数×3

---

## C. 离线标定

### C.1 数据切分

使用时间序列切分（不打乱顺序）：

```
训练集: [2023-01-01, 2024-06-30]  # 较早数据
验证集: [2024-07-01, 2024-12-31]  # 较晚数据
```

**要求**：
- 训练集和验证集都必须包含 worst-case 时期
- 验证集用于防止过拟合

### C.2 优化目标

多目标优化（风险约束为硬条件）：

**硬条件**：
- worst-case envelope 不变或更好
  - `Return_P95` ≤ baseline
  - `MDD_P95` ≤ baseline
  - `Duration_P95` ≤ baseline

**软目标**（最小化）：
- gating 触发次数
- 停机总时长
- 收益损失

**目标函数**：

```python
objective = α * trigger_count
           + β * downtime
           + γ * return_loss

subject to:
    envelope_constraints_satisfied()
```

### C.3 标定方法

#### 方法1: 网格搜索（推荐）

```python
# 搜索空间
param_grid = {
    'vol_low_threshold': [15, 20, 25],
    'vol_med_threshold': [25, 30, 35],
    'vol_high_threshold': [35, 40, 45],
    # ... 其他参数
}

# 网格搜索
best_params = None
best_score = inf

for params in ParameterGrid(param_grid):
    # 运行回测
    result = backtest_with_adaptive_gating(params, train_data)

    # 检查约束
    if not check_envelope_constraints(result):
        continue

    # 计算目标
    score = objective_function(result)

    if score < best_score:
        best_score = score
        best_params = params
```

#### 方法2: 贝叶斯优化（可选）

```python
from skopt import gp_minimize

def objective(params):
    # 参数→阈值映射
    thresholds = params_to_thresholds(params)

    # 回测
    result = backtest_with_adaptive_gating(thresholds, train_data)

    # 检查约束（违反则返回大值）
    if not check_envelope_constraints(result):
        return 1e6

    # 返回目标值
    return calc_objective(result)

# 优化空间
space = [
    Real(15, 25, name='vol_low'),
    Real(25, 35, name='vol_med'),
    Real(35, 45, name='vol_high'),
]

# 运行优化
result = gp_minimize(objective, space, n_calls=50)
```

### C.4 输出格式

```json
{
  "strategy_id": "ma_5_20",
  "version": "adaptive_v1",
  "calibration_date": "2024-02-01",
  "train_period": ["2023-01-01", "2024-06-30"],
  "valid_period": ["2024-07-01", "2024-12-31"],
  "commit_hash": "abc123",

  "threshold_functions": {
    "stability_gating": {
      "type": "piecewise_constant",
      "input": "realized_vol",
      "buckets": {
        "low": {"max": 0.01, "threshold": 20.0},
        "med": {"min": 0.01, "max": 0.02, "threshold": 30.0},
        "high": {"min": 0.02, "threshold": 40.0}
      }
    },
    "return_reduction": {
      "type": "piecewise_constant",
      "input": "realized_vol",
      "buckets": {
        "low": {"max": 0.01, "threshold": -0.08},
        "med": {"min": 0.01, "max": 0.02, "threshold": -0.10},
        "high": {"min": 0.02, "threshold": -0.15}
      }
    }
  },

  "validation_metrics": {
    "envelope_return_p95": -0.2697,
    "envelope_mdd_p95": 0.0194,
    "gating_count": 12,
    "downtime_days": 45,
    "return_loss": -0.005
  }
}
```

---

## D. Runtime 实现

### D.1 架构

```python
class AdaptiveRiskGate(RiskGate):
    """自适应风控门"""

    def __init__(
        self,
        ruleset: RiskRuleset,
        threshold_functions: Dict[str, ThresholdFunction],
        window_length: str = "20d"
    ):
        super().__init__(ruleset, window_length)

        # 替换固定阈值为函数
        self.threshold_functions = threshold_functions
        self.regime_calculator = RegimeFeatureCalculator()

    def check(
        self,
        ctx: RunContext,
        positions: Dict,
        cash: float
    ) -> GateDecision:
        # 1. 计算当前状态特征
        regime = self.regime_calculator.calculate(ctx)

        # 2. 根据状态调整阈值
        for rule in self.ruleset.rules:
            func = self.threshold_functions.get(rule.rule_id)
            if func:
                rule.trigger_threshold = func(regime)

        # 3. 调用父类检查逻辑
        return super().check(ctx, positions, cash)
```

### D.2 状态特征计算器

```python
class RegimeFeatureCalculator:
    """计算市场状态特征"""

    def __init__(self, window: int = 20):
        self.window = window
        self.price_history = deque(maxlen=window)
        self.return_history = deque(maxlen=window)

    def calculate(self, ctx: RunContext) -> RegimeFeatures:
        """计算当前状态特征"""
        # 更新历史
        self.price_history.append(ctx.current_price)
        self.return_history.append(ctx.returns)

        # 计算波动率
        realized_vol = np.std(self.return_history)

        # 计算ADX（如果有数据）
        adx = self._calculate_adx()

        # 估算价差
        spread_proxy = self._estimate_spread()

        # 分桶
        vol_bucket = self._bucket_vol(realized_vol)
        trend_bucket = self._bucket_adx(adx)
        cost_bucket = self._bucket_spread(spread_proxy)

        return RegimeFeatures(
            realized_vol=realized_vol,
            vol_bucket=vol_bucket,
            adx=adx,
            trend_bucket=trend_bucket,
            spread_proxy=spread_proxy,
            cost_bucket=cost_bucket
        )
```

### D.3 对照验证

运行三组回测对比：

| 配置 | 描述 | 预期结果 |
|------|------|----------|
| 无Gating | 原始策略 | 最差收益，最多交易 |
| 固定阈值 | SPL-4 gating | 减少极端损失，增加误杀 |
| 自适应阈值 | SPL-5a gating | 极端损失不增加，误杀减少 |

**对比指标**：

```python
comparison_metrics = {
    "worst_case_return": {
        "no_gating": -0.27,
        "fixed": -0.20,
        "adaptive": -0.20  # 目标：不劣于fixed
    },
    "gating_count": {
        "no_gating": 0,
        "fixed": 8,
        "adaptive": 5  # 目标：少于fixed
    },
    "downtime_days": {
        "no_gating": 0,
        "fixed": 45,
        "adaptive": 30  # 目标：少于fixed
    },
    "final_return": {
        "no_gating": -0.017,
        "fixed": -0.020,
        "adaptive": -0.018  # 目标：接近no_gating
    }
}
```

---

## E. 回归框架集成

### E.1 新增测试

在 SPL-4c 的5大测试基础上新增：

```python
# tests/risk_regression/adaptive_gating_test.py

class AdaptiveGatingTests:
    """SPL-5a 自适应gating回归测试"""

    def test_adaptive_non_regression(self):
        """自适应gating不应劣于固定阈值"""
        # worst-case envelope
        assert adaptive_return_p95 <= fixed_return_p95
        assert adaptive_mdd_p95 <= fixed_mdd_p95

        # gating效率
        assert adaptive_triggers < fixed_triggers
        assert adaptive_downtime < fixed_downtime

    def test_regime_bucket_drift(self):
        """Regime桶边界变化需要显式审查"""
        # 检查桶边界是否变化
        current_buckets = load_adaptive_buckets()
        baseline_buckets = load_baseline_buckets()

        for feature in ['vol', 'trend', 'cost']:
            if current_buckets[feature] != baseline_buckets[feature]:
                # 标记需要人工审查
                mark_for_review(feature)
```

### E.2 输出文件

```
baselines/risk/
├── adaptive_gating_params.json    # 自适应阈值参数
├── adaptive_gating_baseline.json   # 基线性能
└── adaptive_gating_report.md       # 验证报告
```

---

## SPL-5a Exit Criteria

- [ ] worst-case envelope 不劣于 SPL-4（Return/MDD/Duration P95）
- [ ] gating 次数/停机时长显著下降（或有明确 tradeoff）
- [ ] 通过 SPL-4c 的 risk regression
- [ ] 参数已固化并与 commit/config 绑定

---

# SPL-5b: Risk Budget Allocation

## 目标

将 SPL-4 的组合分析升级为预算化、可分配、可优化的组合控制：

| 指标 | SPL-4 | SPL-5b |
|------|-------|--------|
| 分配方式 | 固定权重 | 动态风险预算 |
| 风险约束 | 组合级分析 | 策略级预算+组合约束 |
| 协同风险 | 识别 | 限制 |

---

## A. 组合风险预算定义

### A.1 硬约束指标

选择组合层面的约束指标：

| 指标 | 描述 | 阈值 |
|------|------|------|
| Budget_Return_P95 | 组合最坏收益P95 | -10% |
| Budget_MDD_P95 | 组合最大回撤P95 | 15% |
| Budget_Duration_P95 | 回撤持续P95 | 30天 |

### A.2 预算目标值

```python
@dataclass
class PortfolioRiskBudget:
    """组合风险预算"""

    # 组合级约束
    budget_return_p95: float     # -0.10
    budget_mdd_p95: float        # 0.15
    budget_duration_p95: int     # 30

    # 策略级预算单位
    budget_unit: str = "tail_loss_contribution"  # 或 "risk_weight"

    # 元数据
    version: str = "v1.0"
    commit_hash: str = ""
    created_at: datetime = None

    def validate(self):
        """验证预算合理性"""
        assert self.budget_return_p95 < 0, "Return budget should be negative"
        assert self.budget_mdd_p95 > 0, "MDD budget should be positive"
```

---

## B. 风险归因与分摊

### B.1 策略贡献分解

基于 SPL-4b 的组合最坏窗口，分解每个策略的贡献：

```python
def decompose_strategy_contributions(
    portfolio: Portfolio,
    worst_windows: List[PortfolioWindowMetrics]
) -> Dict[str, float]:
    """分解策略对组合worst-case的贡献

    Returns:
        {strategy_id: contribution_ratio}
    """
    contributions = defaultdict(float)

    for window in worst_windows:
        # 计算每个策略在该窗口的收益
        for strat_id in portfolio.config.strategy_ids:
            strat_return = window.strategy_returns[strat_id]
            strat_weight = portfolio.config.weights[strat_id]

            # 贡献 = 权重 × 收益
            contribution = strat_weight * strat_return

            # 累加贡献（只计算负收益）
            if contribution < 0:
                contributions[strat_id] += contribution

    # 归一化
    total = sum(abs(v) for v in contributions.values())
    if total > 0:
        contributions = {k: abs(v)/total for k, v in contributions.items()}

    return contributions
```

### B.2 协同爆炸对识别

识别在压力期间一起亏损的策略对：

```python
def identify_co_crash_pairs(
    portfolio: Portfolio,
    worst_windows: List[PortfolioWindowMetrics],
    correlation_threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    """识别协同爆炸对

    Returns:
        [(strat1, strat2, correlation), ...]
    """
    co_crash_pairs = []

    # 在最坏窗口中计算策略相关性
    for window in worst_windows:
        # 获取策略收益序列
        returns_df = portfolio.get_window_returns(window)

        # 计算相关性矩阵
        corr_matrix = returns_df.corr()

        # 找到高相关性对
        for i, strat1 in enumerate(returns_df.columns):
            for j, strat2 in enumerate(returns_df.columns):
                if i < j:  # 避免重复
                    corr = corr_matrix.loc[strat1, strat2]
                    if corr > correlation_threshold:
                        # 检查是否同时亏损
                        if (returns_df[strat1].mean() < 0 and
                            returns_df[strat2].mean() < 0):
                            co_crash_pairs.append((strat1, strat2, corr))

    return co_crash_pairs
```

### B.3 策略风险评分

综合多个维度给策略打分：

```python
def calculate_strategy_risk_score(
    strategy_id: str,
    envelope: RiskEnvelope,
    structural_result: StructuralAnalysisResult,
    stability_report: StabilityReport,
    regime_sensitivity: Dict[str, float]
) -> float:
    """计算策略风险评分（0-100，越高越风险）

    维度：
    1. Envelope worst-case (40分)
    2. Structural label (30分)
    3. Stability (20分)
    4. Regime sensitivity (10分)
    """
    score = 0

    # 1. Envelope (0-40分)
    worst_return = envelope.return_p95
    score += min(40, abs(worst_return) * 100)

    # 2. Structural (0-30分)
    if structural_result.risk_pattern_type == "structural":
        score += 30  # 结构性风险
    elif structural_result.risk_pattern_type == "single_outlier":
        score += 10  # 单一异常

    # 3. Stability (0-20分)
    score += min(20, (100 - stability_report.stability_score) * 0.2)

    # 4. Regime sensitivity (0-10分)
    # 计算不同regime下的收益方差
    regime_variance = np.var(list(regime_sensitivity.values()))
    score += min(10, regime_variance * 100)

    return min(100, score)
```

### B.4 初始预算分配规则

基于风险评分的简单规则：

```python
def allocate_initial_budget(
    strategy_scores: Dict[str, float],
    total_budget: float
) -> Dict[str, float]:
    """基于风险评分分配预算

    规则：
    - 高风险(>60) → 预算↓
    - 低分(<40) → 预算↑
    - 协同对 → 互斥约束
    """
    allocations = {}

    # 计算权重（风险越高权重越低）
    weights = {}
    for strat_id, score in strategy_scores.items():
        if score > 60:
            weights[strat_id] = 0.5  # 减半
        elif score < 40:
            weights[strat_id] = 1.5  # 增加
        else:
            weights[strat_id] = 1.0

    # 归一化
    total_weight = sum(weights.values())
    for strat_id in weights:
        allocations[strat_id] = (weights[strat_id] / total_weight) * total_budget

    return allocations
```

---

## C. 动态预算分配器

### C.1 v1: 规则+阈值（推荐起步）

```python
class RuleBasedAllocator:
    """基于规则的预算分配器"""

    def __init__(
        self,
        rules: List[AllocationRule],
        risk_budget: PortfolioRiskBudget
    ):
        self.rules = rules
        self.risk_budget = risk_budget

    def allocate(
        self,
        current_regime: RegimeFeatures,
        strategy_states: Dict[str, StrategyState],
        portfolio_exposure: float
    ) -> AllocationResult:
        """计算分配方案"""
        result = AllocationResult()

        for rule in self.rules:
            # 应用每条规则
            rule.apply(current_regime, strategy_states, result)

        # 检查总预算
        result.check_budget(self.risk_budget)

        return result

@dataclass
class AllocationResult:
    """分配结果"""
    target_weights: Dict[str, float]     # 目标权重
    position_caps: Dict[str, float]      # 仓位上限
    disabled_strategies: List[str]        # 禁用列表
```

### C.2 规则示例

```python
# 规则1: 高波动降低总仓位
if current_regime.vol_bucket == "high":
    total_cap = 0.8  # 降低到80%

# 规则2: 震荡市场禁用趋势策略
if current_regime.trend_bucket == "weak":
    for strat_id in trend_following_strategies:
        result.disabled_strategies.append(strat_id)

# 规则3: 协同对互斥
for (s1, s2, _) in co_crash_pairs:
    # 如果s1持仓高，降低s2
    if portfolio.get_weight(s1) > 0.5:
        result.position_caps[s2] = 0.3
```

### C.3 v2: 优化器（可选）

```python
from scipy.optimize import minimize

class OptimizationAllocator:
    """基于优化的预算分配器"""

    def allocate(
        self,
        expected_returns: Dict[str, float],
        risk_budget: PortfolioRiskBudget,
        current_exposures: Dict[str, float]
    ) -> Dict[str, float]:
        """优化分配权重

        目标：最大化收益
        约束：风险预算
        """

        # 目标函数：最小化负收益
        def objective(weights):
            portfolio_return = sum(
                weights[i] * expected_returns[strat_id]
                for i, strat_id in enumerate(expected_returns.keys())
            )
            return -portfolio_return

        # 约束条件
        constraints = [
            # 权重和为1
            {'type': 'eq', 'fun': lambda w: sum(w) - 1},

            # 风险预算约束
            {
                'type': 'ineq',
                'fun': lambda w: risk_budget.budget_return_p95 -
                                self._calculate_portfolio_var(w)
            }
        ]

        # 边界（权重在0-1之间）
        bounds = [(0, 1) for _ in expected_returns]

        # 初始值（等权重）
        x0 = [1/len(expected_returns)] * len(expected_returns)

        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return {
            strat_id: result.x[i]
            for i, strat_id in enumerate(expected_returns.keys())
        }
```

---

## D. 组合回测与扫描

### D.1 对照实验

三组配置对比：

| 配置 | 描述 | 分配方式 |
|------|------|----------|
| Baseline | SPL-4组合 | 固定权重 |
| +Budget | SPL-5b预算分配 | 动态权重+预算约束 |
| +Budget+Adaptive | SPL-5a+5b | 自适应gating+预算分配 |

### D.2 组合Worst-Case扫描

```python
def scan_portfolio_worst_cases(
    portfolio: Portfolio,
    allocation_result: AllocationResult,
    window_lengths: List[str] = ["20d", "60d"]
) -> Dict[str, List[PortfolioWindowMetrics]]:
    """扫描组合最坏窗口"""

    worst_windows = {}

    for window_length in window_lengths:
        # 应用分配结果
        adjusted_portfolio = apply_allocation(portfolio, allocation_result)

        # 扫描窗口
        scanner = PortfolioWindowScanner()
        windows = scanner.find_worst_portfolio_windows(
            adjusted_portfolio,
            window_length,
            top_k=5
        )

        worst_windows[window_length] = windows

    return worst_windows
```

### D.3 协同爆炸检测

```python
def detect_synergy_reduction(
    baseline_worst: List[PortfolioWindowMetrics],
    budget_worst: List[PortfolioWindowMetrics]
) -> Dict[str, float]:
    """检测协同风险是否被削弱"""

    return {
        "co_crash_count_reduction":
            count_co_crashes(baseline_worst) - count_co_crashes(budget_worst),

        "simultaneous_tail_loss_reduction":
            measure_simultaneous_tail_losses(baseline_worst) -
            measure_simultaneous_tail_losses(budget_worst),

        "correlation_spike_reduction":
            count_correlation_spikes(baseline_worst) -
            count_correlation_spikes(budget_worst)
    }
```

### D.4 预算命中报告

```python
@dataclass
class BudgetHitReport:
    """预算使用报告"""

    # 预算是否被触达
    budgets_hit: Dict[str, bool]

    # 经常顶到cap的策略
    capped_strategies: Dict[str, int]  # {strategy_id: hit_count}

    # 触发预算收缩的regime
    contraction_triggers: List[str]

    # 统计
    total_checks: int
    budget_hit_rate: float
```

---

## E. 组合回归框架

### E.1 新增测试

```python
class PortfolioRegressionTests:
    """SPL-5b 组合回归测试"""

    def test_portfolio_envelope_non_regression(self):
        """组合worst-case不应退化"""
        # P95指标
        assert current_return_p95 <= baseline_return_p95
        assert current_mdd_p95 <= baseline_mdd_p95

    def test_correlation_spike_guard(self):
        """压力期相关性上升不应突破阈值"""
        worst_correlation = calculate_max_correlation_in_stress()
        assert worst_correlation < 0.9

    def test_co_crash_count_guard(self):
        """同时爆的策略数应受限"""
        co_crash_count = count_simultaneous_tail_losses()
        assert co_crash_count <= 2  # 最多2个策略同时爆

    def test_budget_breach_detection(self):
        """预算违规应被检测并记录"""
        breaches = detect_budget_breaches()
        assert len(breaches) == 0, f"预算违规: {breaches}"
```

### E.2 输出文件

```
baselines/portfolio/
├── portfolio_budget_spec.json       # 预算定义
├── allocator_rules.md              # 分配规则
└── portfolio_regression_report.md  # 回归报告
```

---

## SPL-5b Exit Criteria

- [ ] 组合 worst-case 满足 risk budget
- [ ] 协同爆炸事件频率/强度下降
- [ ] 相比 SPL-4 收益/稳定性改进或 tradeoff 清晰
- [ ] 组合级回归测试通过

---

# 实施计划

## 阶段划分

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| **Phase 1** | SPL-5a A+B+C | 自适应输入+阈值函数+离线标定 | 2周 |
| **Phase 2** | SPL-5a D+E | Runtime实现+回归集成 | 1周 |
| **Phase 3** | SPL-5b A+B | 预算定义+归因分摊 | 1周 |
| **Phase 4** | SPL-5b C+D | 分配器+回测验证 | 1周 |
| **Phase 5** | SPL-5b E | 组合回归+文档 | 1周 |

**总计**: 约6周

---

# 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 过拟合 | 阈值失效 | 时间切分验证+参数上限 |
| 计算复杂度 | Runtime延迟 | 简化规则+缓存 |
| 数据不足 | 特征不准 | 使用proxy指标+降维 |
| 组合不稳定 | 预算违规 | 保守预算+实时监控 |

---

# 附录

## A. 特征计算细节

### A.1 Realized Volatility

```python
def calculate_realized_vol(returns: List[float], window: int = 20) -> float:
    """计算已实现波动率"""
    if len(returns) < window:
        return 0.0
    return np.std(returns[-window:])
```

### A.2 ADX

```python
def calculate_adx(prices: List[float], period: int = 14) -> float:
    """计算平均趋向指数"""
    # +DI, -DI, DX计算
    # ADX = SMA(DX, period)
    # 简化实现...
    pass
```

### A.3 Spread Proxy

```python
def estimate_spread(ctx: RunContext) -> float:
    """估算价差"""
    # 使用高低价差作为proxy
    if ctx.high and ctx.low:
        return (ctx.high - ctx.low) / ctx.close
    return 0.001  # 默认值
```

## B. 阈值函数示例

```python
# 完整的阈值函数配置
THRESHOLD_FUNCTIONS = {
    "stability_gating": {
        "type": "piecewise_constant",
        "input": "realized_vol",
        "monotonic": "increasing",  # vol↑ → threshold↑
        "buckets": {
            "low": {"max": 0.01, "threshold": 20.0},
            "med": {"min": 0.01, "max": 0.02, "threshold": 30.0},
            "high": {"min": 0.02, "threshold": 40.0}
        }
    },

    "return_reduction": {
        "type": "piecewise_linear",
        "input": "realized_vol",
        "function": lambda vol: -0.08 - 2.0 * vol  # 线性函数
    },

    "regime_disable": {
        "type": "binary",
        "input": "adx",
        "condition": lambda adx: adx < 25,
        "action": "disable"
    }
}
```

---

**文档版本**: v1.0
**最后更新**: 2026-02-01
**作者**: Claude (Sonnet 4.5)
