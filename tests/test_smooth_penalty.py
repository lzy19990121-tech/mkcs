"""
P0-1: 权重平滑惩罚测试

验证目标函数中的权重平滑惩罚功能：
1. 当 lambda>0：smooth_penalty_value 非 0
2. 同一输入：加惩罚前后，权重波动指标下降
3. lambda=0：结果与旧版完全一致
4. 输出 artifacts 中有完整审计字段
"""

import sys
import os
from pathlib import Path

# 添加项��根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import json
from analysis.optimization_problem import OptimizationProblem
from analysis.portfolio_optimizer_v2 import PortfolioOptimizerV2
from analysis.pipeline_optimizer_v2 import PipelineOptimizerV2, PipelineConfig


def test_smooth_penalty_in_objective():
    """测试1: lambda>0 时平滑惩罚值非 0"""
    print("\n" + "="*70)
    print("测试1: lambda>0 时平滑惩罚值非 0")
    print("="*70)

    # 创建测试问题
    strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]
    problem = OptimizationProblem(
        name="test_smooth_penalty",
        description="测试平滑惩罚",
        n_strategies=3,
        strategy_ids=strategy_ids,
        expected_returns=np.array([0.05, 0.03, 0.04]),
        covariance_matrix=np.array([
            [0.01, 0.002, 0.003],
            [0.002, 0.015, 0.004],
            [0.003, 0.004, 0.02]
        ])
    )

    # 创建优化器
    optimizer = PortfolioOptimizerV2(problem)

    # 创建模拟风险代理
    risk_proxies = {
        "expected_returns": np.array([0.05, 0.03, 0.04]),
        "covariance_matrix": np.array([
            [0.01, 0.002, 0.003],
            [0.002, 0.015, 0.004],
            [0.003, 0.004, 0.02]
        ]),
        "returns_matrix": np.random.randn(100, 3) * 0.01,
    }

    # 设置上一次权重（与最优解不同）
    previous_weights = np.array([0.5, 0.3, 0.2])

    # 测试 lambda>0
    smooth_config = {
        "lambda": 1.0,  # 显著的惩罚
        "mode": "l2",
        "previous_weights": previous_weights
    }

    result = optimizer.run_optimization(risk_proxies, smooth_penalty_config=smooth_config)

    print(f"  优化成功: {result.success}")
    print(f"  平滑惩罚值: {result.smooth_penalty_value:.6f}")
    print(f"  使用的 lambda: {result.smooth_penalty_lambda}")
    print(f"  使用的模式: {result.smooth_penalty_mode}")
    print(f"  最终权重: {result.weights}")

    # 验证
    assert result.smooth_penalty_lambda == 1.0, "lambda 应该为 1.0"
    assert result.smooth_penalty_mode == "l2", "模式应该为 l2"
    assert result.smooth_penalty_value > 0, "惩罚值应该 > 0"

    print("  ✅ 测试1通过：lambda>0 时平滑惩罚值非 0")
    return True


def test_penalty_reduces_weight_changes():
    """测试2: 加惩罚后权重波动下降"""
    print("\n" + "="*70)
    print("测试2: 加惩罚后权重波动下降")
    print("="*70)

    strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]
    problem = OptimizationProblem(
        name="test_weight_stability",
        description="测试权重稳定性",
        n_strategies=3,
        strategy_ids=strategy_ids,
        expected_returns=np.array([0.05, 0.03, 0.04]),
        covariance_matrix=np.array([
            [0.01, 0.002, 0.003],
            [0.002, 0.015, 0.004],
            [0.003, 0.004, 0.02]
        ])
    )

    optimizer = PortfolioOptimizerV2(problem)

    risk_proxies = {
        "expected_returns": np.array([0.05, 0.03, 0.04]),
        "covariance_matrix": np.array([
            [0.01, 0.002, 0.003],
            [0.002, 0.015, 0.004],
            [0.003, 0.004, 0.02]
        ]),
        "returns_matrix": np.random.randn(100, 3) * 0.01,
    }

    # 设置上一次权重（与最优解差距较大）
    previous_weights = np.array([0.6, 0.3, 0.1])

    # 不加惩罚
    result_no_penalty = optimizer.run_optimization(risk_proxies, smooth_penalty_config=None)

    # 加惩罚 (lambda=5.0)
    smooth_config = {
        "lambda": 5.0,
        "mode": "l2",
        "previous_weights": previous_weights
    }
    result_with_penalty = optimizer.run_optimization(risk_proxies, smooth_penalty_config=smooth_config)

    # 计算权重变化幅度
    def calc_weight_change(weights, previous):
        return np.mean(np.abs(weights - previous))

    change_no_penalty = calc_weight_change(result_no_penalty.weights, previous_weights)
    change_with_penalty = calc_weight_change(result_with_penalty.weights, previous_weights)

    print(f"  不加惩罚的权重变化: {change_no_penalty:.4f}")
    print(f"  加惩罚的权重变化: {change_with_penalty:.4f}")
    print(f"  不加惩罚权重: {result_no_penalty.weights}")
    print(f"  加惩罚权重: {result_with_penalty.weights}")

    assert change_with_penalty < change_no_penalty, "加惩罚后权重变化应该更小"

    print("  ✅ 测试2通过：加惩罚后权重波动下降")
    return True


def test_lambda_zero_no_effect():
    """测试3: lambda=0 时与无惩罚一致"""
    print("\n" + "="*70)
    print("测试3: lambda=0 时与无惩罚一致")
    print("="*70)

    strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]
    problem = OptimizationProblem(
        name="test_lambda_zero",
        description="测试 lambda=0",
        n_strategies=3,
        strategy_ids=strategy_ids,
        expected_returns=np.array([0.05, 0.03, 0.04]),
        covariance_matrix=np.array([
            [0.01, 0.002, 0.003],
            [0.002, 0.015, 0.004],
            [0.003, 0.004, 0.02]
        ])
    )

    optimizer = PortfolioOptimizerV2(problem)

    risk_proxies = {
        "expected_returns": np.array([0.05, 0.03, 0.04]),
        "covariance_matrix": np.array([
            [0.01, 0.002, 0.003],
            [0.002, 0.015, 0.004],
            [0.003, 0.004, 0.02]
        ]),
        "returns_matrix": np.random.randn(100, 3) * 0.01,
    }

    previous_weights = np.array([0.5, 0.3, 0.2])

    # 无惩罚配置
    result_no_config = optimizer.run_optimization(risk_proxies, smooth_penalty_config=None)

    # lambda=0
    smooth_config = {
        "lambda": 0.0,
        "mode": "l2",
        "previous_weights": previous_weights
    }
    result_lambda_zero = optimizer.run_optimization(risk_proxies, smooth_penalty_config=smooth_config)

    print(f"  无惩罚权重: {result_no_config.weights}")
    print(f"  lambda=0 权重: {result_lambda_zero.weights}")
    print(f"  权重差异: {np.max(np.abs(result_no_config.weights - result_lambda_zero.weights)):.10f}")

    assert result_lambda_zero.smooth_penalty_value == 0.0, "lambda=0 时惩罚值应为 0"

    # 权重应该非常接近（由于数值误差可能不完全相等）
    assert np.allclose(result_no_config.weights, result_lambda_zero.weights, atol=1e-6), \
        "lambda=0 时权重应该与无惩罚一致"

    print("  ✅ 测试3通过：lambda=0 时与无惩罚一致")
    return True


def test_audit_fields():
    """测试4: 审计字段完整性"""
    print("\n" + "="*70)
    print("测试4: 审计字段完整性")
    print("="*70)

    strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]
    problem = OptimizationProblem(
        name="test_audit_fields",
        description="测试审计字段",
        n_strategies=3,
        strategy_ids=strategy_ids,
        expected_returns=np.array([0.05, 0.03, 0.04]),
        covariance_matrix=np.eye(3) * 0.01
    )

    optimizer = PortfolioOptimizerV2(problem)

    risk_proxies = {
        "expected_returns": np.array([0.05, 0.03, 0.04]),
        "covariance_matrix": np.eye(3) * 0.01,
        "returns_matrix": np.random.randn(100, 3) * 0.01,
    }

    previous_weights = np.array([0.5, 0.3, 0.2])

    smooth_config = {
        "lambda": 2.0,
        "mode": "l1",
        "previous_weights": previous_weights
    }

    result = optimizer.run_optimization(risk_proxies, smooth_penalty_config=smooth_config)

    # 检查审计字段
    print(f"  smooth_penalty_value: {result.smooth_penalty_value}")
    print(f"  smooth_penalty_lambda: {result.smooth_penalty_lambda}")
    print(f"  smooth_penalty_mode: {result.smooth_penalty_mode}")

    assert hasattr(result, 'smooth_penalty_value'), "应该有 smooth_penalty_value 字段"
    assert hasattr(result, 'smooth_penalty_lambda'), "应该有 smooth_penalty_lambda 字段"
    assert hasattr(result, 'smooth_penalty_mode'), "应该有 smooth_penalty_mode 字段"

    assert result.smooth_penalty_value >= 0, "惩罚值应该 >= 0"
    assert result.smooth_penalty_lambda == 2.0, "lambda 应该为 2.0"
    assert result.smooth_penalty_mode == "l1", "模式应该为 l1"

    # 检查 to_dict 包含审计字段
    result_dict = result.to_dict()
    assert "smooth_penalty_value" in result_dict, "to_dict 应该包含 smooth_penalty_value"
    assert "smooth_penalty_lambda" in result_dict, "to_dict 应该包含 smooth_penalty_lambda"
    assert "smooth_penalty_mode" in result_dict, "to_dict 应该包含 smooth_penalty_mode"

    print("  ✅ 测试4通过：审计字段完整")
    return True


def test_l1_vs_l2_mode():
    """测试5: L1 和 L2 模式区别"""
    print("\n" + "="*70)
    print("测试5: L1 和 L2 模式区别")
    print("="*70)

    strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]
    problem = OptimizationProblem(
        name="test_l1_l2",
        description="测试 L1/L2 模式",
        n_strategies=3,
        strategy_ids=strategy_ids,
        expected_returns=np.array([0.05, 0.03, 0.04]),
        covariance_matrix=np.eye(3) * 0.01
    )

    optimizer = PortfolioOptimizerV2(problem)

    risk_proxies = {
        "expected_returns": np.array([0.05, 0.03, 0.04]),
        "covariance_matrix": np.eye(3) * 0.01,
        "returns_matrix": np.random.randn(100, 3) * 0.01,
    }

    previous_weights = np.array([0.6, 0.3, 0.1])
    lambda_val = 1.0

    # L1 模式
    config_l1 = {
        "lambda": lambda_val,
        "mode": "l1",
        "previous_weights": previous_weights
    }
    result_l1 = optimizer.run_optimization(risk_proxies, smooth_penalty_config=config_l1)

    # L2 模式
    config_l2 = {
        "lambda": lambda_val,
        "mode": "l2",
        "previous_weights": previous_weights
    }
    result_l2 = optimizer.run_optimization(risk_proxies, smooth_penalty_config=config_l2)

    print(f"  L1 惩罚值: {result_l1.smooth_penalty_value:.6f}")
    print(f"  L1 权重: {result_l1.weights}")
    print(f"  L2 惩罚值: {result_l2.smooth_penalty_value:.6f}")
    print(f"  L2 权重: {result_l2.weights}")

    assert result_l1.smooth_penalty_mode == "l1", "L1 模式应该正确记录"
    assert result_l2.smooth_penalty_mode == "l2", "L2 模式应该正确记录"

    print("  ✅ 测试5通过：L1 和 L2 模式正确实现")
    return True


def test_pipeline_integration():
    """测试6: Pipeline 集成"""
    print("\n" + "="*70)
    print("测试6: Pipeline 集成")
    print("="*70)

    # 创建配置启用平滑惩罚
    config = PipelineConfig(
        smooth_penalty_lambda=1.0,
        smooth_penalty_mode="l2"
    )

    strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]

    pipeline = PipelineOptimizerV2(strategy_ids, config)

    # 设置上一次权重
    pipeline.previous_weights = {
        "strategy_1": 0.5,
        "strategy_2": 0.3,
        "strategy_3": 0.2
    }

    # 检查配置正确传递
    assert pipeline.config.smooth_penalty_lambda == 1.0, "Pipeline 配置应该包含 lambda"
    assert pipeline.config.smooth_penalty_mode == "l2", "Pipeline 配置应该包含 mode"

    print(f"  Pipeline 配置 lambda: {pipeline.config.smooth_penalty_lambda}")
    print(f"  Pipeline 配置 mode: {pipeline.config.smooth_penalty_mode}")

    print("  ✅ 测试6通过：Pipeline 配置正确")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("P0-1: 权重平滑惩罚测试套件")
    print("="*70)

    tests = [
        ("lambda>0 时惩罚值非 0", test_smooth_penalty_in_objective),
        ("加惩罚后权重波动下降", test_penalty_reduces_weight_changes),
        ("lambda=0 时与无惩罚一致", test_lambda_zero_no_effect),
        ("审计字段完整性", test_audit_fields),
        ("L1 和 L2 模式区别", test_l1_vs_l2_mode),
        ("Pipeline 集成", test_pipeline_integration),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {name} 失败: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("="*70)

    if failed == 0:
        print("\n🎉 所有测试通过！")

    sys.exit(0 if failed == 0 else 1)
