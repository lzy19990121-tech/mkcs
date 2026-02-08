"""
P2-2: 反事实分析完善测试

测试反事实分析功能：
- 被拒绝交易分析
- 权重平滑对比
- 假设场景分析
- 对照实验报告生成
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.counterfactual import (
    CounterfactualAnalyzer,
    CounterfactualType,
    CounterfactualResult,
    RejectedTradeAnalysis,
    analyze_rejected_trades_impact
)


def create_sample_backtest_result(total_return: float, rejected_count: int = 0) -> dict:
    """创建示例回测结果"""
    rejects = []
    for i in range(rejected_count):
        rejects.append({
            "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
            "symbol": "AAPL" if i % 2 == 0 else "MSFT",
            "action": "BUY" if i % 2 == 0 else "SELL",
            "reason": "风险限制",
            "confidence": 0.6 + i * 0.05
        })

    return {
        "summary": {
            "total_return": total_return,
            "total_trades": 100,
            "win_rate": 0.55,
            "initial_equity": 100000,
            "final_equity": 100000 * (1 + total_return)
        },
        "metrics": {
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08
        },
        "risk_rejects": rejects,
        "weight_history": [
            {"weights": {"strategy_1": 0.5, "strategy_2": 0.3, "strategy_3": 0.2}},
            {"weights": {"strategy_1": 0.55, "strategy_2": 0.25, "strategy_3": 0.2}},
            {"weights": {"strategy_1": 0.45, "strategy_2": 0.35, "strategy_3": 0.2}},
        ]
    }


def test_rejected_trades_analysis():
    """测试1: 被拒绝交易分析"""
    print("\n" + "="*70)
    print("测试1: 被拒绝交易分析")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = CounterfactualAnalyzer(storage_path=temp_dir)

        # 创建有拒绝记录的回测结果
        backtest_result = create_sample_backtest_result(total_return=0.15, rejected_count=10)
        price_data = {
            "AAPL": [(datetime.now(), 150.0)],
            "MSFT": [(datetime.now(), 300.0)]
        }

        analysis = analyzer.analyze_rejected_trades(backtest_result, price_data)

        print(f"  原始收益: {analysis.original_return*100:.2f}%")
        print(f"  如果接受: {analysis.if_accepted_return*100:.2f}%")
        print(f"  收益差: {(analysis.if_accepted_return - analysis.original_return)*100:+.2f}%")
        print(f"  被拒绝交易数: {len(analysis.rejected_trades)}")
        print(f"  本该盈利: {analysis.would_be_profitable}")
        print(f"  本该亏损: {analysis.would_be_loss}")

        # 验证
        assert analysis.original_return == 0.15, "原始收益应该正确"
        assert len(analysis.rejected_trades) == 10, "应该有 10 笔拒绝记录"
        assert len(analyzer.results) == 1, "应该记录了 1 个结果"

        result = analyzer.results[0]
        assert result.type == CounterfactualType.REJECTED_TRADES, "类型应该是 REJECTED_TRADES"

        print(f"\n  ✅ 被拒绝交易分析测试通过")

    return True


def test_weight_smoothing_comparison():
    """测试2: 权重平滑对比"""
    print("\n" + "="*70)
    print("测试2: 权重平滑对比")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = CounterfactualAnalyzer(storage_path=temp_dir)

        # 无平滑结果（高波动）
        no_smoothing = create_sample_backtest_result(total_return=0.18)
        no_smoothing["weight_history"] = [
            {"weights": {"s1": 0.5, "s2": 0.5}},
            {"weights": {"s1": 0.1, "s2": 0.9}},  # 大幅变化
            {"weights": {"s1": 0.8, "s2": 0.2}},  # 大幅变化
        ]

        # 有平滑结果（低波动）
        with_smoothing = create_sample_backtest_result(total_return=0.16)
        with_smoothing["weight_history"] = [
            {"weights": {"s1": 0.5, "s2": 0.5}},
            {"weights": {"s1": 0.52, "s2": 0.48}},  # 小幅变化
            {"weights": {"s1": 0.51, "s2": 0.49}},  # 小幅变化
        ]

        result = analyzer.compare_weight_smoothing(no_smoothing, with_smoothing, lambda_value=2.0)

        print(f"  无平滑收益: {result.baseline_return*100:.2f}%")
        print(f"  有平滑收益: {result.counterfactual_return*100:.2f}%")
        print(f"  收益差: {result.return_delta*100:+.2f}%")
        print(f"  无平滑波动: {result.metadata['no_smoothing_jitter']:.4f}")
        print(f"  有平滑波动: {result.metadata['with_smoothing_jitter']:.4f}")
        print(f"  波动降低: {result.metadata['jitter_reduction']*100:.1f}%")

        # 验证
        assert result.type == CounterfactualType.SMOOTHING_COMPARISON, "类型应该是 SMOOTHING_COMPARISON"
        assert result.metadata["lambda"] == 2.0, "Lambda 应该记录"
        assert result.metadata["jitter_reduction"] > 0, "波动应该降低"

        print(f"\n  ✅ 权重平滑对比测试通过")

    return True


def test_what_if_scenario():
    """测试3: 假设场景分析"""
    print("\n" + "="*70)
    print("测试3: 假设场景分析")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = CounterfactualAnalyzer(storage_path=temp_dir)

        backtest_result = create_sample_backtest_result(total_return=0.15)

        # 场景1: 降低手续费
        scenario1 = analyzer.what_if_scenario(
            backtest_result,
            {
                "description": "降低手续费 50%",
                "change_commission": 0.5,
                "return_adjustment": 0.02,  # 预期收益增加 2%
                "affected_trades": 100
            }
        )

        print(f"  场景1: 降低手续费")
        print(f"    原始收益: {scenario1.baseline_return*100:.2f}%")
        print(f"    场景收益: {scenario1.counterfactual_return*100:.2f}%")
        print(f"    收益提升: {scenario1.return_delta*100:+.2f}%")

        # 场景2: 禁用某个策略
        scenario2 = analyzer.what_if_scenario(
            backtest_result,
            {
                "description": "禁用波动策略",
                "disable_strategies": ["volatility_strategy"],
                "return_adjustment": -0.01,  # 预期收益减少 1%
                "affected_trades": 30
            }
        )

        print(f"\n  场景2: 禁用波动策略")
        print(f"    原始收益: {scenario2.baseline_return*100:.2f}%")
        print(f"    场景收益: {scenario2.counterfactual_return*100:.2f}%")
        print(f"    收益变化: {scenario2.return_delta*100:+.2f}%")

        # 验证
        assert scenario1.type == CounterfactualType.WHAT_IF_SCENARIO, "类型应该是 WHAT_IF_SCENARIO"
        assert scenario1.return_delta == 0.02, "收益调整应该正确"
        assert scenario2.return_delta == -0.01, "收益调整应该正确"

        print(f"\n  ✅ 假设场景分析测试通过")

    return True


def test_comparison_report_generation():
    """测试4: 对照实验报告生成"""
    print("\n" + "="*70)
    print("测试4: 对照实验报告生成")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = CounterfactualAnalyzer(storage_path=temp_dir)

        # 准备各组结果
        group_results = {
            "Group A (SPL-5b Rules)": create_sample_backtest_result(total_return=0.12),
            "Group B (SPL-6b Optimizer λ=0)": create_sample_backtest_result(total_return=0.15),
            "Group B+ (SPL-6b + Smoothing λ=2.0)": create_sample_backtest_result(total_return=0.16),
            "Group C (SPL-5a + SPL-6b)": create_sample_backtest_result(total_return=0.18),
        }

        # 添加一些分析结果
        analyzer.compare_weight_smoothing(
            group_results["Group B (SPL-6b Optimizer λ=0)"],
            group_results["Group B+ (SPL-6b + Smoothing λ=2.0)"],
            lambda_value=2.0
        )

        # 生成报告
        report = analyzer.generate_comparison_report(group_results)

        print(f"  报告长度: {len(report)} 字符")
        print(f"\n  报告预览:")
        print("  " + "\n  ".join(report.split("\n")[:30]))

        # 验证报告内容
        assert "# 反事实分析对照实验报告" in report, "应该包含标题"
        assert "Group A" in report, "应该包含 Group A"
        assert "Group B" in report, "应该包含 Group B"
        assert "概览" in report, "应该包含概览表格"
        assert "详细分析" in report, "应该包含详细分析"
        assert "结论" in report, "应该包含结论"

        # 保存报告
        report_path = analyzer.save_report(report, "test_comparison_report.md")
        assert report_path.exists(), "报告文件应该存在"

        print(f"\n  ✅ 对照实验报告生成测试通过")
        print(f"  报告已保存: {report_path}")

    return True


def test_export_results_json():
    """测试5: 导出 JSON 结果"""
    print("\n" + "="*70)
    print("测试5: 导出 JSON 结果")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = CounterfactualAnalyzer(storage_path=temp_dir)

        # 添加一些结果
        backtest_result = create_sample_backtest_result(total_return=0.15, rejected_count=5)
        analyzer.analyze_rejected_trades(backtest_result, {})

        # 导出 JSON
        json_path = analyzer.export_results_json("test_results.json")

        assert json_path.exists(), "JSON 文件应该存在"

        # 验证内容
        with open(json_path) as f:
            data = json.load(f)
            assert "timestamp" in data, "应该包含 timestamp"
            assert "results" in data, "应该包含 results"
            assert len(data["results"]) == 1, "应该有 1 个结果"
            assert data["results"][0]["type"] == "rejected_trades", "结果类型应该正确"

        print(f"  ✅ 导出 JSON 结果测试通过")
        print(f"  文件已保存: {json_path}")

    return True


def test_convenience_function():
    """测试6: 便捷函数"""
    print("\n" + "="*70)
    print("测试6: 便捷函数")
    print("="*70)

    backtest_result = create_sample_backtest_result(total_return=0.15, rejected_count=8)

    result = analyze_rejected_trades_impact(backtest_result, {})

    print(f"  原始收益: {result['original_return']*100:.2f}%")
    print(f"  如果接受: {result['if_accepted_return']*100:.2f}%")
    print(f"  收益差: {result['return_delta']*100:+.2f}%")
    print(f"  被拒绝数: {result['rejected_count']}")
    print(f"  本该盈利: {result['would_be_profitable']}")
    print(f"  本该亏损: {result['would_be_loss']}")

    # 验证
    assert "original_return" in result, "应该包含原始收益"
    assert "if_accepted_return" in result, "应该包含假设收益"
    assert "return_delta" in result, "应该包含收益差"
    assert result["rejected_count"] == 8, "拒绝数量应该正确"

    print(f"\n  ✅ 便捷函数测试通过")

    return True


def test_counterfactual_result_schema():
    """测试7: CounterfactualResult schema"""
    print("\n" + "="*70)
    print("测试7: CounterfactualResult schema")
    print("="*70)

    result = CounterfactualResult(
        type=CounterfactualType.WHAT_IF_SCENARIO,
        description="测试场景",
        baseline_return=0.10,
        counterfactual_return=0.12,
        return_delta=0.02,
        affected_trades=50,
        metadata={"test": "value"}
    )

    # 检查 to_dict
    result_dict = result.to_dict()

    assert result_dict["type"] == "what_if_scenario", "类型应该正确转换"
    assert result_dict["description"] == "测试场景", "描述应该正确"
    assert result_dict["baseline_return"] == 0.10, "基准收益应该正确"
    assert result_dict["counterfactual_return"] == 0.12, "反事实收益应该正确"
    assert result_dict["return_delta"] == 0.02, "收益差应该正确"
    assert result_dict["affected_trades"] == 50, "影响交易数应该正确"
    assert result_dict["metadata"]["test"] == "value", "元数据应该正确"
    assert "timestamp" in result_dict, "应该包含时间戳"

    print(f"  ✅ CounterfactualResult schema 测试通过")

    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("P2-2: 反事实分析完善测试")
    print("="*70)

    tests = [
        ("被拒绝交易分析", test_rejected_trades_analysis),
        ("权重平滑对比", test_weight_smoothing_comparison),
        ("假设场景分析", test_what_if_scenario),
        ("对照实验报告生成", test_comparison_report_generation),
        ("导出 JSON 结果", test_export_results_json),
        ("便捷函数", test_convenience_function),
        ("CounterfactualResult schema", test_counterfactual_result_schema),
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
