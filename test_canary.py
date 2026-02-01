"""
运行 Canary Test 验证漂移检测功能

这个测试故意引入一个"回归"（改变 top_k 参数）来验证
测试框架能够检测到漂移。
"""

from analysis.baseline_manager import BaselineManager
from analysis.replay_schema import load_replay_outputs
from tests.risk_regression.risk_baseline_test import RiskBaselineTests

def main():
    print("=" * 70)
    print("SPL-4c Canary Test - 验证漂移检测功能")
    print("=" * 70)

    # 1. 加载基线和当前数据
    print("\n1. 加载数据...")
    mgr = BaselineManager()
    snapshot = mgr.load_all_baselines()

    replays = load_replay_outputs('runs')
    replay_map = {r.run_id: r for r in replays}

    print(f"   基线数: {len(snapshot.baselines)}")
    print(f"   当前数据数: {len(replays)}")

    # 2. 创建测试套件
    print("\n2. 创建测试套件...")
    test_suite = RiskBaselineTests(tolerance_pct=0.02)
    print("   ✓ 测试套件已创建")

    # 3. 运行 Canary Test
    print("\n3. 运行 Canary Test...")
    print("-" * 70)

    canary_passed = 0
    canary_failed = 0

    for baseline in snapshot.baselines:
        print(f"\n测试: {baseline.run_id}")

        current = replay_map.get(baseline.run_id)
        if current is None:
            print(f"  ⏭️  SKIP: 无当前数据")
            continue

        # 运行 Canary Test (期望检测到漂移)
        result = test_suite.test_canary_drift_detection(
            baseline,
            current,
            expected_detection=True
        )

        print(f"  状态: {result.status}")
        print(f"  消息: {result.message}")

        if result.status == "PASS":
            canary_passed += 1
            print(f"  ✅ Canary 检测到漂移 - 测试框架工作正常")
        else:
            canary_failed += 1
            print(f"  ❌ Canary 失败 - 测试框架可能有问题")
            print(f"     详情: {result.details}")

    # 4. 总结
    print("\n" + "=" * 70)
    print("Canary Test 总结")
    print("=" * 70)
    total = canary_passed + canary_failed
    print(f"总计: {total}")
    print(f"通过: {canary_passed}")
    print(f"失败: {canary_failed}")

    if canary_failed == 0 and total > 0:
        print("\n✅ Canary Test 全部通过！")
        print("   漂移检测功能工作正常，测试框架可靠。")
    else:
        print(f"\n❌ {canary_failed} 个 Canary 失败")
        print("   需要检查测试框架是否正常工作。")

    print("=" * 70)


if __name__ == "__main__":
    main()
