"""
SPL-6a-E: CI/自动化集成 - Drift Report 生成（简化版）

在 CI 或定时���务中生成 drift report 并上传 artifact。
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.replay_schema import load_replay_outputs, ReplayOutput


def generate_simple_drift_report(
    runs_dir: str,
    output_dir: str = "reports/drift_detection"
) -> Dict[str, Any]:
    """生成简化的漂移检测报告

    Args:
        runs_dir: runs 目录
        output_dir: 输出目录

    Returns:
        漂移检测报告
    """
    print("=" * 60)
    print("SPL-6a-E: Drift Detection Pipeline (简化版)")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    all_replays = load_replay_outputs(runs_dir)
    print(f"加载 {len(all_replays)} 个 replay")

    if len(all_replays) < 2:
        print("数据不足，无法进行漂移检测")
        return {
            "report_date": datetime.now().isoformat(),
            "status": "SKIP",
            "reason": "Insufficient data",
            "overall_status": "SKIP"
        }

    # 简化：只计算基本统计指标
    baseline_replay = all_replays[0]
    current_replay = all_replays[-1]

    print(f"\nBaseline: {baseline_replay.run_id}")
    print(f"Current: {current_replay.run_id}")

    # 计算基本指标
    baseline_df = baseline_replay.to_dataframe()
    current_df = current_replay.to_dataframe()

    baseline_pnl = baseline_df['step_pnl'].values
    current_pnl = current_df['step_pnl'].values

    # 基本统计
    baseline_mean = np.mean(baseline_pnl)
    current_mean = np.mean(current_pnl)
    baseline_std = np.std(baseline_pnl)
    current_std = np.std(current_pnl)

    # 计算漂移
    mean_change = abs(current_mean - baseline_mean)
    mean_change_pct = mean_change / (abs(baseline_mean) + 1e-10)

    std_change = abs(current_std - baseline_std)
    std_change_pct = std_change / (baseline_std + 1e-10)

    # 判断状态
    status = "GREEN"
    if mean_change_pct > 0.5:  # 50% 变化
        status = "YELLOW"
    if mean_change_pct > 1.0:  # 100% 变化
        status = "RED"

    # 生成报告
    report = {
        "report_date": datetime.now().isoformat(),
        "baseline_run_id": baseline_replay.run_id,
        "current_run_id": current_replay.run_id,
        "overall_status": status,
        "metrics": {
            "mean": {
                "baseline": float(baseline_mean),
                "current": float(current_mean),
                "change": float(current_mean - baseline_mean),
                "change_pct": float(mean_change_pct)
            },
            "std": {
                "baseline": float(baseline_std),
                "current": float(current_std),
                "change": float(current_std - baseline_std),
                "change_pct": float(std_change_pct)
            }
        },
        "recalibration_triggered": status == "RED",
        "recalibration_reason": f"均值变化 {mean_change_pct:.1%}" if status != "GREEN" else "无显著漂移"
    }

    # 保存报告
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n报告已保存: {report_file}")

    # 输出总结
    print("\n" + "=" * 60)
    print("漂移检测总结")
    print("=" * 60)
    print(f"总体状态: {report['overall_status']}")
    print(f"均值变化: {mean_change_pct:.1%}")
    print(f"标准差变化: {std_change_pct:.1%}")
    print(f"再标定触发: {report['recalibration_triggered']}")
    if report['recalibration_triggered']:
        print(f"触发原因: {report['recalibration_reason']}")
    print("=" * 60)

    return report


def main():
    """主函数"""
    runs_dir = str(project_root / "runs")

    if not Path(runs_dir).exists():
        print("runs 目录不存在")
        return 0

    report = generate_simple_drift_report(runs_dir)

    # 根据状态返回退出码
    if report.get("status") == "SKIP":
        return 0
    elif report.get("recalibration_triggered", False):
        print("\n🚨 检测到显著漂移！建议运行再标定流程")
        return 1  # RED 状态
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
