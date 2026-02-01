"""
SPL-6a-E: CI/自动化集成 - Drift Report 生成

在 CI 或定时任务中生成 drift report 并上传 artifact。
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.replay_schema import load_replay_outputs, ReplayOutput
from analysis.drift_objects import DriftObjectRegistry, DriftObjectConfig
from analysis.drift_metrics import DriftMetricsEvaluator
from analysis.drift_threshold_evaluator import DriftThresholdEvaluator, DriftReportGenerator


class DriftDetectionPipeline:
    """漂移检测流程"""

    def __init__(self):
        """初始化流程"""
        self.registry = DriftObjectRegistry()
        self.evaluator = DriftMetricsEvaluator()
        self.threshold_eval = DriftThresholdEvaluator()
        self.report_gen = DriftReportGenerator(self.threshold_eval)

    def collect_baseline_data(
        self,
        replay: ReplayOutput,
        config: DriftObjectConfig
    ) -> np.ndarray:
        """收集基线数据

        Args:
            replay: 回测输出
            config: 漂移对象配置

        Returns:
            数据数组
        """
        df = replay.to_dataframe()

        if config.name == "returns":
            return df['step_pnl'].values
        elif config.name == "volatility":
            # 计算 rolling volatility
            return df['step_pnl'].rolling(20).std().dropna().values
        elif config.name == "worst_case_returns":
            # Worst-case returns (P5, P10)
            return df['step_pnl'].values
        else:
            # 默认返回 step_pnl
            return df['step_pnl'].values if 'step_pnl' in df.columns else np.array([])

    def run_drift_detection(
        self,
        runs_dir: str,
        output_dir: str = "reports/drift_detection"
    ) -> Dict[str, Any]:
        """运行完整的漂移检测

        Args:
            runs_dir: runs 目录
            output_dir: 输出目录

        Returns:
            漂移检测报告
        """
        print("=" * 60)
        print("SPL-6a-E: Drift Detection Pipeline")
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
                "reason": "Insufficient data"
            }

        # 选择基线和当前数据
        baseline_replay = all_replays[0]
        current_replay = all_replays[-1]

        print(f"Baseline: {baseline_replay.run_id}")
        print(f"Current: {current_replay.run_id}")

        # 运行所有对象的漂移检测
        print("\n检测漂移...")
        drift_results = []

        # 选择关键对象进行检测（优先检测 critical 和 high）
        priority_objects = (
            self.registry.list_objects()[:10]  # 检测前10个对象
        )

        for object_name in priority_objects:
            config = self.registry.get_object(object_name)
            print(f"\n  {object_name}:")

            # 收集数据
            baseline_data = self.collect_baseline_data(baseline_replay, config)
            current_data = self.collect_baseline_data(current_replay, config)

            if len(baseline_data) == 0 or len(current_data) == 0:
                print(f"    SKIP: 数据不足")
                continue

            # 计算每个指标
            for metric_name in config.metrics[:2]:  # 每个对象计算前2个指标
                try:
                    result = self.evaluator.evaluate(
                        metric_name,
                        baseline_data,
                        current_data,
                        config
                    )

                    # 添加对象名
                    from analysis.drift_objects import DriftResult
                    drift_result = DriftResult(
                        object_name=object_name,
                        category=config.category,
                        metric_name=metric_name,
                        drift_value=result.value,
                        p_value=result.p_value,
                        status=result.status,
                        threshold=result.threshold_yellow,
                        message=result.message
                    )

                    # 添加快照（简化）
                    drift_result.baseline_snapshot = type('obj', (object,), {
                        'to_dict': lambda: {"sample_size": len(baseline_data)}
                    })()
                    drift_result.current_snapshot = type('obj', (object,), {
                        'to_dict': lambda: {"sample_size": len(current_data)}
                    })()

                    # 评估阈值
                    drift_result = self.threshold_eval.evaluate_drift_result(drift_result, config)

                    drift_results.append(drift_result)

                    print(f"    {metric_name}: {result.status} ({result.value:.4f})")

                except Exception as e:
                    print(f"    ERROR: {metric_name} - {e}")

        # 检查再标定触发
        print("\n检查再标定触发条件...")
        recal_trigger = self.threshold_eval.check_recalibration_trigger(
            drift_results,
            historical_red_count={},  # 简化：不追踪历史
            last_recalibration_date=None
        )

        print(f"触发再标定: {recal_trigger.triggered}")
        if recal_trigger.triggered:
            print(f"  原因: {recal_trigger.reason}")

        # 生成汇总报告
        print("\n生成汇总报告...")
        summary_report = self.report_gen.generate_summary_report(
            drift_results,
            recal_trigger
        )

        # 保存报告
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(summary_report, f, indent=2)

        # 保存详细结果
        details_file = output_path / f"drift_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(details_file, 'w') as f:
            json.dump({
                "drift_results": [r.to_dict() for r in drift_results],
                "recalibration_trigger": {
                    "triggered": recal_trigger.triggered,
                    "reason": recal_trigger.reason,
                    "details": recal_trigger.details
                }
            }, f, indent=2)

        print(f"\n报告已保存:")
        print(f"  汇总报告: {report_file}")
        print(f"  详细结果: {details_file}")

        # 输出总结
        print("\n" + "=" * 60)
        print("漂移检测总结")
        print("=" * 60)
        print(f"总体状态: {summary_report['overall_status']}")
        print(f"检测对象: {summary_report['total_objects']}")
        print(f"状态分布: {summary_report['status_counts']}")
        print(f"再标定触发: {summary_report['recalibration_triggered']}")
        print("=" * 60)

        return summary_report


def main():
    """主函数"""
    pipeline = DriftDetectionPipeline()

    runs_dir = str(project_root / "runs")

    if not Path(runs_dir).exists():
        print("runs 目录不存在")
        return 1

    report = pipeline.run_drift_detection(runs_dir)

    # 根据状态返回退出码
    if report.get("status") == "SKIP":
        return 0
    elif report.get("recalibration_triggered", False):
        return 1  # RED 状态触发再标定候选
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
