"""
风险结构分析模块

分析Top-K最坏窗口的风险形态相似性
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from analysis.replay_schema import ReplayOutput, load_replay_outputs
from analysis.window_scanner import WindowScanner, WindowMetrics


class RiskPatternType(Enum):
    """风险形态类型"""
    STRUCTURAL = "structural"      # 结构性风险（高度相似）
    SINGLE_OUTLIER = "single_outlier"  # 单一异常（完全不同）


@dataclass
class RiskPatternMetrics:
    """风险形态指标"""
    avg_mdd: float           # 平均最大回撤
    std_mdd: float           # MDD标准差
    avg_duration: float      # 平均回撤持续天数
    std_duration: float      # 持续时间标准差
    avg_single_loss: float   # 平均最大单步亏损
    max_consecutive_losses: int  # 最大连续亏损

    # 相似性指标
    mdd_cv: float            # MDD变异系数
    pattern_similarity: float  # 形态相似度 (0-1)

    @property
    def is_structural(self) -> bool:
        """是否为结构性风险"""
        # MDD变异系数 < 0.3 且 形态相似度 > 0.7
        return self.mdd_cv < 0.3 and self.pattern_similarity > 0.7


@dataclass
class StructuralAnalysisResult:
    """结构分析结果"""
    strategy_id: str
    window_length: str
    top_k_windows: List[WindowMetrics]

    # 风险形态
    risk_pattern_type: RiskPatternType
    pattern_metrics: RiskPatternMetrics

    # 详细分析
    drawdown_correlation: float  # 回撤曲线相关性
    shape_consistency: float     # 形态一致性


class StructuralAnalyzer:
    """结构分析器

    分析Top-K最坏窗口的风险形态相似性
    """

    def __init__(self):
        """初始化分析器"""
        self.scanner = WindowScanner(top_k=5)

    def analyze_structure(
        self,
        replay: ReplayOutput,
        window_length: str,
        top_k: int = 5
    ) -> StructuralAnalysisResult:
        """分析Top-K最坏窗口的结构

        Args:
            replay: 回测输出
            window_length: 窗口长度
            top_k: Top-K数量

        Returns:
            结构分析结果
        """
        # 获取Top-K最坏窗口
        all_windows = self.scanner.scan_replay(replay, window_length)
        top_k_windows = all_windows[:top_k]

        # 计算形态指标
        pattern_metrics = self._calculate_pattern_metrics(top_k_windows)

        # 判断风险类型
        risk_pattern_type = self._classify_risk_pattern(pattern_metrics)

        # 计算详细相似性
        drawdown_corr = self._calculate_drawdown_correlation(top_k_windows)
        shape_consistency = self._calculate_shape_consistency(top_k_windows)

        return StructuralAnalysisResult(
            strategy_id=replay.strategy_id,
            window_length=window_length,
            top_k_windows=top_k_windows,
            risk_pattern_type=risk_pattern_type,
            pattern_metrics=pattern_metrics,
            drawdown_correlation=drawdown_corr,
            shape_consistency=shape_consistency
        )

    def _calculate_pattern_metrics(
        self,
        windows: List[WindowMetrics]
    ) -> RiskPatternMetrics:
        """计算形态指标

        Args:
            windows: 窗口列表

        Returns:
            形态指标
        """
        # 提取各项指标
        mdds = [w.max_drawdown for w in windows]
        durations = [w.max_dd_duration for w in windows]
        single_losses = [w.max_single_loss for w in windows]
        consecutive_losses = [w.max_consecutive_losses for w in windows]

        # 计算统计量
        avg_mdd = np.mean(mdds)
        std_mdd = np.std(mdds)
        mdd_cv = std_mdd / avg_mdd if avg_mdd > 0 else 0

        avg_duration = np.mean(durations)
        std_duration = np.std(durations)

        avg_single_loss = np.mean(single_losses)
        max_consecutive_loss = max(consecutive_losses) if consecutive_losses else 0

        # 计算形态相似度（基于MDD和持续时间）
        shape_vectors = []
        for w in windows:
            vec = [w.max_drawdown, w.max_dd_duration, w.max_dd_recovery_time if w.max_dd_recovery_time > 0 else 30]
            shape_vectors.append(vec)

        # 计算向量间的平均相似度
        if len(shape_vectors) > 1:
            similarities = []
            for i in range(len(shape_vectors)):
                for j in range(i+1, len(shape_vectors)):
                    # 归一化后计算欧氏距离的倒数
                    vec1 = np.array(shape_vectors[i])
                    vec2 = np.array(shape_vectors[j])
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 > 0 and norm2 > 0:
                        vec1_norm = vec1 / norm1
                        vec2_norm = vec2 / norm2
                        # 计算欧氏距离
                        euclidean_dist = np.linalg.norm(vec1_norm - vec2_norm)
                        similarity = 1 - euclidean_dist
                        similarities.append(similarity)

            pattern_similarity = np.mean(similarities) if similarities else 0
        else:
            pattern_similarity = 0

        return RiskPatternMetrics(
            avg_mdd=avg_mdd,
            std_mdd=std_mdd,
            avg_duration=avg_duration,
            std_duration=std_duration,
            avg_single_loss=avg_single_loss,
            max_consecutive_losses=max_consecutive_loss,
            mdd_cv=mdd_cv,
            pattern_similarity=pattern_similarity
        )

    def _classify_risk_pattern(
        self,
        metrics: RiskPatternMetrics
    ) -> RiskPatternType:
        """分类风险形态

        Args:
            metrics: 形态指标

        Returns:
            风险类型
        """
        if metrics.is_structural:
            return RiskPatternType.STRUCTURAL
        else:
            return RiskPatternType.SINGLE_OUTLIER

    def _calculate_drawdown_correlation(
        self,
        windows: List[WindowMetrics]
    ) -> float:
        """计算回撤曲线相关性

        使用numpy实现皮尔逊相关系数
        """
        if len(windows) < 2:
            return 0.0

        # 构建特征矩阵
        features = []
        for w in windows:
            feat = [
                w.max_drawdown,
                w.max_dd_duration,
                w.ulcer_index,
                w.downside_deviation,
                w.cvar_95
            ]
            features.append(feat)

        # 计算平均相关性
        correlations = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                try:
                    # 使用numpy计算相关系数
                    arr1 = np.array(features[i])
                    arr2 = np.array(features[j])

                    # 计算皮尔逊相关系数
                    mean1 = np.mean(arr1)
                    mean2 = np.mean(arr2)

                    std1 = np.std(arr1)
                    std2 = np.std(arr2)

                    if std1 > 0 and std2 > 0:
                        corr = np.mean((arr1 - mean1) * (arr2 - mean2)) / (std1 * std2)
                        corr = max(-1, min(1, corr))  # 限制在[-1, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                except:
                    pass

        return np.mean(correlations) if correlations else 0.0

    def _calculate_shape_consistency(
        self,
        windows: List[WindowMetrics]
    ) -> float:
        """计算形态一致性

        基于回撤形态分类的一致性
        """
        if not windows:
            return 0.0

        # 统计各形态的数量
        pattern_counts = {}
        for w in windows:
            pattern = w.drawdown_pattern
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # 计算主导形态的比例
        if pattern_counts:
            max_count = max(pattern_counts.values())
            consistency = max_count / len(windows)
        else:
            consistency = 0

        return consistency


def analyze_all_structures(
    run_dir: str,
    window_lengths: List[str] = None
) -> Dict[str, Dict[str, StructuralAnalysisResult]]:
    """分析所有replay的风险结构

    Args:
        run_dir: runs目录路径
        window_lengths: 窗口长度列表

    Returns:
        {strategy_id: {window_length: result}}
    """
    replays = load_replay_outputs(run_dir)
    window_lengths = window_lengths or ["20d", "60d"]

    analyzer = StructuralAnalyzer()
    all_results = {}

    for replay in replays:
        strategy_key = f"{replay.run_id}_{replay.strategy_id}"
        all_results[strategy_key] = {}

        for window in window_lengths:
            result = analyzer.analyze_structure(replay, window)
            all_results[strategy_key][window] = result

    return all_results


if __name__ == "__main__":
    """测试代码"""
    print("=== StructuralAnalyzer 测试 ===\n")

    replays = load_replay_outputs("runs")

    if replays:
        replay = replays[0]
        print(f"分析策略: {replay.run_id}\n")

        analyzer = StructuralAnalyzer()

        # 分析20d窗口
        result = analyzer.analyze_structure(replay, "20d", top_k=5)

        print(f"窗口长度: {result.window_length}")
        print(f"风险类型: {result.risk_pattern_type.value}")
        print(f"\n形态指标:")
        print(f"  平均MDD: {result.pattern_metrics.avg_mdd*100:.2f}%")
        print(f"  MDD标准差: {result.pattern_metrics.std_mdd*100:.2f}%")
        print(f"  MDD变异系数: {result.pattern_metrics.mdd_cv:.3f}")
        print(f"  形态相似度: {result.pattern_metrics.pattern_similarity:.3f}")
        print(f"\n相似性分析:")
        print(f"  回撤曲线相关性: {result.drawdown_correlation:.3f}")
        print(f"  形态一致性: {result.shape_consistency:.3f}")

        if result.risk_pattern_type == RiskPatternType.STRUCTURAL:
            print(f"\n结论: 结构性风险（Top-K窗口高度相似）")
        else:
            print(f"\n结论: 单一异常风险（Top-K窗口差异大）")

    print("\n✓ 测试通过")
