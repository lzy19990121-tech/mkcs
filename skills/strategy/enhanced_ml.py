"""
增强的ML策略

实现：
- 严格的时间序列训练流程
- 模型输出从三分类升级为期望收益+风险
- 预测漂移监控
- ML专用稳健性测试
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import json

import numpy as np

from core.models import Bar, Position
from core.strategy_models import (
    StrategySignal,
    StrategyState,
    RegimeType,
    RegimeInfo,
    RiskHints
)
from skills.strategy.enhanced_base import EnhancedStrategy
from skills.analysis.regime_detector import RegimeDetector, RegimeDetectorConfig
from skills.strategy.ml_strategy import (
    FeatureEngineer,
    FeatureConfig,
    MLModel,
    RandomForestModel
)

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """ML预测结果（增强版）"""
    expected_return: float        # 期望收益率
    confidence: float             # 预测置信度 (0-1)
    risk_estimate: float          # 风险估计（标准差）
    prediction_type: str          # 预测类型: LONG/SHORT/NEUTRAL
    model_version: str = "1.0"
    feature_importance: Dict[str, float] = field(default_factory=dict)
    prediction_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceMetrics:
    """模型性能指标"""
    total_predictions: int = 0
    correct_predictions: int = 0
    long_predictions: int = 0
    short_predictions: int = 0
    neutral_predictions: int = 0

    # 实际表现
    long_accuracy: float = 0.0
    short_accuracy: float = 0.0

    # 预测分布
    avg_expected_return: float = 0.0
    avg_actual_return: float = 0.0

    # 漂移检测
    prediction_drift_score: float = 0.0
    last_drift_check: Optional[datetime] = None

    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


class TimeSeriesSplitter:
    """时间序列交叉验证

    确保训练时没有 look-ahead bias
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2
    ):
        """初始化分割器

        Args:
            n_splits: 分割数量
            train_size: 训练集比例
            val_size: 验证集比例
            test_size: 测试集比例
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def split(self, n_samples: int) -> List[Tuple[slice, slice, slice]]:
        """生成交叉验证分割

        Returns:
            [(train_idx, val_idx, test_idx), ...]
        """
        splits = []
        step = (n_samples - int(n_samples * self.train_size)) // (self.n_splits - 1)

        for i in range(self.n_splits):
            train_end = int(n_samples * self.train_size) + i * step
            val_end = train_end + int(n_samples * self.val_size)

            if val_end >= n_samples:
                break

            train = slice(0, train_end)
            val = slice(train_end, val_end)
            test = slice(val_end, min(val_end + int(n_samples * self.test_size), n_samples))

            splits.append((train, val, test))

        return splits

    def single_split(self, n_samples: int) -> Tuple[slice, slice, slice]:
        """单次分割（用于最终训练）"""
        train_end = int(n_samples * self.train_size)
        val_end = train_end + int(n_samples * self.val_size)

        return (
            slice(0, train_end),
            slice(train_end, val_end),
            slice(val_end, n_samples)
        )


class PredictionDriftMonitor:
    """预测漂移监控

    监控模型预测分布和实际命中率
    """

    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.3,
        accuracy_threshold: float = 0.45
    ):
        """初始化监控器

        Args:
            window_size: 监控窗口大小
            drift_threshold: 漂移阈值（KL散度）
            accuracy_threshold: 最低准确率阈值
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.accuracy_threshold = accuracy_threshold

        self.predictions: deque = deque(maxlen=window_size)
        self.actuals: deque = deque(maxlen=window_size)
        self.expected_returns: deque = deque(maxlen=window_size)
        self.actual_returns: deque = deque(maxlen=window_size)

        self.baseline_distribution: Optional[Dict[str, float]] = None

    def add_prediction(
        self,
        prediction: MLPrediction,
        actual_return: Optional[float] = None
    ):
        """添加预测记录"""
        self.predictions.append(prediction)
        if actual_return is not None:
            self.actuals.append(actual_return)
            self.actual_returns.append(actual_return)

        self.expected_returns.append(prediction.expected_return)

    def set_baseline(self, predictions: List[MLPrediction]):
        """设置基准分布（从训练集统计）"""
        dist = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
        for p in predictions:
            dist[p.prediction_type] += 1

        total = len(predictions)
        self.baseline_distribution = {
            k: v / total for k, v in dist.items()
        }

    def check_drift(self) -> Dict[str, Any]:
        """检查预测漂移

        Returns:
            漂移检测结果
        """
        if not self.predictions or len(self.predictions) < 20:
            return {"drift_detected": False, "reason": "数据不足"}

        # 1. 分布漂移检测
        current_dist = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
        for p in self.predictions:
            current_dist[p.prediction_type] += 1

        total = len(self.predictions)
        current_dist = {k: v / total for k, v in current_dist.items()}

        # 计算 KL 散度
        kl_div = 0.0
        if self.baseline_distribution:
            for k in current_dist:
                p = current_dist[k] + 1e-10
                q = self.baseline_distribution.get(k, 0.001) + 1e-10
                kl_div += p * np.log(p / q)

        # 2. 准确率检测
        accuracy = 0.0
        if len(self.actuals) > 0:
            correct = sum(
                1 for p, a in zip(self.predictions, self.actuals)
                if (p.expected_return > 0 and a > 0) or
                   (p.expected_return < 0 and a < 0) or
                   (abs(p.expected_return) < 0.01 and abs(a) < 0.01)
            )
            accuracy = correct / len(self.actuals)

        # 3. 期望收益偏差
        expected_mean = np.mean(list(self.expected_returns)) if self.expected_returns else 0
        actual_mean = np.mean(list(self.actual_returns)) if self.actual_returns else 0
        return_bias = abs(expected_mean - actual_mean)

        # 综合判断
        drift_detected = (
            kl_div > self.drift_threshold or
            accuracy < self.accuracy_threshold or
            return_bias > 0.02
        )

        reasons = []
        if kl_div > self.drift_threshold:
            reasons.append(f"分布漂移: KL={kl_div:.3f}")
        if accuracy < self.accuracy_threshold:
            reasons.append(f"准确率过低: {accuracy:.2%}")
        if return_bias > 0.02:
            reasons.append(f"收益偏差: {return_bias:.3f}")

        return {
            "drift_detected": drift_detected,
            "kl_divergence": kl_div,
            "accuracy": accuracy,
            "return_bias": return_bias,
            "reasons": reasons,
            "current_distribution": current_dist
        }

    def get_recent_accuracy(self, window: int = 20) -> float:
        """获取最近准确率"""
        if len(self.actuals) < window:
            return 0.0

        recent_preds = list(self.predictions)[-window:]
        recent_actuals = list(self.actuals)[-window:]

        correct = sum(
            1 for p, a in zip(recent_preds, recent_actuals)
            if (p.expected_return > 0 and a > 0) or
               (p.expected_return < 0 and a < 0) or
               (abs(p.expected_return) < 0.01 and abs(a) < 0.01)
        )

        return correct / len(recent_actuals)


class RobustnessTester:
    """ML模型稳健性测试

    测试模型对扰动、延迟、滑点的敏感性
    """

    def __init__(self):
        self.test_results: Dict[str, Any] = {}

    def test_feature_perturbation(
        self,
        model: MLModel,
        features: np.ndarray,
        noise_levels: List[float] = [0.01, 0.05, 0.1]
    ) -> Dict[str, Any]:
        """特征扰动测试

        在特征上添加噪声，观察预测变化
        """
        results = {}

        # 原始预测
        original_pred = model.predict(features)
        original_proba = model.predict_proba(features)

        for noise_level in noise_levels:
            # 添加高斯噪声
            noise = np.random.normal(0, noise_level, features.shape)
            noisy_features = features + noise

            # 预测
            perturbed_pred = model.predict(noisy_features)
            perturbed_proba = model.predict_proba(noisy_features)

            # 计算预测变化率
            change_rate = np.sum(original_pred != perturbed_pred) / len(original_pred)

            # 计算概率平均变化
            prob_change = np.mean(np.abs(original_proba - perturbed_proba))

            results[f"noise_{noise_level}"] = {
                "change_rate": float(change_rate),
                "avg_prob_change": float(prob_change),
                "robustness_score": 1.0 - change_rate  # 越高越稳健
            }

        self.test_results["feature_perturbation"] = results
        return results

    def test_delay_sensitivity(
        self,
        model: MLModel,
        features: np.ndarray,
        delays: List[int] = [1, 2, 3, 5]
    ) -> Dict[str, Any]:
        """延迟敏感性测试

        模拟信号延迟n个周期后的表现
        """
        results = {}

        if len(features) < max(delays) + 10:
            return {"error": "数据不足"}

        # 原始预测（使用最新特征）
        current_pred = model.predict(features[-1:])[0]

        for delay in delays:
            if delay >= len(features):
                continue

            # 使用delay周期前的特征
            delayed_pred = model.predict(features[-(delay+1):-delay])[0]

            # 判断预测是否一致
            consistent = (current_pred == delayed_pred)
            results[f"delay_{delay}"] = {
                "consistent": bool(consistent),
                "consistency_rate": 1.0 if consistent else 0.0
            }

        self.test_results["delay_sensitivity"] = results
        return results

    def test_slippage_impact(
        self,
        signals: List[float],
        slippage_rates: List[float] = [0.001, 0.005, 0.01]
    ) -> Dict[str, Any]:
        """滑点影响测试

        测试不同滑点水平下的预期收益
        """
        results = {}

        for slip_rate in slippage_rates:
            adjusted_returns = []
            for signal in signals:
                # 买入时价格更高，卖出时价格更低
                if signal > 0:
                    adjusted = signal * (1 - slip_rate * 2)
                else:
                    adjusted = signal * (1 - slip_rate * 2)
                adjusted_returns.append(adjusted)

            total_return = sum(adjusted_returns)
            original_return = sum(signals)

            results[f"slippage_{slip_rate}"] = {
                "total_return": float(total_return),
                "return_loss": float(original_return - total_return),
                "return_loss_ratio": float((original_return - total_return) / abs(original_return)) if original_return != 0 else 0
            }

        self.test_results["slippage_impact"] = results
        return results


class EnhancedMLStrategy(EnhancedStrategy):
    """增强版ML策略

    特点：
    1. 严格时间序列训练（无look-ahead）
    2. 输出期望收益+风险（而非三分类）
    3. 预测漂移监控
    4. 稳健性测试
    """

    def __init__(
        self,
        model: Optional[MLModel] = None,
        feature_config: FeatureConfig = None,
        enable_drift_monitor: bool = True,
        disable_on_drift: bool = True,
        max_position_ratio: float = 0.15
    ):
        """初始化策略

        Args:
            model: ML模型
            feature_config: 特征配置
            enable_drift_monitor: 是否启用漂移监控
            disable_on_drift: 漂移时是否禁用策略
            max_position_ratio: 最大仓位比例
        """
        super().__init__(
            name="ml",
            enable_regime_gating=True
        )

        self.model = model or RandomForestModel(n_estimators=100)
        self.feature_engineer = FeatureEngineer(feature_config)
        self.enable_drift_monitor = enable_drift_monitor
        self.disable_on_drift = disable_on_drift
        self.max_position_ratio = max_position_ratio

        # 漂移监控
        self.drift_monitor = PredictionDriftMonitor() if enable_drift_monitor else None
        self.drift_detected = False

        # 性能指标
        self.metrics = ModelPerformanceMetrics()

        # 时间序列分割器
        self.splitter = TimeSeriesSplitter()

    def get_min_bars_required(self) -> int:
        return self.feature_engineer.config.lookback_window + 10

    def _generate_strategy_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None,
        regime_info: Optional[RegimeInfo] = None
    ) -> List[StrategySignal]:
        """生成策略信号"""
        if len(bars) < self.get_min_bars_required():
            return []

        # 检查漂移
        if self.drift_detected:
            return []

        try:
            # 提取特征
            features = self.feature_engineer.extract_features(bars)
            if len(features) == 0:
                return []

            latest_features = features[-1:]

            # 获取预测（使用MLPrediction）
            prediction = self._predict_with_confidence(latest_features)

            # 更新漂移监控
            if self.drift_monitor:
                self.drift_monitor.add_prediction(prediction)

                # 定期检查漂移
                drift_result = self.drift_monitor.check_drift()
                if drift_result["drift_detected"]:
                    self.drift_detected = True
                    logger.warning(f"ML策略检测到预测漂移: {drift_result['reasons']}")
                    return []

            # 根据预测生成信号
            current_bar = bars[-1]
            current_price = current_bar.close

            # 根据期望收益和置信度计算仓位
            target_weight = self._calculate_position_from_prediction(
                prediction,
                regime_info
            )

            if target_weight == 0:
                return []

            # 计算目标持仓数量
            max_qty = 1000
            target_position = int(abs(target_weight) * max_qty)
            target_position = (target_position // 100) * 100

            if target_position < 100:
                return []

            # 确定方向
            raw_action = "BUY" if target_weight > 0 else "SELL"

            # 风险提示（基于预测的风险估计）
            atr = self._calculate_atr(bars, 14)
            risk_hints = RiskHints(
                stop_loss=current_price - (atr * Decimal("2")),
                take_profit=current_price + (atr * Decimal("2")),
                trailing_stop=Decimal("0.025"),
                position_limit=self.max_position_ratio * abs(target_weight)
            )

            signal = StrategySignal(
                symbol=current_bar.symbol,
                timestamp=current_bar.timestamp,
                target_position=target_position if target_weight > 0 else -target_position,
                target_weight=target_weight,
                confidence=prediction.confidence,
                reason=f"ML预测: {prediction.prediction_type}, 期望收益={prediction.expected_return:.3f}",
                regime=regime_info.regime if regime_info else RegimeType.UNKNOWN,
                risk_hints=risk_hints,
                raw_action=raw_action,
                metadata={
                    "expected_return": prediction.expected_return,
                    "risk_estimate": prediction.risk_estimate,
                    "model_version": prediction.model_version
                }
            )

            return [signal]

        except Exception as e:
            logger.error(f"ML策略信号生成失败: {e}")
            return []

    def _predict_with_confidence(self, features: np.ndarray) -> MLPrediction:
        """获取带置信度的预测"""
        # 获取预测类别
        prediction_class = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        # 转换为期望收益
        # 假设: 0=跌, 1=平, 2=涨
        if prediction_class == 2:  # 涨
            expected_return = probabilities[2] * 0.03  # 期望收益3%
            prediction_type = "LONG"
        elif prediction_class == 0:  # 跌
            expected_return = -probabilities[0] * 0.03
            prediction_type = "SHORT"
        else:  # 平
            expected_return = 0.0
            prediction_type = "NEUTRAL"

        # 置信度是最大概率
        confidence = float(np.max(probabilities))

        # 风险估计（概率分布的标准差）
        risk_estimate = float(np.std(probabilities))

        return MLPrediction(
            expected_return=expected_return,
            confidence=confidence,
            risk_estimate=risk_estimate,
            prediction_type=prediction_type
        )

    def _calculate_position_from_prediction(
        self,
        prediction: MLPrediction,
        regime_info: Optional[RegimeInfo]
    ) -> float:
        """根据预测计算仓位"""
        # 基础仓位：期望收益 * 置信度
        base_weight = (prediction.expected_return * 10) * prediction.confidence
        base_weight = max(-self.max_position_ratio, min(self.max_position_ratio, base_weight))

        # 绝对值太小则不交易
        if abs(base_weight) < 0.05:
            return 0.0

        # 市场状态调整
        if regime_info:
            vol_factor = 1.0 / max(regime_info.volatility_level, 0.5)
            base_weight *= vol_factor

        return max(-self.max_position_ratio, min(self.max_position_ratio, base_weight))

    def _calculate_atr(self, bars: List[Bar], period: int = 14) -> Decimal:
        """计算ATR"""
        if len(bars) < period + 1:
            return Decimal("1")

        true_ranges = []
        for i in range(1, len(bars)):
            prev_close = bars[i - 1].close
            current_high = bars[i].high
            current_low = bars[i].low

            tr = max(
                current_high - current_low,
                abs(current_high - prev_close),
                abs(current_low - prev_close)
            )
            true_ranges.append(tr)

        recent_tr = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
        atr = sum(recent_tr) / Decimal(len(recent_tr))

        return atr

    def train(
        self,
        bars: List[Bar],
        use_time_series_split: bool = True
    ) -> Dict[str, Any]:
        """训练模型（严格时间序列）

        Args:
            bars: 训练数据
            use_time_series_split: 是否使用时间序列交叉验证

        Returns:
            训练结果
        """
        logger.info(f"开始ML模型训练，数据量: {len(bars)}")

        # 提取特征
        features = self.feature_engineer.extract_features(bars)

        # 生成标签（未来收益）
        labels = self._generate_return_labels(bars)

        # 对齐
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]

        # 将连续收益转为分类
        # 收益 > 1% → 2 (涨)
        # 收益 < -1% → 0 (跌)
        # 其他 → 1 (平)
        class_labels = np.zeros(len(labels), dtype=int)
        class_labels[labels > 0.01] = 2
        class_labels[labels < -0.01] = 0
        class_labels[(labels >= -0.01) & (labels <= 0.01)] = 1

        if use_time_series_split:
            # 使用时间序列分割
            train_idx, val_idx, test_idx = self.splitter.single_split(len(features))

            X_train, y_train = features[train_idx], class_labels[train_idx]
            X_val, y_val = features[val_idx], class_labels[val_idx]
            X_test, y_test = features[test_idx], class_labels[test_idx]
        else:
            # 简单分割
            split = int(0.8 * len(features))
            X_train, y_train = features[:split], class_labels[:split]
            X_val, y_val = features[split:], class_labels[split:]
            X_test = y_test = None

        # 训练
        logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")
        self.model.train(X_train, y_train)

        # 验证
        val_pred = self.model.predict(X_val)
        val_acc = np.mean(val_pred == y_val)

        result = {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "val_accuracy": float(val_acc),
            "class_distribution": {
                "0": int(np.sum(y_train == 0)),
                "1": int(np.sum(y_train == 1)),
                "2": int(np.sum(y_train == 2))
            }
        }

        # 设置漂移监控基准
        if self.drift_monitor:
            train_predictions = [self._predict_with_confidence(X_train[i:i+1]) for i in range(min(100, len(X_train)))]
            self.drift_monitor.set_baseline(train_predictions)

        logger.info(f"训练完成, 验证准确率: {val_acc:.3f}")
        return result

    def _generate_return_labels(self, bars: List[Bar]) -> np.ndarray:
        """生成收益率标签（不look-ahead）"""
        horizon = self.feature_engineer.config.prediction_horizon
        returns = []

        for i in range(len(bars) - horizon):
            current_price = float(bars[i].close)
            future_price = float(bars[i + horizon].close)
            ret = (future_price - current_price) / current_price
            returns.append(ret)

        return np.array(returns)

    def update_drift_monitor(self, actual_return: float):
        """更新漂移监控（用实际收益）"""
        if self.drift_monitor:
            # 获取最后一次预测
            if self.drift_monitor.predictions:
                last_pred = list(self.drift_monitor.predictions)[-1]
                self.drift_monitor.add_prediction(last_pred, actual_return)

    def reset_drift_status(self):
        """重置漂移状态"""
        self.drift_detected = False


if __name__ == "__main__":
    """测试代码"""
    print("=== EnhancedMLStrategy 测试 ===\n")

    # 测试漂移监控
    print("1. 预测漂移监控:")
    monitor = PredictionDriftMonitor()

    # 设置基准
    baseline_preds = [
        MLPrediction(0.02, 0.8, 0.1, "LONG"),
        MLPrediction(-0.01, 0.6, 0.1, "SHORT"),
        MLPrediction(0.0, 0.5, 0.1, "NEUTRAL"),
    ] * 20
    monitor.set_baseline(baseline_preds)

    # 添加预测
    for i in range(50):
        pred = MLPrediction(0.01 * (i % 3 - 1), 0.7, 0.1, ["LONG", "SHORT", "NEUTRAL"][i % 3])
        actual = 0.01 * (i % 2)  # 模拟实际收益
        monitor.add_prediction(pred, actual)

    drift_result = monitor.check_drift()
    print(f"   漂移检测: {drift_result['drift_detected']}")
    print(f"   KL散度: {drift_result['kl_divergence']:.3f}")
    print(f"   准确率: {drift_result['accuracy']:.2%}")

    # 测试稳健性测试
    print("\n2. 稳健性测试:")
    tester = RobustnessTester()

    # 特征扰动
    features = np.random.randn(100, 20)
    model = RandomForestModel(n_estimators=10)
    model.train(features, np.random.randint(0, 3, 100))

    perturb_result = tester.test_feature_perturbation(model, features)
    print("   特征扰动测试:")
    for k, v in perturb_result.items():
        print(f"     {k}: 稳健性={v['robustness_score']:.2%}")

    # 测试时间序列分割
    print("\n3. 时间序列分割:")
    splitter = TimeSeriesSplitter(n_splits=3)
    splits = splitter.split(100)
    print(f"   分割数量: {len(splits)}")
    for i, (train, val, test) in enumerate(splits):
        print(f"     Split {i}: train={len(range(train.start, train.stop))}, "
              f"val={len(range(val.start, val.stop))}, "
              f"test={len(range(test.start, test.stop))}")

    print("\n✓ 所有测试通过")
