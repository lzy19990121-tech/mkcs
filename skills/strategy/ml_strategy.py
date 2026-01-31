"""
机器学习策略

使用 LSTM、随机森林等 ML 模型进行价格预测和交易信号生成
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
import pickle
import json
from pathlib import Path

import numpy as np

from core.models import Bar, Signal, Position, TradeSide
from skills.strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """特征工程配置"""
    use_price_features: bool = True      # 价格相关特征
    use_volume_features: bool = True     # 成交量特征
    use_technical_features: bool = True  # 技术指标特征
    lookback_window: int = 20            # 回看窗口大小
    prediction_horizon: int = 5          # 预测未来N个周期


class FeatureEngineer:
    """特征工程"""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

    def extract_features(self, bars: List[Bar]) -> np.ndarray:
        """从K线数据中提取特征

        Args:
            bars: K线数据列表

        Returns:
            特征矩阵 (samples, features)
        """
        if len(bars) < self.config.lookback_window:
            raise ValueError(f"数据不足，需要至少 {self.config.lookback_window} 根K线")

        features = []
        window = self.config.lookback_window

        for i in range(window, len(bars)):
            window_bars = bars[i-window:i]
            sample_features = []

            if self.config.use_price_features:
                sample_features.extend(self._price_features(window_bars))

            if self.config.use_volume_features:
                sample_features.extend(self._volume_features(window_bars))

            if self.config.use_technical_features:
                sample_features.extend(self._technical_features(window_bars))

            features.append(sample_features)

        return np.array(features)

    def _price_features(self, bars: List[Bar]) -> List[float]:
        """价格特征"""
        closes = [float(b.close) for b in bars]
        opens = [float(b.open) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]

        features = []

        # 当前价格
        features.append(closes[-1])

        # 价格变化率
        returns = np.diff(closes) / closes[:-1]
        features.extend([
            np.mean(returns),
            np.std(returns),
            np.max(returns),
            np.min(returns),
        ])

        # 价格位置（在高低点之间的位置）
        price_position = (closes[-1] - min(lows)) / (max(highs) - min(lows) + 1e-10)
        features.append(price_position)

        # 均线
        for period in [5, 10, 20]:
            if len(closes) >= period:
                ma = np.mean(closes[-period:])
                features.append(ma)
                features.append((closes[-1] - ma) / ma)  # 偏离度
            else:
                features.extend([closes[-1], 0])

        return features

    def _volume_features(self, bars: List[Bar]) -> List[float]:
        """成交量特征"""
        volumes = [b.volume for b in bars]
        closes = [float(b.close) for b in bars]

        features = []

        # 当前成交量
        features.append(volumes[-1])

        # 成交量变化
        features.append(np.mean(volumes))
        features.append(np.std(volumes))

        # 量价关系
        if len(volumes) > 1:
            volume_change = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1)
            price_change = (closes[-1] - closes[-2]) / (closes[-2] + 1e-10)
            features.append(volume_change)
            features.append(price_change * volume_change)  # 量价配合
        else:
            features.extend([0, 0])

        return features

    def _technical_features(self, bars: List[Bar]) -> List[float]:
        """技术指标特征"""
        closes = [float(b.close) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]

        features = []

        # RSI (14期)
        rsi = self._calculate_rsi(closes, 14)
        features.append(rsi)

        # MACD
        macd, signal, hist = self._calculate_macd(closes)
        features.extend([macd, signal, hist])

        # Bollinger Bands
        bb_upper, bb_lower, bb_position = self._calculate_bollinger_bands(closes)
        features.extend([bb_upper, bb_lower, bb_position])

        # ATR (Average True Range)
        atr = self._calculate_atr(highs, lows, closes)
        features.append(atr)

        return features

    @staticmethod
    def _calculate_rsi(prices: List[float], period: int = 14) -> float:
        """计算 RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """计算 MACD"""
        if len(prices) < slow + signal:
            return 0, 0, 0

        def ema(data, span):
            return np.mean(data)  # 简化版，实际应使用指数移动平均

        ema_fast = ema(prices[-fast:], fast)
        ema_slow = ema(prices[-slow:], slow)
        macd = ema_fast - ema_slow
        signal_line = ema([macd] * signal, signal)  # 简化
        histogram = macd - signal_line

        return macd, signal_line, histogram

    @staticmethod
    def _calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """计算布林带"""
        if len(prices) < period:
            return prices[-1], prices[-1], 0.5

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper = sma + std_dev * std
        lower = sma - std_dev * std
        position = (prices[-1] - lower) / (upper - lower + 1e-10)

        return upper, lower, position

    @staticmethod
    def _calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """计算 ATR"""
        if len(highs) < period + 1:
            return 0

        tr_list = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr_list.append(max(tr1, tr2, tr3))

        return np.mean(tr_list[-period:])


class MLModel(ABC):
    """机器学习模型基类"""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """加载模型"""
        pass


class RandomForestModel(MLModel):
    """随机森林模型"""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self._try_import()

    def _try_import(self):
        """尝试导入 sklearn"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            self._rf_class = RandomForestClassifier
        except ImportError:
            logger.warning("sklearn 未安装，使用备用实现")
            self._rf_class = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练随机森林"""
        if self._rf_class:
            self.model = self._rf_class(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
            self.model.fit(X, y)
        else:
            # 简化版：使用多数类
            from collections import Counter
            self.model = Counter(y)
            logger.warning("使用简化模型（sklearn 未安装）")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if self._rf_class and self.model:
            return self.model.predict(X)
        else:
            # 返回多数类
            if self.model:
                return np.array([self.model.most_common(1)[0][0]] * len(X))
            return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self._rf_class and self.model:
            return self.model.predict_proba(X)
        else:
            # 均匀分布
            return np.array([[0.33, 0.33, 0.34]] * len(X))

    def save(self, path: str) -> None:
        """保存模型"""
        if self.model:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"模型已保存到 {path}")

    def load(self, path: str) -> None:
        """加载模型"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"模型已从 {path} 加载")


class LSTMModel(MLModel):
    """LSTM 深度学习模型（简化版占位）"""

    def __init__(self, sequence_length: int = 20, units: int = 50):
        self.sequence_length = sequence_length
        self.units = units
        self.model = None
        self._try_import()

    def _try_import(self):
        """尝试导入 tensorflow"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            self._has_tf = True
            self._Sequential = Sequential
            self._LSTM = LSTM
            self._Dense = Dense
            self._Dropout = Dropout
        except ImportError:
            logger.warning("tensorflow 未安装，LSTM 模型不可用")
            self._has_tf = False

    def _build_model(self, input_shape: Tuple[int, int]):
        """构建 LSTM 模型"""
        if not self._has_tf:
            return

        # 禁用 cuDNN 优化以兼容 CUDA 12.x
        # 或者使用 CPU 训练
        import tensorflow as tf

        model = self._Sequential([
            self._LSTM(
                self.units,
                return_sequences=True,
                input_shape=input_shape,
                # 禁用 cuDNN 优化（不使用默认激活函数）
                activation='sigmoid',  # 非 tanh 以禁用 cuDNN
                recurrent_activation='hard_sigmoid',
                use_bias=True,
                unroll=False,  # 不展开循环
            ),
            self._Dropout(0.2),
            self._LSTM(
                self.units // 2,
                return_sequences=False,
                activation='sigmoid',
                recurrent_activation='hard_sigmoid'
            ),
            self._Dropout(0.2),
            self._Dense(32, activation='relu'),
            self._Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练 LSTM"""
        if not self._has_tf:
            logger.error("tensorflow 未安装，无法训练 LSTM")
            return

        # 重塑输入为 (samples, timesteps, features)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        timesteps = self.sequence_length

        # 简化：假设特征可以分成时间步
        features_per_step = n_features // timesteps
        X_reshaped = X[:, :timesteps * features_per_step].reshape(
            n_samples, timesteps, features_per_step
        )

        self._build_model((timesteps, features_per_step))

        # One-hot 编码
        from tensorflow.keras.utils import to_categorical
        y_cat = to_categorical(y, num_classes=3)

        self.model.fit(
            X_reshaped, y_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            return np.zeros(len(X))

        n_samples = X.shape[0]
        timesteps = self.sequence_length
        features_per_step = X.shape[1] // timesteps

        X_reshaped = X[:, :timesteps * features_per_step].reshape(
            n_samples, timesteps, features_per_step
        )

        predictions = self.model.predict(X_reshaped)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            return np.array([[0.33, 0.33, 0.34]] * len(X))

        n_samples = X.shape[0]
        timesteps = self.sequence_length
        features_per_step = X.shape[1] // timesteps

        X_reshaped = X[:, :timesteps * features_per_step].reshape(
            n_samples, timesteps, features_per_step
        )

        return self.model.predict(X_reshaped)

    def save(self, path: str) -> None:
        """保存模型"""
        if self.model:
            self.model.save(path)
            logger.info(f"LSTM 模型已保存到 {path}")

    def load(self, path: str) -> None:
        """加载模型"""
        if not self._has_tf:
            return

        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        logger.info(f"LSTM 模型已从 {path} 加载")


class MLStrategy(Strategy):
    """机器学习策略

    使用 ML 模型预测价格走势并生成交易信号
    """

    def __init__(
        self,
        model: Optional[MLModel] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
        train_mode: bool = False
    ):
        """初始化 ML 策略

        Args:
            model: ML 模型实例
            feature_engineer: 特征工程器
            model_path: 预训练模型路径
            confidence_threshold: 置信度阈值
            train_mode: 是否处于训练模式
        """
        self.model = model or RandomForestModel()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.confidence_threshold = confidence_threshold
        self.train_mode = train_mode

        # 加载预训练模型
        if model_path and Path(model_path).exists():
            self.model.load(model_path)

        # 训练数据缓存
        self._training_data: List[Tuple[np.ndarray, int]] = []

    def generate_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None
    ) -> List[Signal]:
        """生成交易信号

        Args:
            bars: K线数据列表
            position: 当前持仓

        Returns:
            交易信号列表
        """
        if len(bars) < self.feature_engineer.config.lookback_window + 5:
            return []

        signals = []

        try:
            # 提取特征
            features = self.feature_engineer.extract_features(bars)

            if len(features) == 0:
                return []

            # 使用最新的特征进行预测
            latest_features = features[-1:]

            if self.train_mode:
                # 训练模式：收集数据
                label = self._generate_label(bars)
                if label is not None:
                    self._training_data.append((latest_features[0], label))
                return []

            # 预测
            prediction = self.model.predict(latest_features)[0]
            probabilities = self.model.predict_proba(latest_features)[0]

            confidence = probabilities[prediction]

            if confidence < self.confidence_threshold:
                return []

            # 根据预测生成信号
            current_bar = bars[-1]
            current_price = current_bar.close

            # 0: 跌(卖出), 1: 平(持有), 2: 涨(买入)
            if prediction == 2:  # 上涨预测
                signal = Signal(
                    symbol=current_bar.symbol,
                    timestamp=current_bar.timestamp,
                    action="BUY",
                    price=current_price,
                    quantity=100,  # 默认数量
                    confidence=confidence,
                    reason=f"ML模型预测上涨 (置信度: {confidence:.2%})"
                )
                signals.append(signal)

            elif prediction == 0:  # 下跌预测
                # 只有有持仓时才卖出
                if position and position.quantity > 0:
                    signal = Signal(
                        symbol=current_bar.symbol,
                        timestamp=current_bar.timestamp,
                        action="SELL",
                        price=current_price,
                        quantity=min(position.quantity, 100),
                        confidence=confidence,
                        reason=f"ML模型预测下跌 (置信度: {confidence:.2%})"
                    )
                    signals.append(signal)

        except Exception as e:
            logger.error(f"ML策略信号生成失败: {e}")

        return signals

    def _generate_label(self, bars: List[Bar]) -> Optional[int]:
        """生成训练标签

        Returns:
            0: 跌, 1: 平, 2: 涨
        """
        horizon = self.feature_engineer.config.prediction_horizon

        if len(bars) < horizon + 1:
            return None

        current_price = float(bars[-horizon-1].close)
        future_price = float(bars[-1].close)

        change = (future_price - current_price) / current_price

        if change > 0.01:  # 涨超过1%
            return 2
        elif change < -0.01:  # 跌超过1%
            return 0
        else:
            return 1

    def get_min_bars_required(self) -> int:
        """获取策略所需的最小K线数量"""
        return self.feature_engineer.config.lookback_window + self.feature_engineer.config.prediction_horizon

    def train(self, bars: List[Bar]) -> None:
        """训练模型

        Args:
            bars: 训练数据
        """
        logger.info("开始训练 ML 模型...")

        # 提取特征和标签
        features = self.feature_engineer.extract_features(bars)

        labels = []
        window = self.feature_engineer.config.lookback_window
        horizon = self.feature_engineer.config.prediction_horizon

        for i in range(window, len(bars) - horizon):
            current_price = float(bars[i].close)
            future_price = float(bars[i + horizon].close)
            change = (future_price - current_price) / current_price

            if change > 0.01:
                labels.append(2)
            elif change < -0.01:
                labels.append(0)
            else:
                labels.append(1)

        # 对齐特征和标签
        features = features[:len(labels)]

        if len(features) < 100:
            logger.warning(f"训练数据不足: {len(features)} 样本")
            return

        X = np.array(features)
        y = np.array(labels)

        logger.info(f"训练数据: {len(X)} 样本")
        logger.info(f"类别分布: 跌={sum(y==0)}, 平={sum(y==1)}, 涨={sum(y==2)}")

        # 训练模型
        self.model.train(X, y)

        logger.info("模型训练完成")

    def save_model(self, path: str) -> None:
        """保存模型"""
        self.model.save(path)

    def load_model(self, path: str) -> None:
        """加载模型"""
        self.model.load(path)

    def get_training_data_count(self) -> int:
        """获取训练数据数量"""
        return len(self._training_data)


def train_ml_strategy(
    symbol: str = "AAPL",
    start_date: datetime = None,
    end_date: datetime = None,
    model_type: str = "rf",
    save_path: str = None
) -> MLStrategy:
    """训练 ML 策略

    Args:
        symbol: 训练用的标的
        start_date: 开始日期
        end_date: 结束日期
        model_type: 模型类型 (rf: 随机森林, lstm: LSTM)
        save_path: 模型保存路径

    Returns:
        训练好的 MLStrategy
    """
    from datetime import timedelta
    from skills.market_data.yahoo_source import YahooFinanceSource

    if start_date is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

    # 获取训练数据
    data_source = YahooFinanceSource()
    bars = data_source.get_bars(symbol, start_date, end_date, "1d")

    logger.info(f"获取到 {len(bars)} 根训练数据")

    # 创建模型
    if model_type == "lstm":
        model = LSTMModel()
    else:
        model = RandomForestModel(n_estimators=100)

    # 创建策略
    strategy = MLStrategy(model=model, train_mode=True)

    # 训练
    strategy.train(bars)

    # 保存模型
    if save_path:
        strategy.save_model(save_path)

    return strategy


if __name__ == "__main__":
    """自测代码"""
    print("=== ML 策略测试 ===\n")

    # 1. 测试特征工程
    print("1. 测试特征工程:")
    from skills.market_data.mock_source import MockMarketSource

    mock_source = MockMarketSource(seed=42)
    bars = mock_source.get_bars(
        "AAPL",
        datetime(2024, 1, 1),
        datetime(2024, 3, 1),
        "1d"
    )
    print(f"   生成 {len(bars)} 根K线")

    engineer = FeatureEngineer()
    features = engineer.extract_features(bars)
    print(f"   提取特征: {features.shape}")
    print(f"   特征示例: {features[0][:5]}")

    # 2. 测试模型训练
    print("\n2. 测试随机森林模型:")
    model = RandomForestModel(n_estimators=10)

    # 生成模拟标签
    X = np.random.randn(100, features.shape[1])
    y = np.random.randint(0, 3, 100)

    model.train(X, y)
    predictions = model.predict(X[:5])
    print(f"   预测结果: {predictions}")

    # 3. 测试 ML 策略
    print("\n3. 测试 ML 策略:")
    strategy = MLStrategy(model=model, confidence_threshold=0.5)

    # 生成信号
    signals = strategy.generate_signals(bars, position=None)
    print(f"   生成 {len(signals)} 个信号")
    for signal in signals[:3]:
        print(f"   {signal.timestamp}: {signal.action} @ ${signal.price} ({signal.reason})")

    print("\n✓ 测试完成")
