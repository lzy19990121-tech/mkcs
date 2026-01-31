#!/usr/bin/env python3
"""
训练 Attention-LSTM 模型

使用真实数据训练带有注意力机制的 LSTM 模型
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_attention_lstm(
    data_path='data/real_market_data.csv',
    model_path='models/attention_lstm.h5',
    epochs=100,
    sequence_length=30
):
    """训练 Attention-LSTM 模型

    Args:
        data_path: 数据文件路径
        model_path: 模型保存路径
        epochs: 训练轮数
        sequence_length: 序列长度
    """
    from skills.models.attention_lstm import AttentionLSTMModel
    from skills.indicators.technical import IndicatorFeatures
    from skills.strategy.ml_strategy import FeatureEngineer

    print("=" * 60)
    print("训练 Attention-LSTM 模型")
    print("=" * 60)

    # 加载数据
    print(f"\n📊 加载数据: {data_path}")
    bars = load_real_data(data_path)
    print(f"   ✓ 加载 {len(bars)} 条 K 线")

    # 提取特征（使用新的技术指标库）
    print("\n🔧 提取技术指标特征...")
    indicator_features = IndicatorFeatures.extract_all(bars)

    # 转换为特征矩阵
    feature_list = []
    for i in range(len(bars)):
        row = []
        for name, values in indicator_features.items():
            if i < len(values):
                val = values[i]
                if float('nan') == val:
                    # 使用前一个有效值
                    for j in range(i-1, -1, -1):
                        if j < len(values) and not float('nan') == values[j]:
                            val = values[j]
                            break
                    else:
                        val = 0.0
                else:
                    val = float(val)
                row.append(val)
            else:
                row.append(0.0)
        feature_list.append(row)

    features = np.array(feature_list)
    print(f"   ✓ 特征形状: {features.shape}")

    # 生成标签
    print("\n🏷️  生成标签...")
    labels, horizon = generate_labels(bars, sequence_length, prediction_horizon=5)
    print(f"   ✓ 标签数量: {len(labels)}")
    print(f"   ✓ 类别分布: 跌={sum(labels==0)}, 平={sum(labels==1)}, 涨={sum(labels==2)}")

    # 对齐数据
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels = labels[:min_len]

    # 重塑为 LSTM 格式
    print(f"\n📐 重塑数据为 LSTM 格式...")
    features_per_step = features.shape[1] // sequence_length
    X = features[:, :sequence_length * features_per_step].reshape(
        -1, sequence_length, features_per_step
    )
    y = labels[:len(X)]

    print(f"   ✓ X.shape: {X.shape}")
    print(f"   ✓ y.shape: {y.shape}")

    # 创建模型
    print(f"\n🤖 创建 Attention-LSTM 模型...")
    model = AttentionLSTMModel(
        sequence_length=sequence_length,
        lstm_units=128,
        attention_units=64,
        dropout_rate=0.3,
        bidirectional=True
    )

    # 训练
    print(f"\n🚀 开始训练 ({epochs} epochs)...\n")
    history = model.train(X, y, epochs=epochs, batch_size=32)

    # 保存模型
    print(f"\n💾 保存模型: {model_path}")
    model.save(model_path)

    # 打印最终指标
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    print(f"\n📊 最终指标:")
    print(f"   训练损失: {final_loss:.4f}")
    print(f"   训练准确率: {final_acc:.4f}")
    print(f"   验证损失: {final_val_loss:.4f}")
    print(f"   验证准确率: {final_val_acc:.4f}")

    print("\n✅ 训练完成!")
    return model


def load_real_data(path):
    """加载真实数据"""
    import csv
    from decimal import Decimal
    from core.models import Bar

    bars = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bar = Bar(
                symbol=row['symbol'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                open=Decimal(row['open']),
                high=Decimal(row['high']),
                low=Decimal(row['low']),
                close=Decimal(row['close']),
                volume=int(row['volume']),
                interval="1d"
            )
            bars.append(bar)

    bars.sort(key=lambda x: x.timestamp)
    return bars


def generate_labels(bars, min_bars, prediction_horizon=5):
    """生成训练标签"""
    labels = []

    for i in range(min_bars, len(bars) - prediction_horizon):
        current_price = float(bars[i].close)
        future_price = float(bars[i + prediction_horizon].close)
        change = (future_price - current_price) / current_price

        if change > 0.01:  # 涨超过1%
            labels.append(2)
        elif change < -0.01:  # 跌超过1%
            labels.append(0)
        else:
            labels.append(1)

    return np.array(labels), prediction_horizon


def main():
    import argparse

    parser = argparse.ArgumentParser(description='训练 Attention-LSTM 模型')
    parser.add_argument('--data-path', default='data/real_market_data.csv')
    parser.add_argument('--model-path', default='models/attention_lstm.h5')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--sequence-length', type=int, default=30)

    args = parser.parse_args()

    train_attention_lstm(
        data_path=args.data_path,
        model_path=args.model_path,
        epochs=args.epochs,
        sequence_length=args.sequence_length
    )


if __name__ == "__main__":
    main()
