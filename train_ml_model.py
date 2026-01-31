#!/usr/bin/env python3
"""
ML 模型训练脚本（支持 CUDA/GPU）

使用 RTX 3070 进行深度学习模型训练
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu():
    """检查 GPU 是否可用"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ 发现 {len(gpus)} 个 GPU:")
            for gpu in gpus:
                print(f"   - {gpu}")

            # 获取 GPU 信息
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"   设备名称: {details.get('device_name', 'Unknown')}")
                print(f"   计算能力: {details.get('compute_capability', 'Unknown')}")

            # 设置内存增长（避免占用全部显存）
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("   ✓ 已启用 GPU 内存按需增长")

            return True
        else:
            print("⚠️ 未检测到 GPU，将使用 CPU 训练")
            return False
    except ImportError:
        print("⚠️ TensorFlow 未安装，无法使用 GPU")
        print("   安装命令: pip install tensorflow>=2.15.0")
        return False


def train_random_forest(symbols, days=365, save_path=None):
    """训练随机森林模型（CPU）"""
    print("\n" + "=" * 60)
    print("训练随机森林模型")
    print("=" * 60)

    from skills.market_data.yahoo_source import YahooFinanceSource
    from skills.market_data.mock_source import MockMarketSource
    from skills.strategy.ml_strategy import MLStrategy, RandomForestModel

    # 获取训练数据
    print(f"\n📊 获取训练数据 ({len(symbols)} 个标的, {days} 天)...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        data_source = YahooFinanceSource()
        all_bars = []
        for symbol in symbols:
            bars = data_source.get_bars(symbol, start_date, end_date, "1d")
            if bars:
                all_bars.extend(bars)
                print(f"   ✓ {symbol}: {len(bars)} 条数据")

        if not all_bars:
            raise ValueError("无数据")
    except Exception as e:
        print(f"   ⚠ Yahoo Finance 失败: {e}，使用 Mock 数据")
        mock_source = MockMarketSource(seed=42)
        all_bars = []
        for symbol in symbols:
            bars = mock_source.get_bars(symbol, start_date, end_date, "1d")
            all_bars.extend(bars)
        print(f"   ✓ 使用 Mock 数据: {len(all_bars)} 条")

    # 按时间排序
    all_bars.sort(key=lambda x: x.timestamp)

    # 创建模型
    print("\n🤖 创建随机森林模型...")
    model = RandomForestModel(n_estimators=200, max_depth=15)
    strategy = MLStrategy(model=model, confidence_threshold=0.6)

    # 训练
    print("\n🚀 开始训练...")
    strategy.train(all_bars)

    # 保存模型
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        strategy.save_model(str(save_path))
        print(f"\n💾 模型已保存: {save_path}")

    print("\n✅ 训练完成!")
    return strategy


def train_lstm(symbols, days=365, epochs=50, batch_size=32, save_path=None):
    """训练 LSTM 模型（GPU 加速）"""
    print("\n" + "=" * 60)
    print("训练 LSTM 模型")
    print("=" * 60)

    # 检查 GPU
    has_gpu = check_gpu()

    if not has_gpu:
        print("\n⚠️ 未检测到 GPU，LSTM 训练将非常慢")
        response = input("   是否继续? (y/n): ")
        if response.lower() != 'y':
            return None

    import tensorflow as tf
    from skills.market_data.yahoo_source import YahooFinanceSource
    from skills.market_data.mock_source import MockMarketSource
    from skills.strategy.ml_strategy import MLStrategy, LSTMModel

    # 打印 TensorFlow 信息
    print(f"\n📋 TensorFlow 信息:")
    print(f"   版本: {tf.__version__}")
    print(f"   设备: {'GPU' if has_gpu else 'CPU'}")

    # 获取训练数据
    print(f"\n📊 获取训练数据 ({len(symbols)} 个标的, {days} 天)...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        data_source = YahooFinanceSource()
        all_bars = []
        for symbol in symbols:
            bars = data_source.get_bars(symbol, start_date, end_date, "1d")
            if bars:
                all_bars.extend(bars)
                print(f"   ✓ {symbol}: {len(bars)} 条数据")

        if not all_bars:
            raise ValueError("无数据")
    except Exception as e:
        print(f"   ⚠ Yahoo Finance 失败: {e}，使用 Mock 数据")
        mock_source = MockMarketSource(seed=42)
        all_bars = []
        for symbol in symbols:
            bars = mock_source.get_bars(symbol, start_date, end_date, "1d")
            all_bars.extend(bars)
        print(f"   ✓ 使用 Mock 数据: {len(all_bars)} 条")

    # 按时间排序
    all_bars.sort(key=lambda x: x.timestamp)

    # 创建模型
    print("\n🤖 创建 LSTM 模型...")
    model = LSTMModel(sequence_length=20, units=64)
    strategy = MLStrategy(model=model, confidence_threshold=0.6)

    # 训练
    print(f"\n🚀 开始训练 (epochs={epochs}, batch_size={batch_size})...")
    print("   按 Ctrl+C 可提前结束\n")

    try:
        strategy.train(all_bars)
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被中断")

    # 保存模型
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        strategy.save_model(str(save_path))
        print(f"\n💾 模型已保存: {save_path}")

    print("\n✅ 训练完成!")
    return strategy


def main():
    parser = argparse.ArgumentParser(description='训练 ML 交易模型')
    parser.add_argument('--model', choices=['rf', 'lstm'], default='rf',
                       help='模型类型 (rf: 随机森林, lstm: LSTM)')
    parser.add_argument('--symbols', nargs='+',
                       default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
                       help='训练用的股票代码')
    parser.add_argument('--days', type=int, default=365,
                       help='训练数据天数')
    parser.add_argument('--epochs', type=int, default=50,
                       help='LSTM 训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--save-path', type=str,
                       help='模型保存路径')

    args = parser.parse_args()

    print("=" * 60)
    print("ML 模型训练")
    print("=" * 60)
    print(f"模型类型: {args.model.upper()}")
    print(f"训练标的: {', '.join(args.symbols)}")
    print(f"数据天数: {args.days}")

    if args.model == 'lstm':
        print(f"训练轮数: {args.epochs}")
        print(f"批次大小: {args.batch_size}")

    if not args.save_path:
        args.save_path = f"models/{args.model}_model.pkl"

    print(f"保存路径: {args.save_path}")

    # 训练
    if args.model == 'rf':
        strategy = train_random_forest(
            symbols=args.symbols,
            days=args.days,
            save_path=args.save_path
        )
    else:
        strategy = train_lstm(
            symbols=args.symbols,
            days=args.days,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.save_path
        )

    if strategy:
        print("\n" + "=" * 60)
        print("🎉 训练成功完成!")
        print("=" * 60)
        print(f"\n使用模型进行回测:")
        print(f"  python -c \"from skills.strategy.ml_strategy import MLStrategy; ")
        print(f"  strategy = MLStrategy(model_path='{args.save_path}')\"")


if __name__ == "__main__":
    main()
