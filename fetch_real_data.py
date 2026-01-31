#!/usr/bin/env python3
"""
获取真实市场数据并训练模型

从 Yahoo Finance 获取 5 ��历史数据，重新训练 LSTM 模型
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_real_data(symbols, years=5):
    """从 Yahoo Finance 获取真实数据

    Args:
        symbols: 股票代码列表
        years: 获取年数

    Returns:
        所有标的的 K 线数据列表
    """
    import yfinance as yf
    from skills.market_data.yahoo_source import YahooFinanceSource
    from core.models import Bar
    from decimal import Decimal

    print("=" * 60)
    print(f"获取真实市场数据 ({years} 年)")
    print("=" * 60)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    print(f"\n📅 日期范围: {start_date.date()} ~ {end_date.date()}")
    print(f"📈 标的: {', '.join(symbols)}\n")

    all_bars = []
    source = YahooFinanceSource()

    for symbol in symbols:
        try:
            print(f"正在获取 {symbol}...")

            # 使用 yfinance 直接获取（更可靠）
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")

            if df.empty:
                print(f"   ⚠️ {symbol} 无数据")
                continue

            # 转换为 Bar 对象
            for timestamp, row in df.iterrows():
                bar = Bar(
                    symbol=symbol,
                    timestamp=timestamp.to_pydatetime(),
                    open=Decimal(str(round(row['Open'], 4))),
                    high=Decimal(str(round(row['High'], 4))),
                    low=Decimal(str(round(row['Low'], 4))),
                    close=Decimal(str(round(row['Close'], 4))),
                    volume=int(row['Volume']),
                    interval="1d"
                )
                all_bars.append(bar)

            print(f"   ✓ {symbol}: {len(df)} 条数据")

        except Exception as e:
            print(f"   ✗ {symbol} 失败: {e}")

    # 按时间排序
    all_bars.sort(key=lambda x: x.timestamp)

    print(f"\n✅ 总共获取 {len(all_bars)} 条 K 线数据")
    return all_bars


def save_real_data(bars, output_path):
    """保存真实数据到 CSV

    Args:
        bars: K 线数据列表
        output_path: 输出文件路径
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

        for bar in bars:
            writer.writerow([
                bar.timestamp.isoformat(),
                bar.symbol,
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                bar.volume
            ])

    print(f"📄 数据已保存: {output_path}")


def load_real_data(input_path):
    """从 CSV 加载真实数据

    Args:
        input_path: 输入文件路径

    Returns:
        K 线数据列表
    """
    import csv
    from decimal import Decimal
    from core.models import Bar

    bars = []
    with open(input_path, 'r') as f:
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


def train_with_real_data(bars, model_type='lstm', epochs=100, save_path=None):
    """使用真实数据训练模型

    Args:
        bars: K 线数据列表
        model_type: 模型类型 (lstm, rf)
        epochs: 训练轮数
        save_path: 模型保存路径
    """
    from skills.strategy.ml_strategy import MLStrategy, LSTMModel

    print("\n" + "=" * 60)
    print(f"训练 {model_type.upper()} 模型 (真实数据)")
    print("=" * 60)

    # 创建模型
    if model_type == 'lstm':
        model = LSTMModel(sequence_length=30, units=128)
    else:
        from skills.strategy.ml_strategy import RandomForestModel
        model = RandomForestModel(n_estimators=200, max_depth=15)

    strategy = MLStrategy(model=model, confidence_threshold=0.6)

    # 训练
    print(f"\n🚀 开始训练 ({epochs} epochs)...\n")
    strategy.train(bars)

    # 保存模型
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        strategy.save_model(str(save_path))
        print(f"\n💾 模型已保存: {save_path}")

    print("\n✅ 训练完成!")
    return strategy


def main():
    import argparse

    parser = argparse.ArgumentParser(description='获取真实数据并训练模型')
    parser.add_argument('--symbols', nargs='+',
                       default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'],
                       help='股票代码')
    parser.add_argument('--years', type=int, default=5,
                       help='获取年数')
    parser.add_argument('--model', choices=['lstm', 'rf'], default='lstm',
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--data-path', default='data/real_market_data.csv',
                       help='数据保存路径')
    parser.add_argument('--model-path', default='models/lstm_real_data.h5',
                       help='模型保存路径')
    parser.add_argument('--use-cached', action='store_true',
                       help='使用已缓存的数据')

    args = parser.parse_args()

    # 获取或加载数据
    if args.use_cached and Path(args.data_path).exists():
        print(f"使用缓存数据: {args.data_path}")
        bars = load_real_data(args.data_path)
    else:
        bars = fetch_real_data(args.symbols, args.years)
        if bars:
            save_real_data(bars, args.data_path)

    if not bars:
        print("❌ 无可用数据")
        return

    # 训练模型
    train_with_real_data(
        bars=bars,
        model_type=args.model,
        epochs=args.epochs,
        save_path=args.model_path
    )

    print("\n" + "=" * 60)
    print("🎉 完成!")
    print("=" * 60)
    print(f"\n使用模型进行回测:")
    print(f"  python -c \"from skills.strategy.ml_strategy import MLStrategy; ")
    print(f"  strategy = MLStrategy(model_path='{args.model_path}')\"")


if __name__ == "__main__":
    main()
