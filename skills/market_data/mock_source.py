"""
模拟市场数据源

生成随机的K线数据用于测试和回测
"""

import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List

from core.models import Bar, Quote
from skills.market_data.base import MarketDataSource


class MockMarketSource(MarketDataSource):
    """模拟市场数据源"""

    def __init__(self, seed: int = None):
        """初始化

        Args:
            seed: 随机种子，用于生成可重复的测试数据
        """
        if seed is not None:
            random.seed(seed)

        # 预设一些基础价格
        self.base_prices = {
            "AAPL": Decimal("150.0"),
            "GOOGL": Decimal("140.0"),
            "MSFT": Decimal("380.0"),
            "TSLA": Decimal("250.0"),
            "AMZN": Decimal("180.0"),
            "NVDA": Decimal("500.0"),
            "META": Decimal("350.0"),
            "00700.HK": Decimal("150.0"),  # 腾讯控股
            "09988.HK": Decimal("80.0"),   # 阿里巴巴港股
        }

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """生成模拟K线数据

        Args:
            symbol: 标的代码
            start: 开始时间
            end: 结束时间
            interval: 时间周期

        Returns:
            K线数据列表
        """
        bars = []

        # 获取基础价格
        base_price = self.base_prices.get(symbol, Decimal("100.0"))

        # 根据interval确定时间间隔
        interval_minutes = self._parse_interval(interval)

        # 生成每个时间点的数据
        current_time = start
        current_price = float(base_price)

        while current_time <= end:
            # 模拟价格波动（-2% 到 +2%）
            change_percent = random.uniform(-0.02, 0.02)

            # 开盘价（使用上一根bar的收盘价或基础价）
            open_price = Decimal(str(current_price))

            # 计算收盘价
            close_price = Decimal(str(current_price * (1 + change_percent)))

            # 生成最高价和最低价（在开盘和收盘之间）
            high_low_range = abs(open_price - close_price) * Decimal(str(random.uniform(0.5, 1.5)))
            high_price = max(open_price, close_price) + high_low_range * Decimal(str(random.uniform(0, 0.5)))
            low_price = min(open_price, close_price) - high_low_range * Decimal(str(random.uniform(0, 0.5)))

            # 生成成交量（10000 到 1000000 之间）
            volume = random.randint(10000, 1000000)

            # 创建K线
            bar = Bar(
                symbol=symbol,
                timestamp=current_time,
                open=open_price.quantize(Decimal("0.01")),
                high=high_price.quantize(Decimal("0.01")),
                low=low_price.quantize(Decimal("0.01")),
                close=close_price.quantize(Decimal("0.01")),
                volume=volume,
                interval=interval
            )
            bars.append(bar)

            # 更新当前价格
            current_price = float(close_price)

            # 移动到下一个时间点
            if interval == "1d":
                # 如果是天级别，跳过周末
                current_time += timedelta(days=1)
                while current_time.weekday() >= 5:  # 5=周六, 6=周日
                    current_time += timedelta(days=1)
            else:
                current_time += timedelta(minutes=interval_minutes)

        return bars

    def get_quote(self, symbol: str) -> Quote:
        """生成模拟实时报价

        Args:
            symbol: 标的代码

        Returns:
            最新报价
        """
        base_price = self.base_prices.get(symbol, Decimal("100.0"))

        # 在基础价格附近生成买卖价
        spread_percent = Decimal(str(random.uniform(0.0001, 0.001)))  # 0.01% - 0.1% 价差
        mid_price = base_price * Decimal(str(random.uniform(0.99, 1.01)))

        bid_price = mid_price * (1 - spread_percent / 2)
        ask_price = mid_price * (1 + spread_percent / 2)

        return Quote(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=bid_price.quantize(Decimal("0.01")),
            ask_price=ask_price.quantize(Decimal("0.01")),
            bid_size=random.randint(100, 10000),
            ask_size=random.randint(100, 10000)
        )

    def _parse_interval(self, interval: str) -> int:
        """将时间周期转换为分钟数

        Args:
            interval: 时间周期字符串

        Returns:
            分钟数
        """
        interval_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,  # 24小时，实际处理时会跳过周末
        }
        return interval_map.get(interval, 60)


class TrendingMockSource(MockMarketSource):
    """生成带趋势的模拟数据"""

    def __init__(self, seed: int = None, trend: float = 0.0001):
        """初始化

        Args:
            seed: 随机种子
            trend: 趋势系数（正数上涨，负数下跌）
        """
        super().__init__(seed)
        self.trend = trend

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """生成带趋势的K线数据"""
        # 先生成基础数据
        bars = super().get_bars(symbol, start, end, interval)

        # 应用趋势
        if not bars:
            return bars

        trended_bars = []
        cumulative_trend = 1.0

        for bar in bars:
            cumulative_trend *= (1 + self.trend)

            # 调整价格
            open_price = (bar.open * Decimal(str(cumulative_trend))).quantize(Decimal("0.01"))
            close_price = (bar.close * Decimal(str(cumulative_trend))).quantize(Decimal("0.01"))
            high_price = (bar.high * Decimal(str(cumulative_trend))).quantize(Decimal("0.01"))
            low_price = (bar.low * Decimal(str(cumulative_trend))).quantize(Decimal("0.01"))

            trended_bar = Bar(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=bar.volume,
                interval=bar.interval
            )
            trended_bars.append(trended_bar)

        return trended_bars


if __name__ == "__main__":
    """自测代码"""
    print("=== MockMarketSource 测试 ===\n")

    # 创建数据源
    source = MockMarketSource(seed=42)

    # 测试获取K线数据
    print("1. 获取 AAPL 日线数据 (2024-01-01 到 2024-01-10):")
    bars = source.get_bars(
        "AAPL",
        datetime(2024, 1, 1),
        datetime(2024, 1, 10),
        "1d"
    )

    print(f"   获取到 {len(bars)} 根K线\n")
    print("   前5根K线:")
    for bar in bars[:5]:
        print(f"   {bar.timestamp.strftime('%Y-%m-%d')}: "
              f"开={bar.open} 高={bar.high} 低={bar.low} 收={bar.close} 量={bar.volume}")

    # 测试获取报价
    print("\n2. 获取 AAPL 实时报价:")
    quote = source.get_quote("AAPL")
    print(f"   买一: {quote.bid_price} (量: {quote.bid_size})")
    print(f"   卖一: {quote.ask_price} (量: {quote.ask_size})")
    print(f"   价差: {quote.spread} ({(quote.spread/quote.mid_price*100):.4f}%)")
    print(f"   中间价: {quote.mid_price}")

    # 测试港股
    print("\n3. 获取港股 00700.HK 数据:")
    hk_bars = source.get_bars(
        "00700.HK",
        datetime(2024, 1, 1),
        datetime(2024, 1, 5),
        "1d"
    )
    print(f"   获取到 {len(hk_bars)} 根K线")
    for bar in hk_bars:
        print(f"   {bar.timestamp.strftime('%Y-%m-%d')}: 收={bar.close}")

    # 测试趋势数据
    print("\n4. 测试趋势数据 (上涨趋势):")
    trending_source = TrendingMockSource(seed=42, trend=0.01)  # 每天1%的上涨
    trend_bars = trending_source.get_bars(
        "AAPL",
        datetime(2024, 1, 1),
        datetime(2024, 1, 10),
        "1d"
    )
    print(f"   获取到 {len(trend_bars)} 根K线")
    print("   收盘价变化:")
    for bar in trend_bars:
        print(f"   {bar.timestamp.strftime('%Y-%m-%d')}: {bar.close}")

    # 测试不同时间周期
    print("\n5. 测试5分钟K线:")
    bars_5m = source.get_bars(
        "AAPL",
        datetime(2024, 1, 1, 9, 30),
        datetime(2024, 1, 1, 10, 30),
        "5m"
    )
    print(f"   获取到 {len(bars_5m)} 根5分钟K线")
    for bar in bars_5m[:3]:
        print(f"   {bar.timestamp.strftime('%Y-%m-%d %H:%M')}: 收={bar.close}")

    # 测试数据验证
    print("\n6. 测试数据验证:")
    try:
        # 尝试创建无效的K线（应该失败）
        invalid_bar = Bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("90"),  # 错误：最高价 < 最低价
            low=Decimal("95"),
            close=Decimal("92"),
            volume=1000,
            interval="1d"
        )
        print("   ✗ 数据验证失败（应该抛出异常）")
    except ValueError as e:
        print(f"   ✓ 数据验证正常: {e}")

    print("\n✓ 所有测试通过")
