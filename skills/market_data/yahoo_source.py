"""
Yahoo Finance 数据源适配器

从 Yahoo Finance 获取真实市场数据
支持美股、港股等全球主要市场

依赖: yfinance>=0.2.28
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict
import logging

from core.models import Bar, Quote
from skills.market_data.base import MarketDataSource

# 配置日志
logger = logging.getLogger(__name__)


class YahooFinanceSource(MarketDataSource):
    """Yahoo Finance 数据源

    从 Yahoo Finance 获取真实股票数据
    支持自动重试、数据缓存、错误处理
    """

    # 市场时区映射
    MARKET_TIMEZONES = {
        "US": "America/New_York",
        "HK": "Asia/Hong_Kong",
        "CN": "Asia/Shanghai",
        "JP": "Asia/Tokyo",
        "UK": "Europe/London",
    }

    # 后缀映射（Yahoo Finance 对不同市场使用不同后缀）
    SYMBOL_SUFFIX = {
        "HK": ".HK",      # 港股: 0700.HK
        "SS": ".SS",      # 上证: 600000.SS
        "SZ": ".SZ",      # 深证: 000001.SZ
        "JP": ".T",       # 日股: 7203.T
        "L": ".L",        # 英股: VOD.L
        "DE": ".DE",      # 德股: BMW.DE
    }

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """初始化 Yahoo Finance 数据源

        Args:
            cache_dir: 缓存目录路径
            enable_cache: 是否启用本地缓存
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
        """
        self.enable_cache = enable_cache
        self.max_retries = max_retries
        self.timeout = timeout
        self._yf = None  # 延迟导入
        self._cache: Dict[str, List[Bar]] = {}
        self._cache_metadata: Dict[str, dict] = {}

    def _get_yf(self):
        """延迟导入 yfinance"""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
                logger.info("Yahoo Finance API 初始化成功")
            except ImportError:
                raise ImportError(
                    "yfinance 库未安装。请运行: pip install yfinance>=0.2.28"
                )
        return self._yf

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取K线数据

        Args:
            symbol: 标的代码 (如 AAPL, 0700.HK)
            start: 开始时间
            end: 结束时间
            interval: 时间周期（1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo）

        Returns:
            K线数据列表
        """
        cache_key = f"{symbol}_{interval}_{start.date()}_{end.date()}"

        # 检查缓存
        if self.enable_cache and cache_key in self._cache:
            cached_bars = self._cache[cache_key]
            logger.debug(f"使用缓存数据: {symbol} ({len(cached_bars)} bars)")
            return [b for b in cached_bars if start <= b.timestamp <= end]

        # 获取数据
        bars = self._fetch_bars(symbol, start, end, interval)

        # 存入缓存
        if self.enable_cache:
            self._cache[cache_key] = bars.copy()

        return bars

    def get_bars_until(
        self,
        symbol: str,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取截至某个时间点的K线数据

        Args:
            symbol: 标的代码
            end: 截止时间
            interval: 时间周期

        Returns:
            K线数据列表，按时间升序排列
        """
        # Yahoo Finance 限制：1m 数据最多 7 天，15m 最多 60 天
        # 根据 interval 确定合适的起始时间
        start = self._calculate_start_time(end, interval)

        bars = self.get_bars(symbol, start, end, interval)

        # 返回按时间排序的数据
        return sorted(bars, key=lambda x: x.timestamp)

    def get_quote(self, symbol: str) -> Quote:
        """获取实时报价

        Args:
            symbol: 标的代码

        Returns:
            最新报价
        """
        yf = self._get_yf()
        ticker = yf.Ticker(symbol)

        for attempt in range(self.max_retries):
            try:
                info = ticker.info
                fast_info = ticker.fast_info

                # 获取实时报价
                bid = fast_info.get("last_price", 0)
                ask = fast_info.get("last_price", 0)
                bid_size = 0
                ask_size = 0

                # 尝试获取买卖盘
                if "bid" in info and info["bid"]:
                    bid = Decimal(str(info["bid"]))
                if "ask" in info and info["ask"]:
                    ask = Decimal(str(info["ask"]))
                if "bidSize" in info:
                    bid_size = int(info["bidSize"])
                if "askSize" in info:
                    ask_size = int(info["askSize"])

                # 使用最新价格作为备选
                if bid == 0 and "regularMarketImpactPrice" in info:
                    bid = Decimal(str(info["regularMarketPrice"]))
                    ask = bid

                # 获取themes昨日收盘价（用于计算涨跌幅）
                prev_close = None
                if "previousClose" in info and info["previousClose"]:
                    prev_close = Decimal(str(info["previousClose"]))

                return Quote(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bid_price=bid if isinstance(bid, Decimal) else Decimal(str(bid)),
                    ask_price=ask if isinstance(ask, Decimal) else Decimal(str(ask)),
                    bid_size=bid_size,
                    ask_size=ask_size,
                    prev_close=prev_close
                )

            except Exception as e:
                logger.warning(f"获取报价失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise

        raise RuntimeError(f"无法获取 {symbol} 的报价")

    def _fetch_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """从 Yahoo Finance 获取数据

        Args:
            symbol: 标的代码
            start: 开始时间
            end: 结束时间
            interval: 时间周期

        Returns:
            K线数据列表
        """
        yf = self._get_yf()
        ticker = yf.Ticker(symbol)

        # 转换时间周期
        yf_interval = self._convert_interval(interval)

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"获取 {symbol} 数据: {start} ~ {end}, 周期={interval}")

                # 下载数据
                df = ticker.history(
                    start=start,
                    end=end + timedelta(days=1),  # yfinance 的 end 是排他的
                    interval=yf_interval,
                    timeout=self.timeout
                )

                if df.empty:
                    logger.warning(f"未获取到数据: {symbol} ({start} ~ {end})")
                    return []

                # 转换为 Bar 对象
                bars = []
                for timestamp, row in df.iterrows():
                    bar = Bar(
                        symbol=symbol,
                        timestamp=timestamp.to_pydatetime(),
                        open=Decimal(str(round(row["Open"], 4))),
                        high=Decimal(str(round(row["High"], 4))),
                        low=Decimal(str(round(row["Low"], 4))),
                        close=Decimal(str(round(row["Close"], 4))),
                        volume=int(row["Volume"]),
                        interval=interval
                    )
                    bars.append(bar)

                logger.info(f"成功获取 {symbol}: {len(bars)} 根K线")
                return bars

            except Exception as e:
                logger.warning(f"获取数据失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"无法获取 {symbol} 的数据: {e}")
                    raise

        return []

    def _calculate_start_time(self, end: datetime, interval: str) -> datetime:
        """根据 interval 计算合适的起始时间

        Yahoo Finance 对不同时间周期有数据长度限制
        """
        now = datetime.now()

        interval_limits = {
            "1m": timedelta(days=7),      # 1分钟数据最多7天
            "2m": timedelta(days=30),
            "5m": timedelta(days=60),
            "15m": timedelta(days=60),
            "30m": timedelta(days=60),
            "60m": timedelta(days=730),   # 1小时最多2年
            "1h": timedelta(days=730),
            "1d": timedelta(days=3650),   # 日线最多10年
            "5d": timedelta(days=3650),
            "1wk": timedelta(days=3650),
            "1mo": timedelta(days=36500), # 月线最多100年
        }

        max_lookback = interval_limits.get(interval, timedelta(days=365))

        # 取当前时间和 end 的较小值作为基准
        base_time = min(end, now)
        start = base_time - max_lookback

        return start

    def _convert_interval(self, interval: str) -> str:
        """将内部时间周期转换为 Yahoo Finance 格式"""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "1h",  # Yahoo 不支持 4h，用 1h 代替
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo",
        }
        return mapping.get(interval, "1d")

    def get_ticker_info(self, symbol: str) -> dict:
        """获取标的详细信息

        Args:
            symbol: 标的代码

        Returns:
            包含公司信息、市值等的字典
        """
        yf = self._get_yf()
        ticker = yf.Ticker(symbol)

        try:
            info = ticker.info
            return {
                "symbol": symbol,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
            }
        except Exception as e:
            logger.error(f"获取标的信息失败: {e}")
            return {"symbol": symbol, "error": str(e)}

    def clear_cache(self):
        """清空本地缓存"""
        self._cache.clear()
        self._cache_metadata.clear()
        logger.info("缓存已清空")


class HybridDataSource(MarketDataSource):
    """混合数据源

    优先使用真实数据，失败时回退到 Mock 数据
    适用于开发和测试环境
    """

    def __init__(
        self,
        real_source: Optional[MarketDataSource] = None,
        mock_source: Optional[MarketDataSource] = None,
        fallback_on_error: bool = True
    ):
        """初始化混合数据源

        Args:
            real_source: 真实数据源（默认 YahooFinanceSource）
            mock_source: 模拟数据源（默认 MockMarketSource）
            fallback_on_error: 出错时是否回退到模拟数据
        """
        self.real_source = real_source or YahooFinanceSource()
        self.mock_source = mock_source
        self.fallback_on_error = fallback_on_error
        self._use_real = True

        if self.mock_source is None:
            from skills.market_data.mock_source import MockMarketSource
            self.mock_source = MockMarketSource(seed=42)

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取K线数据，优先使用真实数据源"""
        if self._use_real:
            try:
                return self.real_source.get_bars(symbol, start, end, interval)
            except Exception as e:
                logger.warning(f"真实数据源失败: {e}")
                if not self.fallback_on_error:
                    raise

        logger.info(f"使用模拟数据源: {symbol}")
        return self.mock_source.get_bars(symbol, start, end, interval)

    def get_bars_until(
        self,
        symbol: str,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取截至某个时间点的K线数据"""
        if self._use_real:
            try:
                return self.real_source.get_bars_until(symbol, end, interval)
            except Exception as e:
                logger.warning(f"真实数据源失败: {e}")
                if not self.fallback_on_error:
                    raise

        return self.mock_source.get_bars_until(symbol, end, interval)

    def get_quote(self, symbol: str) -> Quote:
        """获取实时报价"""
        if self._use_real:
            try:
                return self.real_source.get_quote(symbol)
            except Exception as e:
                logger.warning(f"真实数据源失败: {e}")
                if not self.fallback_on_error:
                    raise

        return self.mock_source.get_quote(symbol)

    def set_mode(self, use_real: bool):
        """切换数据源模式

        Args:
            use_real: True 使用真实数据，False 使用模拟数据
        """
        self._use_real = use_real
        logger.info(f"数据源模式切换为: {'真实数据' if use_real else '模拟数据'}")


if __name__ == "__main__":
    """自测代码"""
    print("=== Yahoo Finance 数据源测试 ===\n")

    # 创建数据源
    source = YahooFinanceSource(enable_cache=True)

    # 测试美股数据
    print("1. 获取 AAPL 日线数据 (最近5天):")
    try:
        end = datetime.now()
        start = end - timedelta(days=5)
        bars = source.get_bars("AAPL", start, end, "1d")

        print(f"   获取到 {len(bars)} 根K线")
        for bar in bars[:3]:
            print(f"   {bar.timestamp.strftime('%Y-%m-%d')}: "
                  f"开={bar.open} 高={bar.high} 低={bar.low} 收={bar.close} 量={bar.volume}")
    except Exception as e:
        print(f"   ✗ 获取失败: {e}")

    # 测试港股数据
    print("\n2. 获取港股 0700.HK (腾讯) 数据:")
    try:
        bars = source.get_bars("0700.HK", start, end, "1d")
        print(f"   获取到 {len(bars)} 根K线")
        for bar in bars[:3]:
            print(f"   {bar.timestamp.strftime('%Y-%m-%d')}: 收={bar.close}")
    except Exception as e:
        print(f"   ✗ 获取失败: {e}")

    # 测试实时报价
    print("\n3. 获取 AAPL 实时报价:")
    try:
        quote = source.get_quote("AAPL")
        print(f"   买一: {quote.bid_price}")
        print(f"   卖一: {quote.ask_price}")
        print(f"   价差: {quote.spread}")
    except Exception as e:
        print(f"   ✗ 获取失败: {e}")

    # 测试标的信息
    print("\n4. 获取 AAPL 详细信息:")
    try:
        info = source.get_ticker_info("AAPL")
        print(f"   名称: {info.get('name')}")
        print(f"   行业: {info.get('industry')}")
        print(f"   市值: {info.get('market_cap', 0):,}")
        print(f"   市盈率: {info.get('pe_ratio')}")
    except Exception as e:
        print(f"   ✗ 获取失败: {e}")

    # 测试缓存
    print("\n5. 测试缓存功能:")
    try:
        # 第二次获取应该更快（从缓存读取）
        bars = source.get_bars("AAPL", start, end, "1d")
        print(f"   缓存命中，获取到 {len(bars)} 根K线")
    except Exception as e:
        print(f"   ✗ 缓存测试失败: {e}")

    # 测试混合数据源
    print("\n6. 测试混合数据源:")
    hybrid = HybridDataSource()
    bars = hybrid.get_bars("AAPL", start, end, "1d")
    print(f"   获取到 {len(bars)} 根K线")

    print("\n✓ 测试完成")
