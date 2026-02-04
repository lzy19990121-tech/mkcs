"""
Finnhub 数据源适配器

从 Finnhub 获取实时市场数据
支持美股、加密货币、外汇等

依赖: finnhub-python>=2023.8.0
API Key: 免费，注册即可获得 https://finnhub.io/
"""

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Callable

import requests
import websocket

from core.models import Bar, Quote
from skills.market_data.base import MarketDataSource

logger = logging.getLogger(__name__)


class FinnhubSource(MarketDataSource):
    """Finnhub 数据源

    从 Finnhub 获取实时股票数据
    免费额度: 60次/分钟
    文档: https://finnhub.io/docs/api
    """

    # API 基础地址
    BASE_URL = "https://finnhub.io/api/v1"

    # 支持的时间周期
    SUPPORTED_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]

    # Finnhub 的时间周期映射
    INTERVAL_MAPPING = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "D",
        "1wk": "W",
        "1mo": "M",
    }

    # 数据限制（基于 Finnhub 免费版）
    INTERVAL_LIMITS = {
        "1m": timedelta(days=1),      # 1分钟最多1天（免费版限制）
        "5m": timedelta(days=1),
        "15m": timedelta(days=1),
        "30m": timedelta(days=30),
        "1h": timedelta(days=30),
        "4h": timedelta(days=365),
        "1d": timedelta(days=365),
        "1wk": timedelta(days=3650),
        "1mo": timedelta(days=3650),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """初始化 Finnhub 数据源

        Args:
            api_key: Finnhub API Key (默认从环境变量 FINNHUB_API_KEY 读取)
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        if not self.api_key:
            logger.warning("未设置 FINNHUB_API_KEY，请到 https://finnhub.io/ 注册获取免费 API Key")

        self.max_retries = max_retries
        self.timeout = timeout
        self._cache: Dict[str, List[Bar]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=1)  # 缓存1分钟

        # WebSocket 相关
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_running = False
        self._ws_callbacks: Dict[str, List[Callable]] = {}

    def _get_session(self) -> requests.Session:
        """获取 requests session"""
        session = requests.Session()
        session.headers.update({
            "X-Finnhub-Token": self.api_key,
            "Content-Type": "application/json"
        })
        return session

    def _check_cache(self, cache_key: str) -> Optional[List[Bar]]:
        """检查缓存"""
        if cache_key in self._cache:
            cache_time = self._cache_time.get(cache_key)
            if cache_time and datetime.now() - cache_time < self._cache_ttl:
                return self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, bars: List[Bar]):
        """设置缓存"""
        self._cache[cache_key] = bars
        self._cache_time[cache_key] = datetime.now()

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取K线数据

        Args:
            symbol: 标的代码 (如 AAPL, TSLA)
            start: 开始时间
            end: 结束时间
            interval: 时间周期

        Returns:
            K线数据列表
        """
        if not self.api_key:
            raise ValueError("未设置 Finnhub API Key，请设置 FINNHUB_API_KEY 环境变量")

        # 检查周期是否支持
        if interval not in self.INTERVAL_MAPPING:
            logger.warning(f"不支持的周期 {interval}，使用日线代替")
            interval = "1d"

        cache_key = f"{symbol}_{interval}_{start.date()}_{end.date()}"

        # 检查缓存
        cached = self._check_cache(cache_key)
        if cached:
            logger.debug(f"使用缓存数据: {symbol} ({len(cached)} bars)")
            return [b for b in cached if start <= b.timestamp <= end]

        # 获取数据
        bars = self._fetch_bars(symbol, start, end, interval)

        # 存入缓存
        if bars:
            self._set_cache(cache_key, bars)

        return bars

    def get_bars_until(
        self,
        symbol: str,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取截至某个时间点的K线数据"""
        # 根据 interval 确定合适的起始时间
        max_lookback = self.INTERVAL_LIMITS.get(interval, timedelta(days=365))
        start = end - max_lookback

        bars = self.get_bars(symbol, start, end, interval)

        # 返回按时间排序的数据
        return sorted(bars, key=lambda x: x.timestamp)

    def get_quote(self, symbol: str) -> Quote:
        """获取实时报价"""
        if not self.api_key:
            raise ValueError("未设置 Finnhub API Key")

        session = self._get_session()

        for attempt in range(self.max_retries):
            try:
                # 获取实时报价
                response = session.get(
                    f"{self.BASE_URL}/quote",
                    params={"symbol": symbol},
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                # 获取当前价格
                current_price = Decimal(str(data.get("c", 0)))  # c = current price
                change = Decimal(str(data.get("d", 0)))  # d = change
                change_pct = data.get("dp", 0)  # dp = percent change
                high = Decimal(str(data.get("h", 0)))  # h = high price of the day
                low = Decimal(str(data.get("l", 0)))  # l = low price of the day
                open_price = Decimal(str(data.get("o", 0)))  # o = open price of the day
                prev_close = current_price - change if current_price and change else None

                # 获取前一收盘价用于计算 spread
                timestamp = datetime.now()

                return Quote(
                    symbol=symbol,
                    timestamp=timestamp,
                    bid_price=current_price * Decimal("0.999") if current_price else Decimal("0"),  # 模拟买一
                    ask_price=current_price * Decimal("1.001") if current_price else Decimal("0"),   # 模拟卖一
                    bid_size=100,
                    ask_size=100,
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
        """从 Finnhub 获取K线数据"""
        session = self._get_session()
        finnhub_interval = self.INTERVAL_MAPPING.get(interval, "D")

        # Finnhub 使用 Unix 时间戳（秒）
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"获取 {symbol} 数据: {start} ~ {end}, 周期={interval}")

                response = session.get(
                    f"{self.BASE_URL}/stock/candle",
                    params={
                        "symbol": symbol,
                        "resolution": finnhub_interval,
                        "from": start_ts,
                        "to": end_ts,
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                # 检查响应
                if data.get("s") == "no_data":
                    logger.warning(f"未获取到数据: {symbol} ({start} ~ {end})")
                    return []

                # 转换为 Bar 对象
                bars = []
                timestamps = data.get("t", [])
                opens = data.get("o", [])
                highs = data.get("h", [])
                lows = data.get("l", [])
                closes = data.get("c", [])
                volumes = data.get("v", [])

                for i in range(len(timestamps)):
                    bar = Bar(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(timestamps[i]),
                        open=Decimal(str(round(opens[i], 4))),
                        high=Decimal(str(round(highs[i], 4))),
                        low=Decimal(str(round(lows[i], 4))),
                        close=Decimal(str(round(closes[i], 4))),
                        volume=int(volumes[i]) if i < len(volumes) else 0,
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

    def get_ticker_info(self, symbol: str) -> dict:
        """获取标的详细信息"""
        if not self.api_key:
            return {"symbol": symbol, "error": "未设置 API Key"}

        session = self._get_session()

        try:
            response = session.get(
                f"{self.BASE_URL}/stock/profile2",
                params={"symbol": symbol},
                timeout=self.timeout
            )
            response.raise_for_status()
            info = response.json()

            return {
                "symbol": symbol,
                "name": info.get("name", ""),
                "sector": info.get("gics", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCapitalization", 0),
                "pe_ratio": info.get("pe", 0),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
                "country": info.get("country", ""),
                "website": info.get("weburl", ""),
            }
        except Exception as e:
            logger.error(f"获取标的信息失败: {e}")
            return {"symbol": symbol, "error": str(e)}

    # ========== WebSocket 实时推送 ==========

    def connect_websocket(self, on_trade: Callable = None, on_news: Callable = None):
        """连接 Finnhub WebSocket 进行实时数据推送

        Args:
            on_trade: 交易回调函数，接收 (symbol, price, volume, timestamp)
            on_news: 新闻回调函数
        """
        if not self.api_key:
            logger.error("未设置 API Key，无法连接 WebSocket")
            return

        if self._ws_running:
            logger.warning("WebSocket 已在运行")
            return

        self._ws_running = True

        def on_message(ws, message):
            try:
                import json
                data = json.loads(message)

                if data.get("type") == "trade":
                    for trade in data.get("data", []):
                        symbol = trade.get("s")
                        price = trade.get("p")
                        volume = trade.get("v")
                        timestamp = trade.get("t")

                        if on_trade and symbol and price:
                            on_trade(symbol, price, volume, timestamp)

                elif data.get("type") == "news":
                    if on_news:
                        on_news(data.get("data"))

            except Exception as e:
                logger.error(f"WebSocket 消息处理错误: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket 错误: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket 连接关闭")
            self._ws_running = False

        def on_open(ws):
            logger.info("WebSocket 连接成功")

        # WebSocket URL
        ws_url = f"wss://ws.finnhub.io?token={self.api_key}"

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # 在新线程中运行
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 30},
            daemon=True
        )
        self._ws_thread.start()

    def subscribe_symbols(self, symbols: List[str]):
        """订阅标的的实时数据

        Args:
            symbols: 标的代码列表
        """
        if not self._ws or not self._ws_running:
            logger.warning("WebSocket 未连接，请先调用 connect_websocket()")
            return

        def subscribe():
            import json
            for symbol in symbols:
                msg = json.dumps({"type": "subscribe", "symbol": symbol})
                self._ws.send(msg)
                logger.info(f"订阅 {symbol}")

        # 延迟订阅，等待连接建立
        threading.Timer(1.0, subscribe).start()

    def unsubscribe_symbols(self, symbols: List[str]):
        """取消订阅"""
        if not self._ws or not self._ws_running:
            return

        import json
        for symbol in symbols:
            msg = json.dumps({"type": "unsubscribe", "symbol": symbol})
            self._ws.send(msg)
            logger.info(f"取消订阅 {symbol}")

    def disconnect_websocket(self):
        """断开 WebSocket 连接"""
        if self._ws:
            self._ws.close()
            self._ws_running = False
            logger.info("WebSocket 已断开")

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_time.clear()
        logger.info("缓存已清空")


# 支持加密货币
class FinnhubCryptoSource(FinnhubSource):
    """Finnhub 加密货币数据源

    支持主流加密货币的实时数据
    """

    # 常见加密货币映射
    BINANCE_SYMBOLS = {
        "BTC": "binance:BTCUSDT",
        "ETH": "binance:ETHUSDT",
        "BNB": "binance:BNBUSDT",
        "SOL": "binance:SOLUSDT",
        "XRP": "binance:XRPUSDT",
        "ADA": "binance:ADAUSDT",
        "DOGE": "binance:DOGEUSDT",
        "MATIC": "binance:MATICUSDT",
        "DOT": "binance:DOTUSDT",
    }

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取加密货币K线数据"""
        # 转换符号
        finnhub_symbol = self.BINANCE_SYMBOLS.get(symbol.upper(), symbol)

        # 对于加密货币，直接调用父类方法
        return super().get_bars(finnhub_symbol, start, end, interval)

    def get_quote(self, symbol: str) -> Quote:
        """获取加密货币实时报价"""
        finnhub_symbol = self.BINANCE_SYMBOLS.get(symbol.upper(), symbol)
        quote = super().get_quote(finnhub_symbol)
        # 返回原始符号
        quote.symbol = symbol.upper()
        return quote


if __name__ == "__main__":
    """自测代码"""
    print("=== Finnhub 数据源测试 ===\n")

    # 创建数据源
    source = FinnhubSource()

    if not source.api_key:
        print("⚠ 请先设置 FINNHUB_API_KEY 环境变量")
        print("  1. 访问 https://finnhub.io/ 注册")
        print("  2. 获取免费 API Key")
        print("  3. export FINNHUB_API_KEY='your_key_here'")
    else:
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

        # 测试实时报价
        print("\n2. 获取 AAPL 实时报价:")
        try:
            quote = source.get_quote("AAPL")
            print(f"   买一: {quote.bid_price}")
            print(f"   卖一: {quote.ask_price}")
            print(f"   中间价: {quote.mid_price}")
        except Exception as e:
            print(f"   ✗ 获取失败: {e}")

        # 测试标的信息
        print("\n3. 获取 AAPL 详细信息:")
        try:
            info = source.get_ticker_info("AAPL")
            print(f"   名称: {info.get('name')}")
            print(f"   行业: {info.get('industry')}")
            print(f"   市值: {info.get('market_cap', 0):,}")
        except Exception as e:
            print(f"   ✗ 获取失败: {e}")

        print("\n✓ 测试完成")
