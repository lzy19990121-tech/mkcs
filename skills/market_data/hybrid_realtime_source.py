"""
混合数据源 - Yahoo 历史数据 + Finnhub 实时数据

- Yahoo Finance: 获取历史K线数据
- Finnhub: 获取实时报价和 WebSocket 实时推送
- 组合使用：历史数据用于策略计算 + 实时数据用于快速交易

优势：
- 高频率更新（1-2秒）
- 实时价格无延时
- 完整的策略回测数据
"""

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

from core.models import Bar, Quote
from skills.market_data.base import MarketDataSource
from skills.market_data.yahoo_source import YahooFinanceSource

if TYPE_CHECKING:
    from skills.market_data.finnhub_source import FinnhubSource

logger = logging.getLogger(__name__)


class HybridRealtimeSource(MarketDataSource):
    """混合实时数据源

    Yahoo Finance + Finnhub 的最佳组合：
    - Yahoo: 历史K线（用于策略计算）
    - Finnhub: 实时报价 + WebSocket（用于快速交易）
    """

    def __init__(
        self,
        use_finnhub: bool = True,
        finnhub_api_key: Optional[str] = None,
        check_interval: float = 2.0,  # Finnhub API 调用间隔（秒）
        enable_websocket: bool = True,  # 是否启用 WebSocket
        websocket_interval: float = 1.0,  # WebSocket 更新间隔（秒）
    ):
        """初始化混合实时数据源

        Args:
            use_finnhub: 是否使用 Finnhub（否则只用 Yahoo）
            finnhub_api_key: Finnhub API Key
            check_interval: API 检查间隔（秒），避免触发频率限制
            enable_websocket: 是否启用 WebSocket 实时推送
            websocket_interval: WebSocket 数据更新间隔
        """
        # Yahoo 数据源（历史K线）
        self.yahoo = YahooFinanceSource()

        # Finnhub 数据源（实时报价）
        self.use_finnhub = use_finnhub
        self.finnhub = None
        self.check_interval = check_interval

        if use_finnhub:
            try:
                from skills.market_data.finnhub_source import FinnhubSource
                self.finnhub = FinnhubSource(api_key=finnhub_api_key)
                logger.info("Finnhub 数据源已启用（实时报价 + WebSocket）")
            except ImportError:
                logger.warning("Finnhub 模块不可用，回退到 Yahoo Finance")
                self.use_finnhub = False

        # WebSocket 相关
        self.enable_websocket = enable_websocket
        self.websocket_interval = websocket_interval
        self._ws_running = False
        self._ws_thread: Optional[threading.Thread] = None
        self._latest_quotes: Dict[str, Quote] = {}
        self._quote_lock = threading.Lock()
        self._on_price_update: Optional[Callable[[str, Decimal], None]] = None

        # 缓存
        self._last_api_call: float = 0
        self._api_call_lock = threading.Lock()

        mode_str = "Finnhub" if use_finnhub else "Yahoo"
        logger.info(f"混合实时数据源初始化完成: Yahoo + {mode_str}")

    def set_price_callback(self, callback: Callable[[str, Decimal], None]):
        """设置价格更新回调

        Args:
            callback: (symbol, price) -> None
        """
        self._on_price_update = callback

    def _rate_limit_check(self) -> bool:
        """检查是否允许 API 调用（Finnhub 限制 60次/分钟）"""
        if not self.use_finnhub or not self.finnhub:
            return False

        with self._api_call_lock:
            now = time.time()
            if now - self._last_api_call >= self.check_interval:
                self._last_api_call = now
                return True
            return False

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取K线数据（使用 Yahoo）"""
        return self.yahoo.get_bars(symbol, start, end, interval)

    def get_bars_until(self, symbol: str, end: datetime, interval: str) -> List[Bar]:
        """获取截至某个时间点的K线数据"""
        # 根据 interval 计算起始时间
        limits = {
            "1m": timedelta(days=7),
            "5m": timedelta(days=60),
            "15m": timedelta(days=60),
            "30m": timedelta(days=60),
            "1h": timedelta(days=730),
            "1d": timedelta(days=3650),
        }
        max_lookback = limits.get(interval, timedelta(days=365))
        start = end - max_lookback
        return self.get_bars(symbol, start, end, interval)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """获取实时报价"""
        # 优先使用 WebSocket 数据
        with self._quote_lock:
            if symbol in self._latest_quotes:
                return self._latest_quotes[symbol]

        # 回退到 API 调用
        if self.use_finnhub and self._rate_limit_check():
            try:
                quote = self.finnhub.get_quote(symbol)
                if quote:
                    self._update_quote(quote)
                    return quote
            except Exception as e:
                logger.debug(f"Finnhub quote 失败: {e}")

        # 回退到 Yahoo
        return self.yahoo.get_quote(symbol)

    def _update_quote(self, quote: Quote):
        """更新最新报价"""
        with self._quote_lock:
            self._latest_quotes[quote.symbol] = quote

        # 触发回调
        if self._on_price_update:
            try:
                self._on_price_update(quote.symbol, quote.mid_price)
            except Exception as e:
                logger.debug(f"价格回调错误: {e}")

    def _websocket_listener(self):
        """WebSocket 监听线程"""
        if not self.use_finnhub or not self.finnhub:
            return

        def on_trade(symbol: str, price: float, volume: int, timestamp: int):
            try:
                # 创建 Quote 对象
                quote = Quote(
                    symbol=symbol,
                    bid_price=Decimal(str(price)),
                    ask_price=Decimal(str(price)),
                    bid_size=volume,
                    ask_size=volume,
                    mid_price=Decimal(str(price)),
                    timestamp=datetime.fromtimestamp(timestamp),
                    source="finnhub-ws"
                )
                self._update_quote(quote)
            except Exception as e:
                logger.debug(f"WebSocket 交易处理错误: {e}")

        # 连接 WebSocket
        try:
            self.finnhub.connect_websocket(on_trade=on_trade)
        except Exception as e:
            logger.warning(f"WebSocket 连接失败: {e}")

    def start_realtime(self):
        """启动实时数据流"""
        if not self.enable_websocket or not self.use_finnhub:
            return

        if self._ws_running:
            logger.warning("实时数据流已在运行")
            return

        self._ws_running = True
        self._ws_thread = threading.Thread(target=self._websocket_listener, daemon=True)
        self._ws_thread.start()
        logger.info("实时数据流已启动 (WebSocket)")

    def stop_realtime(self):
        """停止实时数据流"""
        self._ws_running = False
        if self.finnhub:
            try:
                self.finnhub.disconnect_websocket()
            except Exception as e:
                logger.debug(f"WebSocket 断开错误: {e}")
        logger.info("实时数据流已停止")

    def get_realtime_price(self, symbol: str) -> Optional[Decimal]:
        """获取最新实时价格（用于交易）"""
        with self._quote_lock:
            if symbol in self._latest_quotes:
                return self._latest_quotes[symbol].mid_price
        return None

    def clear_cache(self):
        """清空缓存"""
        self.yahoo.clear_cache()
        with self._quote_lock:
            self._latest_quotes.clear()
        logger.info("缓存已清空")


def create_hybrid_source(
    use_finnhub: bool = True,
    finnhub_api_key: Optional[str] = None,
    check_interval: float = 2.0,
    enable_websocket: bool = True
) -> HybridRealtimeSource:
    """创建混合实时数据源的工厂函数"""
    return HybridRealtimeSource(
        use_finnhub=use_finnhub,
        finnhub_api_key=finnhub_api_key,
        check_interval=check_interval,
        enable_websocket=enable_websocket
    )
