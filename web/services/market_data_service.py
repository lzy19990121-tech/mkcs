"""
市场数据服务 - 混合数据源模式

支持多种数据源组合:
- hybrid: Yahoo Finance (K线) + Finnhub (实时报价) - 推荐
- yahoo: 仅 Yahoo Finance
- finnhub: 仅 Finnhub (需要付费版才能获取K线)
- auto: 自动选择 (优先 hybrid)
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional
from typing import List as TypingList

from skills.market_data.yahoo_source import YahooFinanceSource
from skills.market_data.finnhub_source import FinnhubSource
from skills.market_data.hybrid_source import HybridDataSource, SmartDataSource
from core.models import Bar, Quote

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    市场数据服务

    支持混合数据源模式，获得最佳数据质量
    """

    def __init__(self, data_source: Optional[str] = None):
        """
        初始化市场数据服务

        Args:
            data_source: 数据源选择 ('hybrid', 'yahoo', 'finnhub', 'auto')
                        默认从环境变量 DATA_SOURCE 读取
        """
        self._source_name = data_source or os.getenv("DATA_SOURCE", "auto")
        self._data_source = self._init_data_source()

        # 记录数据源配置
        self._log_data_source_config()

    def _init_data_source(self) -> object:
        """初始化数据源"""
        source_name = self._source_name.lower()
        api_key = os.getenv("FINNHUB_API_KEY")

        # 自动选择或 hybrid
        if source_name in ["auto", "hybrid"]:
            if api_key:
                logger.info("使用混合数据源: Yahoo Finance (K线) + Finnhub (实时报价)")
                return HybridDataSource(finnhub_api_key=api_key)
            else:
                logger.info("使用 Yahoo Finance 数据源 (未设置 FINNHUB_API_KEY)")
                return YahooFinanceSource()

        # 显式指定
        if source_name == "yahoo":
            logger.info("使用 Yahoo Finance 数据源")
            return YahooFinanceSource()
        elif source_name == "smart":
            logger.info("使用智能数据源")
            return SmartDataSource(finnhub_api_key=api_key)
        elif source_name == "finnhub":
            if not api_key:
                logger.warning("未设置 FINNHUB_API_KEY，回退到 Yahoo Finance")
                return YahooFinanceSource()
            logger.info("使用 Finnhub 数据源")
            return FinnhubSource()
        else:
            logger.warning(f"未知的数据源 '{source_name}'，使用 Yahoo Finance")
            return YahooFinanceSource()

    def _log_data_source_config(self):
        """记录数据源配置"""
        info = self.get_source_info()
        logger.info(f"数据源配置: {info}")

    @property
    def source_name(self) -> str:
        """获取当前使用的数据源名称"""
        return self._data_source.__class__.__name__

    def get_bars(
        self,
        symbol: str,
        interval: str = '1d',
        days: int = 90
    ) -> List[Dict[str, Any]]:
        """
        获取 K 线数据

        Args:
            symbol: 股票代码
            interval: K 线周期 ('1m', '5m', '1h', '1d', '1wk', '1mo')
            days: 获取天数

        Returns:
            List of bar dictionaries with OHLCV data
        """
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)

            bars = self._data_source.get_bars(
                symbol=symbol,
                start=start,
                end=end,
                interval=interval
            )

            # 转换为字典格式
            result = []
            for bar in bars:
                result.append({
                    'symbol': bar.symbol,
                    'timestamp': bar.timestamp.isoformat() if bar.timestamp else None,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume) if bar.volume else 0,
                    'interval': bar.interval,
                })

            return result

        except Exception as e:
            logger.error(f"获取 K 线数据失败 {symbol}: {e}")
            return []

    def get_bars_with_ma(
        self,
        symbol: str,
        interval: str = '1d',
        days: int = 90,
        ma_periods: List[int] = [5, 20]
    ) -> Dict[str, Any]:
        """
        获取 K 线数据并计算移动平均线

        Args:
            symbol: 股票代码
            interval: K 线周期
            days: 获取天数
            ma_periods: MA 周期列表

        Returns:
            Dict with 'bars' and 'ma' data
        """
        bars = self.get_bars(symbol, interval, days)

        if not bars:
            return {'bars': [], 'ma': {}}

        # 计算 MA
        closes = [bar['close'] for bar in bars]
        ma_data = {}

        for period in ma_periods:
            if len(closes) >= period:
                ma_values = []
                for i in range(len(closes)):
                    if i < period - 1:
                        ma_values.append(None)
                    else:
                        ma = sum(closes[i - period + 1:i + 1]) / period
                        ma_values.append(round(ma, 2))
                ma_data[f'ma{period}'] = ma_values

        return {
            'bars': bars,
            'ma': ma_data,
        }

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取实时报价

        Args:
            symbol: 股票代码

        Returns:
            Quote dictionary or None
        """
        try:
            quote = self._data_source.get_quote(symbol)

            if quote is None:
                return None

            return {
                'symbol': quote.symbol,
                'timestamp': quote.timestamp.isoformat() if quote.timestamp else None,
                'bid_price': float(quote.bid_price) if quote.bid_price else None,
                'bid_size': int(quote.bid_size) if quote.bid_size else None,
                'ask_price': float(quote.ask_price) if quote.ask_price else None,
                'ask_size': int(quote.ask_size) if quote.ask_size else None,
                'spread': float(quote.spread) if quote.spread else None,
                'mid_price': float(quote.mid_price) if quote.mid_price else None,
            }

        except Exception as e:
            logger.error(f"获取报价失败 {symbol}: {e}")
            return None

    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取报价

        Args:
            symbols: 股票代码列表

        Returns:
            Dict mapping symbol to quote
        """
        quotes = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                quotes[symbol] = quote
        return quotes

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        获取最新价格

        Args:
            symbol: 股票代码

        Returns:
            Latest price or None
        """
        quote = self.get_quote(symbol)
        if quote:
            return quote.get('mid_price') or quote.get('last_price')
        return None

    def get_source_info(self) -> Dict[str, Any]:
        """
        获取当前数据源信息

        Returns:
            数据源信息字典
        """
        info = {
            'source_type': self._source_name,
            'source_class': self._data_source.__class__.__name__,
            'finnhub_configured': bool(os.getenv("FINNHUB_API_KEY")),
        }

        # 如果是混合数据源，获取更详细的信息
        if hasattr(self._data_source, 'get_source_info'):
            info.update(self._data_source.get_source_info())

        return info


# 单例实例
market_data_service = MarketDataService()
