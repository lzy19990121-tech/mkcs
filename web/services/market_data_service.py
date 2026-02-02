"""
市场数据服务 - 封装 YahooFinanceSource

提供 K 线数据、实时报价的获取接口
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from typing import List as TypingList

from skills.market_data.yahoo_source import YahooFinanceSource
from core.models import Bar, Quote

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    市场数据服务

    封装 YahooFinanceSource，提供简化的 API
    """

    def __init__(self):
        self._data_source = YahooFinanceSource()

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
            end = datetime.utcnow()
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


# 单例实例
market_data_service = MarketDataService()
