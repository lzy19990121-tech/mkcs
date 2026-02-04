"""
混合数据源 - 结合多个数据源的优势

组合方式:
- K 线历史数据: Yahoo Finance (免费，完整)
- 实时报价: Finnhub (免费，实时报价)

这样可以在不付费的情况下获得最好的数据质量
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from skills.market_data.base import MarketDataSource
from skills.market_data.yahoo_source import YahooFinanceSource
from skills.market_data.finnhub_source import FinnhubSource
from core.models import Bar, Quote

logger = logging.getLogger(__name__)


class HybridDataSource(MarketDataSource):
    """混合数据源

    结合 Yahoo Finance 和 Finnhub 的优势：
    - K 线数据使用 Yahoo Finance（完整历史，免费）
    - 实时报价使用 Finnhub（实时，免费）
    """

    def __init__(
        self,
        finnhub_api_key: Optional[str] = None,
        enable_fallback: bool = True
    ):
        """初始化混合数据源

        Args:
            finnhub_api_key: Finnhub API Key（可选，默认从环境变量读取）
            enable_fallback: 当 Finnhub 失败时是否回退到 Yahoo
        """
        self.enable_fallback = enable_fallback

        # Yahoo Finance 用于 K 线数据
        self.yahoo_source = YahooFinanceSource()

        # Finnhub 用于实时报价
        self.finnhub_source = None
        try:
            self.finnhub_source = FinnhubSource(api_key=finnhub_api_key)
            # 测试一下 Finnhub 是否可用
            test_quote = self.finnhub_source.get_quote("AAPL")
            if test_quote and test_quote.mid_price > 0:
                logger.info("Finnhub 实时报价已启用")
            else:
                logger.warning("Finnhub 测试失败，将使用 Yahoo Finance")
                self.finnhub_source = None
        except Exception as e:
            logger.warning(f"Finnhub 初始化失败: {e}，将使用 Yahoo Finance")
            self.finnhub_source = None

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取K线数据

        使用 Yahoo Finance 获取 K 线数据
        """
        return self.yahoo_source.get_bars(symbol, start, end, interval)

    def get_bars_until(
        self,
        symbol: str,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取截至某个时间点的K线数据"""
        return self.yahoo_source.get_bars_until(symbol, end, interval)

    def get_quote(self, symbol: str) -> Quote:
        """获取实时报价

        优先使用 Finnhub（实时），失败则使用 Yahoo Finance（延迟）
        """
        # 首先尝试 Finnhub
        if self.finnhub_source:
            try:
                quote = self.finnhub_source.get_quote(symbol)
                if quote and quote.mid_price and quote.mid_price > 0:
                    logger.debug(f"使用 Finnhub 实时报价: {symbol} = {quote.mid_price}")
                    return quote
            except Exception as e:
                logger.debug(f"Finnhub 获取 {symbol} 报价失败: {e}")

        # 回退到 Yahoo Finance
        if self.enable_fallback:
            try:
                quote = self.yahoo_source.get_quote(symbol)
                logger.debug(f"使用 Yahoo Finance 报价: {symbol} = {quote.mid_price if quote else 'N/A'}")
                return quote
            except Exception as e:
                logger.error(f"Yahoo Finance 获取 {symbol} 报价也失败: {e}")

        raise RuntimeError(f"无法获取 {symbol} 的报价")

    def get_source_info(self) -> dict:
        """获取当前数据源信息"""
        return {
            "bars_source": "Yahoo Finance",
            "quote_source": "Finnhub" if self.finnhub_source else "Yahoo Finance",
            "finnhub_available": self.finnhub_source is not None,
        }


class SmartDataSource(MarketDataSource):
    """智能数据源

    根据市场状态和请求类型自动选择最佳数据源：
    - 美股实时报价: Finnhub
    - 历史K线: Yahoo Finance
    - 加密货币: Binance API（如果有实现）
    - 其他市场: Yahoo Finance
    """

    def __init__(self, finnhub_api_key: Optional[str] = None):
        """初始化智能数据源"""
        self.yahoo = YahooFinanceSource()
        self.finnhub = None
        self.hybrid = None

        # 尝试初始化 Finnhub
        try:
            self.finnhub = FinnhubSource(api_key=finnhub_api_key)
            # 测试
            test_quote = self.finnhub.get_quote("AAPL")
            if test_quote and test_quote.mid_price > 0:
                logger.info("智能数据源: Finnhub 可用")
                self.hybrid = HybridDataSource(api_key=finnhub_api_key)
        except Exception:
            logger.info("智能数据源: 仅 Yahoo Finance 可用")

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取K线数据 - 使用 Yahoo Finance"""
        return self.yahoo.get_bars(symbol, start, end, interval)

    def get_bars_until(
        self,
        symbol: str,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取K线数据 - 使用 Yahoo Finance"""
        return self.yahoo.get_bars_until(symbol, end, interval)

    def get_quote(self, symbol: str) -> Quote:
        """获取实时报价 - 优先 Finnhub，否则 Yahoo"""
        if self.hybrid:
            return self.hybrid.get_quote(symbol)
        return self.yahoo.get_quote(symbol)

    def get_source_info(self) -> dict:
        """获取数据源信息"""
        info = {
            "mode": "Smart",
            "yahoo_available": True,
            "finnhub_available": self.finnhub is not None,
        }
        if self.hybrid:
            info.update(self.hybrid.get_source_info())
        return info


# 便捷函数
def create_hybrid_source(api_key: Optional[str] = None) -> HybridDataSource:
    """创建混合数据源"""
    return HybridDataSource(finnhub_api_key=api_key)


def create_smart_source(api_key: Optional[str] = None) -> SmartDataSource:
    """创建智能数据源"""
    return SmartDataSource(finnhub_api_key=api_key)


if __name__ == "__main__":
    """测试混合数据源"""
    print("=== 混合数据源测试 ===\n")

    hybrid = HybridDataSource()

    print(f"数据源配置: {hybrid.get_source_info()}\n")

    # 测试 K 线（来自 Yahoo Finance）
    print("1. 获取 AAPL 最近 3 天 K 线 (Yahoo Finance):")
    end = datetime.now()
    start = end - timedelta(days=3)
    bars = hybrid.get_bars("AAPL", start, end, "1d")
    print(f"   获取 {len(bars)} 根 K 线")
    if bars:
        for bar in bars[-3:]:
            print(f"   {bar.timestamp.strftime('%Y-%m-%d')}: O={bar.open} C={bar.close}")

    # 测试实时报价（来自 Finnhub）
    print("\n2. 获取实时报价 (优先 Finnhub):")
    symbols = ["AAPL", "TSLA", "NVDA"]
    for symbol in symbols:
        try:
            quote = hybrid.get_quote(symbol)
            source = "Finnhub" if hybrid.finnhub_source else "Yahoo"
            print(f"   {symbol}: {quote.mid_price} ({source})")
        except Exception as e:
            print(f"   {symbol}: 失败 - {e}")

    print("\n✓ 测试完成")
