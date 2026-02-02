"""
标注服务 - 标注 CRUD 操作

管理用户添加的买卖点和卖出区间
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from web.db import get_session, Annotation, SellRange, TradeRecord

logger = logging.getLogger(__name__)


class AnnotationService:
    """
    标注服务

    提供标注和卖出区间的 CRUD 操作
    """

    def __init__(self):
        pass

    # ============ 买卖点标注 ============

    def get_markers(self, symbol: str) -> List[Dict[str, Any]]:
        """获取指定股票的所有标注"""
        session = get_session()
        try:
            annotations = session.query(Annotation).filter(
                Annotation.symbol == symbol.upper()
            ).order_by(Annotation.timestamp.desc()).all()

            return [self._annotation_to_dict(a) for a in annotations]
        finally:
            session.close()

    def add_marker(
        self,
        symbol: str,
        marker_type: str,
        price: float,
        timestamp: datetime,
        quantity: Optional[int] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        添加买卖点标注

        Args:
            symbol: 股票代码
            marker_type: 'buy', 'sell', 'entry', 'exit'
            price: 价格
            timestamp: 时间点
            quantity: 可选数量
            notes: 可选备注

        Returns:
            Created annotation dict
        """
        session = get_session()
        try:
            annotation = Annotation(
                symbol=symbol.upper(),
                marker_type=marker_type,
                price=price,
                timestamp=timestamp,
                quantity=quantity,
                notes=notes,
            )
            session.add(annotation)
            session.commit()

            logger.info(f"添加标注: {symbol} {marker_type} @ {price}")
            return self._annotation_to_dict(annotation)

        except Exception as e:
            session.rollback()
            logger.error(f"添加标注失败: {e}")
            raise
        finally:
            session.close()

    def update_marker(
        self,
        marker_id: int,
        price: Optional[float] = None,
        quantity: Optional[int] = None,
        notes: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """更新标注"""
        session = get_session()
        try:
            annotation = session.query(Annotation).filter_by(id=marker_id).first()
            if not annotation:
                return None

            if price is not None:
                annotation.price = price
            if quantity is not None:
                annotation.quantity = quantity
            if notes is not None:
                annotation.notes = notes

            annotation.updated_at = datetime.utcnow()
            session.commit()

            return self._annotation_to_dict(annotation)

        except Exception as e:
            session.rollback()
            logger.error(f"更新标注失败: {e}")
            raise
        finally:
            session.close()

    def delete_marker(self, marker_id: int) -> bool:
        """删除标注"""
        session = get_session()
        try:
            annotation = session.query(Annotation).filter_by(id=marker_id).first()
            if not annotation:
                return False

            session.delete(annotation)
            session.commit()
            logger.info(f"删除标注: {marker_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"删除标注失败: {e}")
            raise
        finally:
            session.close()

    # ============ 卖出区间 ============

    def get_sell_ranges(self, symbol: str) -> List[Dict[str, Any]]:
        """获取指定股票的所有卖出区间"""
        session = get_session()
        try:
            ranges = session.query(SellRange).filter(
                SellRange.symbol == symbol.upper()
            ).order_by(SellRange.start_time.desc()).all()

            return [self._range_to_dict(r) for r in ranges]
        finally:
            session.close()

    def add_sell_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        target_price: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        添加卖出区间

        Args:
            symbol: 股票代码
            start_time: 区间开始时间
            end_time: 区间结束时间
            target_price: 可选目标价格
            notes: 可选备注

        Returns:
            Created range dict
        """
        session = get_session()
        try:
            sell_range = SellRange(
                symbol=symbol.upper(),
                start_time=start_time,
                end_time=end_time,
                target_price=target_price,
                notes=notes,
            )
            session.add(sell_range)
            session.commit()

            logger.info(f"添加卖出区间: {symbol} {start_time} - {end_time}")
            return self._range_to_dict(sell_range)

        except Exception as e:
            session.rollback()
            logger.error(f"添加卖出区间失败: {e}")
            raise
        finally:
            session.close()

    def update_sell_range(
        self,
        range_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        target_price: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """更新卖出区间"""
        session = get_session()
        try:
            sell_range = session.query(SellRange).filter_by(id=range_id).first()
            if not sell_range:
                return None

            if start_time is not None:
                sell_range.start_time = start_time
            if end_time is not None:
                sell_range.end_time = end_time
            if target_price is not None:
                sell_range.target_price = target_price
            if notes is not None:
                sell_range.notes = notes

            sell_range.updated_at = datetime.utcnow()
            session.commit()

            return self._range_to_dict(sell_range)

        except Exception as e:
            session.rollback()
            logger.error(f"更新卖出区间失败: {e}")
            raise
        finally:
            session.close()

    def delete_sell_range(self, range_id: int) -> bool:
        """删除卖出区间"""
        session = get_session()
        try:
            sell_range = session.query(SellRange).filter_by(id=range_id).first()
            if not sell_range:
                return False

            session.delete(sell_range)
            session.commit()
            logger.info(f"删除卖出区间: {range_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"删除卖出区间失败: {e}")
            raise
        finally:
            session.close()

    # ============ 交易记录 ============

    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取交易记录"""
        session = get_session()
        try:
            query = session.query(TradeRecord)
            if symbol:
                query = query.filter(TradeRecord.symbol == symbol.upper())

            trades = query.order_by(TradeRecord.trade_time.desc()).limit(limit).all()

            return [self._trade_to_dict(t) for t in trades]
        finally:
            session.close()

    def add_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: int,
        trade_time: datetime,
        commission: float = 0.0,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """添加交易记录"""
        session = get_session()
        try:
            trade = TradeRecord(
                symbol=symbol.upper(),
                side=side.lower(),
                price=price,
                quantity=quantity,
                trade_time=trade_time,
                commission=commission,
                notes=notes,
            )
            session.add(trade)
            session.commit()

            logger.info(f"添加交易记录: {symbol} {side} {quantity} @ {price}")
            return self._trade_to_dict(trade)

        except Exception as e:
            session.rollback()
            logger.error(f"添加交易记录失败: {e}")
            raise
        finally:
            session.close()

    # ============ 观察列表 ============

    def get_watchlist(self) -> List[Dict[str, Any]]:
        """获取观察列表"""
        from web.db import Watchlist
        session = get_session()
        try:
            items = session.query(Watchlist).order_by(Watchlist.sort_order).all()
            return [{'symbol': w.symbol, 'display_name': w.display_name} for w in items]
        finally:
            session.close()

    # ============ 辅助方法 ============

    def _annotation_to_dict(self, annotation: Annotation) -> Dict[str, Any]:
        """Annotation 转为字典"""
        return {
            'id': annotation.id,
            'symbol': annotation.symbol,
            'marker_type': annotation.marker_type.value if hasattr(annotation.marker_type, 'value') else annotation.marker_type,
            'price': annotation.price,
            'quantity': annotation.quantity,
            'timestamp': annotation.timestamp.isoformat() if annotation.timestamp else None,
            'notes': annotation.notes,
            'created_at': annotation.created_at.isoformat() if annotation.created_at else None,
            'updated_at': annotation.updated_at.isoformat() if annotation.updated_at else None,
        }

    def _range_to_dict(self, sell_range: SellRange) -> Dict[str, Any]:
        """SellRange 转为字典"""
        return {
            'id': sell_range.id,
            'symbol': sell_range.symbol,
            'start_time': sell_range.start_time.isoformat() if sell_range.start_time else None,
            'end_time': sell_range.end_time.isoformat() if sell_range.end_time else None,
            'target_price': sell_range.target_price,
            'notes': sell_range.notes,
            'created_at': sell_range.created_at.isoformat() if sell_range.created_at else None,
        }

    def _trade_to_dict(self, trade: TradeRecord) -> Dict[str, Any]:
        """TradeRecord 转为字典"""
        return {
            'id': trade.id,
            'symbol': trade.symbol,
            'side': trade.side,
            'price': trade.price,
            'quantity': trade.quantity,
            'commission': trade.commission,
            'realized_pnl': trade.realized_pnl,
            'notes': trade.notes,
            'trade_time': trade.trade_time.isoformat() if trade.trade_time else None,
        }


# 单例实例
annotation_service = AnnotationService()
