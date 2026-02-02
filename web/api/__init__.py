"""
__init__.py - API 模块

注册所有 API 蓝图
"""

from web.api.stocks import stocks_bp
from web.api.orders import orders_bp
from web.api.annotations import annotations_bp
from web.api.risk import risk_bp


def register_api(app):
    """注册所有 API 蓝图"""
    app.register_blueprint(stocks_bp)
    app.register_blueprint(orders_bp)
    app.register_blueprint(annotations_bp)
    app.register_blueprint(risk_bp)


__all__ = [
    'stocks_bp',
    'orders_bp',
    'annotations_bp',
    'risk_bp',
    'register_api',
]
