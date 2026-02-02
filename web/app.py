"""
Web 应用主模块

提供回测结果可视化、实时交易 UI、权益曲线展示、交易明细查看等功能
"""

import json
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from flask import Flask, render_template, jsonify, request, send_from_directory

# 导入 WebSocket 和 API
from web.socketio_server import create_socketio
from web.api import register_api
from web.db import init_database

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 Decimal 和 datetime"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """创建 Flask 应用"""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / 'templates'),
        static_folder=str(Path(__file__).parent / 'static')
    )
    app.json_encoder = CustomJSONEncoder

    # 默认配置
    app.config.update({
        'DATA_DIR': 'reports/replay',
        'DEBUG': False,
        'DATA_DIR': 'data',
    })

    if config:
        app.config.update(config)

    # 确保数据目录存在
    os.makedirs(app.config.get('DATA_DIR', 'data'), exist_ok=True)

    # 初始化数据库
    try:
        init_database()
    except Exception as e:
        logger.warning(f"数据库初始化失败: {e}")

    # 注册 API 蓝图
    register_api(app)

    # 注册路由
    register_routes(app)

    # 初始化 SocketIO（放在最后确保所有蓝图已注册）
    try:
        socketio = create_socketio(app)
        app.socketio = socketio
        logger.info("SocketIO 服务器已初始化")
    except Exception as e:
        logger.warning(f"SocketIO 初始化失败: {e}")

    return app


def register_routes(app: Flask):
    """注册路由"""

    @app.route('/')
    def index():
        """主页 - 返回 React 应用"""
        frontend_path = Path(__file__).parent / 'frontend' / 'dist'
        index_path = frontend_path / 'index.html'

        if index_path.exists():
            return send_from_directory(frontend_path, 'index.html')
        else:
            # 如果 React 未构建，回退到原始模板
            return render_template('index.html')

    @app.route('/live')
    def live_trading():
        """实时交易页面"""
        return render_template('index.html')

    @app.route('/dashboard')
    def dashboard():
        """仪表盘页面"""
        return render_template('index.html')

    # React 静态文件服务
    @app.route('/static/react/<path:filename>')
    def serve_react_static(filename):
        """服务 React 构建的静态文件"""
        static_path = Path(__file__).parent / 'frontend' / 'dist' / 'assets'
        return send_from_directory(static_path, filename)

    @app.route('/api/backtests')
    def list_backtests():
        """列出所有回测结果"""
        data_dir = Path(app.config['DATA_DIR'])

        if not data_dir.exists():
            return jsonify([])

        backtests = []
        for summary_file in data_dir.glob('**/summary.json'):
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    backtests.append({
                        'id': summary_file.parent.name,
                        'date': data.get('backtest_date', ''),
                        'start_date': data.get('start_date', ''),
                        'end_date': data.get('end_date', ''),
                        'final_equity': data.get('final_equity', 0),
                        'total_return': data.get('total_return', 0),
                        'path': str(summary_file.parent)
                    })
            except Exception as e:
                logger.warning(f"读取回测结果失败: {e}")

        return jsonify(sorted(backtests, key=lambda x: x['date'], reverse=True))

    @app.route('/api/backtest/<backtest_id>')
    def get_backtest(backtest_id: str):
        """获取单个回测详情"""
        data_dir = Path(app.config['DATA_DIR'])
        backtest_dir = data_dir / backtest_id

        if not backtest_dir.exists():
            return jsonify({'error': 'Backtest not found'}), 404

        result = {}

        # 加载 summary
        summary_file = backtest_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                result['summary'] = json.load(f)

        # 加载权益曲线
        equity_file = backtest_dir / 'equity_curve.csv'
        if equity_file.exists():
            result['equity_curve'] = load_equity_curve(equity_file)

        # 加载交易记录
        trades_file = backtest_dir / 'trades.csv'
        if trades_file.exists():
            result['trades'] = load_trades(trades_file)

        # 加载风控拒绝记录
        rejects_file = backtest_dir / 'risk_rejects.csv'
        if rejects_file.exists():
            result['risk_rejects'] = load_risk_rejects(rejects_file)

        return jsonify(result)

    @app.route('/api/backtest/<backtest_id>/metrics')
    def get_metrics(backtest_id: str):
        """获取回测性能指标"""
        data_dir = Path(app.config['DATA_DIR'])
        backtest_dir = data_dir / backtest_id

        # 检查是否有预计算的 metrics
        metrics_file = backtest_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return jsonify(json.load(f))

        # 否则从 summary 和 trades 计算
        summary_file = backtest_dir / 'summary.json'
        if not summary_file.exists():
            return jsonify({'error': 'Backtest not found'}), 404

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        # 返回基本指标
        return jsonify({
            'total_return': summary.get('total_return', 0),
            'total_trades': summary.get('total_trades', 0),
            'win_rate': summary.get('win_rate', 0),
            'sharpe_ratio': summary.get('sharpe_ratio', 0),
        })

    @app.route('/api/health')
    def health_check():
        """健康检查"""
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat()
        })


def load_equity_curve(file_path: Path) -> List[Dict]:
    """加载权益曲线数据"""
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 跳过表头
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            data.append({
                'timestamp': parts[0],
                'equity': float(parts[1]),
                'cash': float(parts[2]) if len(parts) > 2 else None
            })

    return data


def load_trades(file_path: Path) -> List[Dict]:
    """加载交易记录"""
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 跳过表头
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 7:
            data.append({
                'timestamp': parts[0],
                'symbol': parts[1],
                'side': parts[2],
                'price': float(parts[3]),
                'quantity': int(parts[4]),
                'commission': float(parts[5]),
                'pnl': float(parts[6]) if parts[6] else None
            })

    return data


def load_risk_rejects(file_path: Path) -> List[Dict]:
    """加载风控拒绝记录"""
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 跳过表头
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 5:
            data.append({
                'timestamp': parts[0],
                'symbol': parts[1],
                'action': parts[2],
                'reason': parts[3],
                'confidence': float(parts[4]) if len(parts) > 4 else None
            })

    return data


def main():
    """启动 Web 应用"""
    import argparse

    parser = argparse.ArgumentParser(description='Trading System Web UI')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--data-dir', default='reports/replay', help='Data directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--frontend', action='store_true', help='Serve React frontend')

    args = parser.parse_args()

    app = create_app({
        'DATA_DIR': args.data_dir,
        'DEBUG': args.debug
    })

    print(f"\n🚀 MKCS Trading System Web UI")
    print(f"   访问地址: http://{args.host}:{args.port}")
    print(f"   调试模式: {'开启' if args.debug else '关闭'}")
    print("")

    if args.debug:
        # 开发模式：使用 socketio.run
        if hasattr(app, 'socketio'):
            app.socketio.run(app, host=args.host, port=args.port, debug=args.debug)
        else:
            app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        # 生产模式
        if hasattr(app, 'socketio'):
            app.socketio.run(app, host=args.host, port=args.port)
        else:
            app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
