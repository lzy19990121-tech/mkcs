#!/bin/bash
# MKCS Trading System - 启动脚本

# 设置工作目录
cd "$(dirname "$0")"

# 检查并安装依赖
echo "检查依赖..."
python3 -c "import flask; import flask_socketio; import sqlalchemy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "正在安装 Web 依赖..."
    python3 -m pip install -q flask-socketio python-socketio eventlet sqlalchemy flask-cors
fi

# 初始化数据库
echo "初始化数据库..."
python3 -c "from web.db import init_database; init_database()"

# 启动服务
echo ""
echo "=================================="
echo "🚀 启动 MKCS Trading System Web UI"
echo "=================================="
echo "访问地址: http://localhost:5000"
echo ""
echo "功能:"
echo "  • 实时行情和 K 线图"
echo "  • 交易下单和持仓管理"
echo "  • 风控状态监控"
echo "  • 交互式图表标注"
echo ""
echo "=================================="
echo ""

# 启动 Flask + SocketIO
python3 web/app.py --host 0.0.0.0 --port 5000 --debug
