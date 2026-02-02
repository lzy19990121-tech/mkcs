#!/bin/bash
# Build React frontend for production

cd "$(dirname "$0")"

echo "安装前端依赖..."
npm install

echo ""
echo "构建生产版本..."
npm run build

echo ""
echo "=================================="
echo "✅ 构建完成!"
echo "=================================="
echo "构建产物位于: frontend/dist/"
echo "可以通过 Flask 访问: http://localhost:5000/"
echo ""
