#!/bin/bash
# 启动 Jupyter Lab 的便捷脚本

echo "🚀 启动 CORP Notebook..."

# 激活虚拟环境
source venv/bin/activate

# 进入 notebooks 目录
cd notebooks

# 启动 Jupyter Lab
echo ""
echo "正在启动 Jupyter Lab..."
echo "启动后请在浏览器中打开显示的链接"
echo ""

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
