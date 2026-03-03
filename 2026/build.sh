#!/bin/bash
# ======================================================================
# 2026 考核题目 —— 打包脚本（Linux / macOS）
# 用途：将 main.py 打包为单文件可执行程序，同时排除 strategy.py
#       使得学生可以在同目录下修改 strategy.py 而无需重新编译
# 使用：chmod +x build.sh && ./build.sh
# ======================================================================

set -e

echo "========================================"
echo "  2026 考核题目打包 (Linux/macOS)"
echo "========================================"

# 清理旧产物
rm -rf build dist __pycache__ main.spec

# 打包
pyinstaller main.py \
    --onefile \
    --nowindowed \
    --noconfirm \
    --name main \
    --exclude-module strategy \
    --hidden-import='PIL._tkinter_finder' \
    --hidden-import=matplotlib.backends.backend_tkagg

echo ""
echo "打包完成！可执行文件位于 dist/main"
echo ""
echo "发布时，将以下文件放在同一目录下交给学生："
echo "  dist/main      ← 可执行文件"
echo "  strategy.py    ← 学生编辑文件"
echo "  README.md      ← 题目说明"
