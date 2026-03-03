@echo off
REM ======================================================================
REM 2026 考核题目 —— 打包脚本（Windows）
REM 用途：将 main.py 打包为单文件可执行程序，同时排除 strategy.py
REM       使得学生可以在同目录下修改 strategy.py 而无需重新编译
REM 使用：双击运行 build.bat，或在 cmd 中执行
REM ======================================================================

echo ========================================
echo   2026 考核题目打包 (Windows)
echo ========================================

REM 清理旧产物
if exist build     rmdir /s /q build
if exist dist      rmdir /s /q dist
if exist main.spec del /q main.spec

REM 打包
pyinstaller main.py ^
    --onefile ^
    --nowindowed ^
    --noconfirm ^
    --name main ^
    --exclude-module strategy ^
    --hidden-import=PIL._tkinter_finder ^
    --hidden-import=matplotlib.backends.backend_tkagg

echo.
echo 打包完成！可执行文件位于 dist\main.exe
echo.
echo 发布时，将以下文件放在同一目录下交给学生：
echo   dist\main.exe   ^<^-- 可执行文件
echo   strategy.py     ^<^-- 学生编辑文件
echo   README.md       ^<^-- 题目说明
echo.
pause
