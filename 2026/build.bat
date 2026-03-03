@echo off
REM ======================================================================
REM 2026 Robotics Assessment -- Build script (Windows)
REM Purpose: Package main.py into a single-file executable, while
REM          excluding strategy.py so students can edit it externally
REM          without recompiling the binary.
REM Usage: Double-click build.bat, or run it in a cmd window.
REM ======================================================================

echo ========================================
echo   2026 Assessment Build (Windows)
echo ========================================

REM Clean previous build artifacts
if exist build     rmdir /s /q build
if exist dist      rmdir /s /q dist
if exist main.spec del /q main.spec

REM Package
REM   --exclude-module strategy   : keep strategy.py out of the bundle
REM   main.py loads strategy.py at runtime via importlib.util (absolute path)
pyinstaller main.py ^
    --onefile ^
    --nowindowed ^
    --noconfirm ^
    --name main ^
    --exclude-module strategy ^
    --hidden-import=PIL._tkinter_finder ^
    --hidden-import=matplotlib.backends.backend_tkagg ^
    --hidden-import=importlib.util

echo.
echo Build complete!  Executable is at dist\main.exe
echo.
echo Files to distribute to students (place them in the same directory):
echo   dist\main.exe   ^<-- executable
echo   strategy.py     ^<-- student edit file
echo   README.md       ^<-- problem description
echo.
pause
