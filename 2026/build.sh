#!/bin/bash
# ======================================================================
# 2026 Robotics Assessment -- Build script (Linux / macOS)
# Purpose: Package main.py into a single-file executable, while
#          excluding strategy.py so students can edit it externally
#          without recompiling the binary.
# Usage:   chmod +x build.sh && ./build.sh
# ======================================================================

set -e

echo "========================================"
echo "  2026 Assessment Build (Linux/macOS)"
echo "========================================"

# Clean previous build artifacts
rm -rf build dist __pycache__ main.spec

# Package
# --exclude-module strategy : keep strategy.py out of the bundle
# main.py loads strategy.py at runtime via importlib.util (absolute path)
pyinstaller main.py \
    --onefile \
    --nowindowed \
    --noconfirm \
    --name main \
    --exclude-module strategy \
    --hidden-import='PIL._tkinter_finder' \
    --hidden-import=matplotlib.backends.backend_tkagg \
    --hidden-import=importlib.util

echo ""
echo "Build complete!  Executable is at dist/main"
echo ""
echo "Files to distribute to students (place them in the same directory):"
echo "  dist/main      <- executable"
echo "  strategy.py    <- student edit file"
echo "  README.md      <- problem description"
