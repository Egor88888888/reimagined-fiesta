#!/bin/bash
# ============================================================
# DocLens — Quick Start Script
# Supports both development and production modes
# ============================================================
set -e

echo "🔍 DocLens — Document Recognition Engine"
echo "========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install: brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $PYTHON_VERSION"

# Check for Russian language data (CRITICAL for passport recognition)
if ! tesseract --list-langs 2>/dev/null | grep -q "rus"; then
    echo ""
    echo "⚠️  =============================================  ⚠️"
    echo "⚠️  РУССКИЙ ЯЗЫК НЕ УСТАНОВЛЕН В TESSERACT!       ⚠️"
    echo "⚠️  Без него невозможно распознавать кириллицу.    ⚠️"
    echo "⚠️  =============================================  ⚠️"
    echo ""
    echo "📦 Устанавливаю языковые пакеты Tesseract..."
    echo "   (это может занять 1-2 минуты)"

    # Try brew first (macOS)
    if command -v brew &> /dev/null; then
        brew install tesseract-lang
    # Try apt (Linux)
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y tesseract-ocr-rus
    fi

    # Verify installation
    if tesseract --list-langs 2>/dev/null | grep -q "rus"; then
        echo "✓ Русский язык установлен!"
    else
        echo ""
        echo "❌ ОШИБКА: Русский язык НЕ установился!"
        echo "   Попробуйте вручную:"
        echo "   macOS:  brew install tesseract tesseract-lang"
        echo "   Linux:  sudo apt-get install tesseract-ocr tesseract-ocr-rus"
        echo ""
    fi
else
    echo "✓ Tesseract Russian language data found"
fi

# Show available languages
echo "  Доступные языки: $(tesseract --list-langs 2>/dev/null | tail -n +2 | tr '\n' ', ')"

# Check tessdata quality (best > standard > fast)
if [ -f "/usr/share/tessdata/rus.traineddata" ]; then
    RUS_SIZE=$(stat -c%s "/usr/share/tessdata/rus.traineddata" 2>/dev/null || stat -f%z "/usr/share/tessdata/rus.traineddata" 2>/dev/null || echo "0")
elif [ -f "/opt/homebrew/share/tessdata/rus.traineddata" ]; then
    RUS_SIZE=$(stat -f%z "/opt/homebrew/share/tessdata/rus.traineddata" 2>/dev/null || stat -c%s "/opt/homebrew/share/tessdata/rus.traineddata" 2>/dev/null || echo "0")
else
    RUS_SIZE=0
fi

if [ "$RUS_SIZE" -gt 0 ]; then
    RUS_MB=$((RUS_SIZE / 1024 / 1024))
    if [ "$RUS_MB" -lt 5 ]; then
        echo ""
        echo "⚠️  Используется БЫСТРАЯ модель Tesseract (${RUS_MB}MB) — качество OCR для кириллицы будет низким!"
        echo "   Для значительно лучшего распознавания ФИО скачайте tessdata_best:"
        echo "   curl -L -o /usr/share/tessdata/rus.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata"
        echo ""
    elif [ "$RUS_MB" -lt 25 ]; then
        echo "  tessdata: стандартная модель (${RUS_MB}MB)"
    else
        echo "  tessdata: ЛУЧШАЯ модель (${RUS_MB}MB) ✓"
    fi
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -q fastapi uvicorn[standard] python-multipart numpy opencv-python-headless Pillow jinja2 aiofiles pytesseract pillow-heif PyMuPDF easyocr

# ============================================================
# Production Mode (with gunicorn)
# ============================================================
if [ "$PRODUCTION" = "1" ]; then
    echo ""
    echo "🚀 Starting DocLens in PRODUCTION mode"
    echo "========================================="

    # Parse workers flag
    WORKERS=${WORKERS:-4}
    WORKERS_ARG="${WORKERS}"

    # Check if --workers flag was passed
    for arg in "$@"; do
        if [ "$arg" = "--workers" ]; then
            shift
            WORKERS_ARG="$1"
            shift
        fi
    done

    # Install gunicorn if not present
    pip install -q gunicorn

    echo "Starting with $WORKERS_ARG workers..."
    echo ""

    exec gunicorn run_local:app \
        --workers "$WORKERS_ARG" \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 \
        --access-logfile - \
        --error-logfile - \
        --log-level info
fi

# ============================================================
# Development Mode (uvicorn single worker)
# ============================================================
echo ""
echo "🚀 Starting DocLens in DEVELOPMENT mode on http://localhost:8000"
echo "   Demo:      http://localhost:8000/demo"
echo "   Dashboard: http://localhost:8000/dashboard"
echo "   API Docs:  http://localhost:8000/api/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "========================================="
echo ""

python3 run_local.py
