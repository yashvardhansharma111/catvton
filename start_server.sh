#!/bin/bash
# Linux/macOS script to start CatVTON FastAPI server

echo "Starting CatVTON FastAPI Server..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if CUDA is available
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null

echo ""
echo "Server will start on http://0.0.0.0:8000"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 app_fastapi.py --host 0.0.0.0 --port 8000

