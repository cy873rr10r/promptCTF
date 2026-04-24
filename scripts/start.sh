#!/bin/bash
# Start PromptCTF-Env services

set -e

echo "Starting PromptCTF-Env..."

# Create directories if they don't exist
mkdir -p models output data logs

# Start FastAPI server
echo "Starting FastAPI server on port 8000..."
python -m uvicorn src.server.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info &

SERVER_PID=$!

# Wait a bit for server to start
sleep 3

# Start Gradio UI
echo "Starting Gradio UI on port 7860..."
python -m src.ui.app &

UI_PID=$!

# Print URLs
echo ""
echo "========================================"
echo "PromptCTF-Env is running:"
echo "  API:    http://localhost:8000"
echo "  Docs:   http://localhost:8000/docs"
echo "  UI:     http://localhost:7860"
echo "========================================"
echo ""

# Keep services running
wait $SERVER_PID $UI_PID
