#!/bin/bash
# Start only the FastAPI server

python -m uvicorn src.server.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
