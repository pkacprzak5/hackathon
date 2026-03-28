#!/bin/bash
# Run both the squat coach server and the frontend dev server.
# Usage: ./run.sh

set -e

# Get the local IP address for LAN access
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "localhost")
else
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
fi

echo "================================================"
echo "  Squat Coach - Starting servers"
echo "================================================"
echo ""
echo "  Server (backend):  http://${LOCAL_IP}:8000"
echo "  Frontend:          http://${LOCAL_IP}:3000"
echo ""
echo "  On your phone, open: http://${LOCAL_IP}:3000"
echo "  Then go to: Solo Session (from home page)"
echo ""
echo "================================================"
echo ""

# Install Python deps if needed
pip install -q -r squat_coach/requirements.txt

# Install frontend deps if needed
cd frontend
if [ ! -d "node_modules" ]; then
    pnpm install
fi

# Write .env.local with the correct server IP
echo "NEXT_PUBLIC_ANALYSIS_WS_URL=ws://${LOCAL_IP}:8000/ws/session" > .env.local

cd ..

# Start backend server in background
echo "[1/2] Starting backend server on port 8000..."
python -m uvicorn squat_coach.server.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start frontend dev server
echo "[2/2] Starting frontend on port 3000..."
cd frontend
pnpm dev --hostname 0.0.0.0 &
FRONTEND_PID=$!

cd ..

# Handle Ctrl+C to kill both
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

echo ""
echo "Both servers running. Press Ctrl+C to stop."
echo ""

# Wait for either to exit
wait
