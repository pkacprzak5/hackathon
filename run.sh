#!/bin/bash
# Run both the squat coach server and the frontend dev server.
# Uses HTTPS so phone camera works over LAN.
# Usage: ./run.sh

set -e

# Get the local IP address for LAN access
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "localhost")
else
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
fi

CERT_DIR=".certs"

# Generate self-signed certificate if not exists
if [ ! -f "$CERT_DIR/cert.pem" ]; then
    echo "Generating self-signed SSL certificate..."
    mkdir -p "$CERT_DIR"
    openssl req -x509 -newkey rsa:2048 -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
        -days 365 -nodes -subj "/CN=${LOCAL_IP}" \
        -addext "subjectAltName=IP:${LOCAL_IP},DNS:localhost" 2>/dev/null
    echo "Certificate generated."
fi

echo "================================================"
echo "  Squat Coach - Starting servers"
echo "================================================"
echo ""
echo "  Backend (WSS):    https://${LOCAL_IP}:8000"
echo "  Frontend (HTTPS): https://${LOCAL_IP}:3000"
echo ""
echo "  STEP 1: On your phone, open https://${LOCAL_IP}:8000/health"
echo "          Accept the certificate warning."
echo ""
echo "  STEP 2: Then open https://${LOCAL_IP}:3000"
echo "          Accept the certificate warning."
echo ""
echo "  STEP 3: Tap profile → enter name → Home → Solo Session"
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

# Write .env.local with the correct server IP (wss for HTTPS context)
echo "NEXT_PUBLIC_ANALYSIS_WS_URL=wss://${LOCAL_IP}:8000/ws/session" > .env.local

cd ..

# Start backend server with SSL in background
echo "[1/2] Starting backend server on port 8000 (HTTPS)..."
python -m uvicorn squat_coach.server.main:app \
    --host 0.0.0.0 --port 8000 \
    --ssl-keyfile "$CERT_DIR/key.pem" \
    --ssl-certfile "$CERT_DIR/cert.pem" &
BACKEND_PID=$!

# Start frontend dev server with HTTPS (required for camera on phone)
echo "[2/2] Starting frontend on port 3000 (HTTPS)..."
cd frontend
pnpm dev --hostname 0.0.0.0 --experimental-https &
FRONTEND_PID=$!

cd ..

# Handle Ctrl+C to kill both
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

echo ""
echo "Both servers running. Press Ctrl+C to stop."
echo ""

# Wait for either to exit
wait
