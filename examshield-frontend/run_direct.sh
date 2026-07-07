#!/bin/bash
# ExamShield Direct Run (Linux/Mac) — for webcam access
# Docker Desktop cannot pass the webcam into containers on Windows.
# This script runs ExamShield locally while the backend stays in Docker.
#
# IMPORTANT: If Docker is running with all services, stop ExamShield first:
#   docker compose stop examshield-frontend

# Anchor CWD to this script's directory so the script works no matter where
# the user invokes it from (project root, desktop, etc.).  We restore the
# caller's CWD on exit via a trap so we don't leave the user's shell in
# examshield-frontend/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd "$SCRIPT_DIR" > /dev/null

cleanup() {
    popd > /dev/null
}
trap cleanup EXIT

PORT="${STREAMLIT_PORT:-8601}"

export BACKEND_API_URL=http://localhost:8000
export BROWSER_API_URL=http://localhost:8000

echo ""
echo "========================================"
echo "  ExamShield AI - Direct Run Mode"
echo "  Port: $PORT"
echo "  Backend: $BACKEND_API_URL"
echo "========================================"
echo ""
echo "TIP: If port $PORT is busy, set STREAMLIT_PORT:"
echo "  STREAMLIT_PORT=9001 ./run_direct.sh"
echo ""

# Do NOT use --server.address 0.0.0.0 on Windows with Hyper-V
# Let Streamlit default to localhost
streamlit run app.py --server.port "$PORT"
