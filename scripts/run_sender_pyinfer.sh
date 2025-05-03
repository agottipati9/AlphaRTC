#!/bin/bash
set -e

ALPHARTC_PATH="/opt/home_dir/AlphaRTC/out/Default"
ALPHARTC_SCRIPTS="/opt/home_dir/AlphaRTC/scripts"
SETUP_DELAY=3

SENDER_CONFIG=/opt/home_dir/AlphaRTC/configs/sender.json

function cleanup {
    echo "Cleaning up"
    pkill -f peerconnection_serverless || true
    pkill -f cmdinfer.py || true
    # Clean up the named pipes
    rm -f /tmp/cpp_to_py_* /tmp/py_to_cpp_* 2>/dev/null || true
}

trap cleanup EXIT SIGINT SIGTERM

OUTPUT_DIR=/opt/home_dir/outputs/

# Set up Python path
export PYTHONPATH="${ALPHARTC_SCRIPTS}:${PYTHONPATH}"

echo "Running the test call"

# Create named pipes for bidirectional communication
# For sender
mkfifo /tmp/cpp_to_py_sender
mkfifo /tmp/py_to_cpp_sender

# Start the Python scripts first (they'll wait for input)
python3 ${ALPHARTC_SCRIPTS}/cmdinfer.py < /tmp/cpp_to_py_sender > /tmp/py_to_cpp_sender 2>${OUTPUT_DIR}/sender_py.log &
SENDER_PY_PID=$!

sleep "${SETUP_DELAY}"

# Start Sender with bidirectional piping
(${ALPHARTC_PATH}/peerconnection_serverless ${SENDER_CONFIG} > /tmp/cpp_to_py_sender < /tmp/py_to_cpp_sender) 2>${OUTPUT_DIR}/sender.log