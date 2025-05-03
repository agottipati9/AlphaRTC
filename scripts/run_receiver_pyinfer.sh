#!/bin/bash
set -e

ALPHARTC_PATH="/opt/home_dir/AlphaRTC/out/Default"
ALPHARTC_SCRIPTS="/opt/home_dir/AlphaRTC/scripts"
SETUP_DELAY=3

RECEIVER_CONFIG=/opt/home_dir/AlphaRTC/configs/receiver.json

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
# For receiver
mkfifo /tmp/cpp_to_py_receiver
mkfifo /tmp/py_to_cpp_receiver

# Start the Python scripts first (they'll wait for input)
python3 ${ALPHARTC_SCRIPTS}/cmdinfer.py < /tmp/cpp_to_py_receiver > /tmp/py_to_cpp_receiver 2>${OUTPUT_DIR}/receiver_py.log &
RECEIVER_PY_PID=$!

# Start Receiver with bidirectional piping
(${ALPHARTC_PATH}/peerconnection_serverless ${RECEIVER_CONFIG} > /tmp/cpp_to_py_receiver < /tmp/py_to_cpp_receiver) 2>${OUTPUT_DIR}/receiver.log