#!/bin/bash
set -e

ALPHARTC_PATH="/opt/home_dir/AlphaRTC/out/Default"
ALPHARTC_SCRIPTS="/opt/home_dir/AlphaRTC/scripts"
CALL_DURATION=120
CLEANUP_DELAY=10
SETUP_DELAY=3

delay=60
up_pkt_loss=0
down_pkt_loss=0

SENDER_CONFIG=/opt/home_dir/AlphaRTC/configs/sender.json
RECEIVER_CONFIG=/opt/home_dir/AlphaRTC/configs/receiver.json
TRACE_FILE=/opt/home_dir/toy_trace/starlink_trace_0.log

function cleanup {
    echo "Cleaning up"
    pkill -f peerconnection_serverless || true
    pkill -f cmdinfer.py || true
    # Clean up the named pipes
    rm -f /tmp/cpp_to_py_* /tmp/py_to_cpp_* 2>/dev/null || true
}

trap cleanup EXIT SIGINT SIGTERM

OUTPUT_DIR=/opt/home_dir/outputs/

# compile the code
echo "Compiling the code"
bash /opt/home_dir/AlphaRTC/scripts/compile.sh

# if compile fails, exit
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

echo "Clearing the output directory"
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

# Set up Python path
export PYTHONPATH="${ALPHARTC_SCRIPTS}:${PYTHONPATH}"

echo "Running the test call"

# Create named pipes for bidirectional communication
# For receiver
mkfifo /tmp/cpp_to_py_receiver
mkfifo /tmp/py_to_cpp_receiver

# For sender
mkfifo /tmp/cpp_to_py_sender
mkfifo /tmp/py_to_cpp_sender

# Start the Python scripts first (they'll wait for input)
python3 ${ALPHARTC_SCRIPTS}/cmdinfer.py < /tmp/cpp_to_py_receiver > /tmp/py_to_cpp_receiver 2>${OUTPUT_DIR}/receiver_py.log &
RECEIVER_PY_PID=$!

python3 ${ALPHARTC_SCRIPTS}/cmdinfer.py < /tmp/cpp_to_py_sender > /tmp/py_to_cpp_sender 2>${OUTPUT_DIR}/sender_py.log &
SENDER_PY_PID=$!

# Start Receiver with bidirectional piping
(${ALPHARTC_PATH}/peerconnection_serverless ${RECEIVER_CONFIG} > /tmp/cpp_to_py_receiver < /tmp/py_to_cpp_receiver) 2>${OUTPUT_DIR}/receiver.log &
RECEIVER_PID=$!

sleep "${SETUP_DELAY}"

# Start Sender with bidirectional piping
(${ALPHARTC_PATH}/peerconnection_serverless ${SENDER_CONFIG} > /tmp/cpp_to_py_sender < /tmp/py_to_cpp_sender) 2>${OUTPUT_DIR}/sender.log &
SENDER_PID=$!

# wait for clean up
sleep "${CALL_DURATION}"  # let the call run for 2 minutes
cleanup
sleep "${CLEANUP_DELAY}"  # wait for everything to close before starting the next one