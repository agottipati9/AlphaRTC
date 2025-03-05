#!/bin/bash
set -e

ALPHARTC_PATH="/opt/home_dir/AlphaRTC/out/Default"
CALL_DURATION=60
CLEANUP_DELAY=10
SETUP_DELAY=3

delay=60
up_pkt_loss=0
down_pkt_loss=0

SENDER_CONFIG=/opt/home_dir/AlphaRTC/configs/sender_gcc_local.json
RECEIVER_CONFIG=/opt/home_dir/AlphaRTC/configs/receiver_gcc.json
TRACE_FILE=/opt/home_dir/toy_trace/starlink_trace_0.log

function cleanup {
    echo "Cleaning up"
    pkill -f peerconnection_gcc || true
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

echo "Running the test call"

# Start Receiver
${ALPHARTC_PATH}/peerconnection_gcc ${RECEIVER_CONFIG} 2> /dev/null &
sleep "${SETUP_DELAY}"
# Start Sender
${ALPHARTC_PATH}/peerconnection_gcc ${SENDER_CONFIG} 2> /dev/null &
# wait for clean up
sleep "${CALL_DURATION}"  # let the call run for 2 minutes
cleanup
sleep "${CLEANUP_DELAY}"  # wait for everything to close before starting the next one