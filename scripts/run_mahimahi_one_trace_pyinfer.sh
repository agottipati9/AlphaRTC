#!/bin/bash
set -e

ALPHARTC_PATH="/opt/home_dir/AlphaRTC/out/Default"
ALPHARTC_SCRIPTS="/opt/home_dir/AlphaRTC/scripts"
TRAJECTORY_LOGGING_PATH="/mydata/meta_trajectories"
CALL_DURATION=120
CLEANUP_DELAY=10
SETUP_DELAY=3

# cmd args how to run: ./run_mahimahi_one_trace_pyinfer.sh --trace <trace_file> --receiver_config <receiver_config> --sender_config <sender_config> --output_dir <output_dir>
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --trace)
    TRACE_FILE="$2"
    shift # past argument
    shift # past value
    ;;
    --receiver_config)
    RECEIVER_CONFIG="$2"
    shift # past argument
    shift # past value
    ;;
    --sender_config)
    SENDER_CONFIG="$2"
    shift # past argument
    shift # past value
    ;;
    --output_dir)
    OUTPUT_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    --delay)
    DELAY="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

function cleanup {
    echo "Cleaning up"
    pkill -f peerconnection_serverless || true
    pkill -f cmdinfer.py || true
    # Clean up the named pipes
    rm -f /tmp/cpp_to_py_* /tmp/py_to_cpp_* 2>/dev/null || true
}

trap cleanup EXIT SIGINT SIGTERM

delay=${DELAY:-60}
up_pkt_loss=0
down_pkt_loss=0

# make sure the output directories exist
mkdir -p ${OUTPUT_DIR}
mkdir -p ${TRAJECTORY_LOGGING_PATH}

# Set up Python path
export PYTHONPATH="${ALPHARTC_SCRIPTS}:${PYTHONPATH}"

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

# Start Sender with bidirectional piping inside the mahimahi environment
(mm-delay ${delay} mm-loss uplink ${up_pkt_loss} mm-loss downlink ${down_pkt_loss} \
 mm-link ${TRACE_FILE} ${TRACE_FILE} -- \
 bash -c "sed -i 's/\"dest_ip\": \"REPLACE ME\"/\"dest_ip\": \"'\$MAHIMAHI_BASE'\"/g' ${SENDER_CONFIG} && \
 export LD_LIBRARY_PATH=/opt/home_dir/dll && \
 ${ALPHARTC_PATH}/peerconnection_serverless ${SENDER_CONFIG} > /tmp/cpp_to_py_sender < /tmp/py_to_cpp_sender 2>${OUTPUT_DIR}/sender.log" \
) &

# wait for clean up
sleep "${CALL_DURATION}"  # let the call run for the specified duration
cleanup
sleep "${CLEANUP_DELAY}"  # wait for everything to close before starting the next one

# # convert output video to mp4, delete the yuv file, and artifacts
cd ${OUTPUT_DIR}
ffmpeg -f yuv4mpegpipe -i ./receiver_unlimited-60s.yuv -r 30 -c:v libx264 -preset slow -crf 18 receiver_output.mp4
rm ./receiver_unlimited-60s.yuv
rm ./receiver_outaudio.wav
ffmpeg -f yuv4mpegpipe -i ./sender_unlimited-60s.yuv -r 30 -c:v libx264 -preset slow -crf 18 sender_output.mp4
rm ./sender_unlimited-60s.yuv
rm ./sender_outaudio.wav