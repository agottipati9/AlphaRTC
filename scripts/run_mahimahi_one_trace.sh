#!/bin/bash
set -e

ALPHARTC_PATH="/opt/home_dir/AlphaRTC/out/Default"
CALL_DURATION=120
CLEANUP_DELAY=10
SETUP_DELAY=3

# cmd args how to run: ./run_mahimahi_one_trace.sh --trace <trace_file> --receiver_config <receiver_config> --sender_config <sender_config> --output_dir <output_dir>
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
}

trap cleanup EXIT SIGINT SIGTERM

delay=${DELAY}
up_pkt_loss=0
down_pkt_loss=0

# Start Receiver
${ALPHARTC_PATH}/peerconnection_serverless ${RECEIVER_CONFIG} 2> /dev/null &
sleep "${SETUP_DELAY}"
# Start Sender
(mm-delay ${delay} mm-loss uplink ${up_pkt_loss} mm-loss downlink ${down_pkt_loss} \
 mm-link ${TRACE_FILE} ${TRACE_FILE} -- \
 bash -c "sed -i 's/\"dest_ip\": \"REPLACE ME\"/\"dest_ip\": \"'\$MAHIMAHI_BASE'\"/g' ${SENDER_CONFIG} && \
 ${ALPHARTC_PATH}/peerconnection_serverless ${SENDER_CONFIG} > /dev/null 2>&1" \
) &
# wait for clean up
sleep "${CALL_DURATION}"  # let the call run for 2 minutes
cleanup
sleep "${CLEANUP_DELAY}"  # wait for everything to close before starting the next one

# convert output video to mp4, delete the yuv file, and artifacts
cd ${OUTPUT_DIR}
ffmpeg -f yuv4mpegpipe -i ./receiver_unlimited-60s.yuv -r 30 -c:v libx264 -preset slow -crf 18 receiver_output.mp4
rm ./receiver_unlimited-60s.yuv
rm ./receiver_outaudio.wav
ffmpeg -f yuv4mpegpipe -i ./sender_unlimited-60s.yuv -r 30 -c:v libx264 -preset slow -crf 18 sender_output.mp4
rm ./sender_unlimited-60s.yuv
rm ./sender_outaudio.wav