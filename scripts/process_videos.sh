#!/bin/bash
set -e

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --results_dir)
    RESULTS_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

SENDER_DEG_VIDEO=${RESULTS_DIR}/sender_output.mp4
RECEIVER_DEG_VIDEO=${RESULTS_DIR}/receiver_output.mp4
TMP_DIR=${RESULTS_DIR}/tmp

function cleanup {
    echo "Cleaning up"
    cd /opt/home_dir/Video_Call_MOS/
    if [ -d ${TMP_DIR} ]; then
        rm -rf ${TMP_DIR}/*
    fi
    rm ${SENDER_DEG_VIDEO} ${RECEIVER_DEG_VIDEO}
}


trap cleanup EXIT SIGINT SIGTERM

# if the tmp directory does not exist, create it
if [ ! -d ${TMP_DIR} ]; then
    mkdir -p ${TMP_DIR}
fi

# process sender
cd /opt/home_dir/Video_Call_MOS/
python /opt/home_dir/Video_Call_MOS/run_video_call_mos.py --deg_video ${SENDER_DEG_VIDEO} --ref_video /opt/home_dir/sample_media/GTAV_qr.mp4 --results_dir ${RESULTS_DIR} --tmp_dir ${TMP_DIR}

# cleanup
if [ -d ${TMP_DIR} ]; then
    rm -rf ${TMP_DIR}/*
fi
rm ${SENDER_DEG_VIDEO}

# process receiver
cd /opt/home_dir/Video_Call_MOS/
python /opt/home_dir/Video_Call_MOS/run_video_call_mos.py --deg_video ${RECEIVER_DEG_VIDEO} --ref_video /opt/home_dir/sample_media/GTAV_qr.mp4 --results_dir ${RESULTS_DIR} --tmp_dir ${TMP_DIR}

# cleanup
if [ -d ${TMP_DIR} ]; then
    rm -rf ${TMP_DIR}/*
fi
rm ${RECEIVER_DEG_VIDEO}



