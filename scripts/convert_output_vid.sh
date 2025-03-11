#!/bin/bash

cd /opt/home_dir/outputs
ffmpeg -f yuv4mpegpipe -i ./receiver_unlimited-60s.yuv -c:v libx264 -preset slow -crf 18 receiver_output.mp4
