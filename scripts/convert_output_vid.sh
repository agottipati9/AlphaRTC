#!/bin/bash

cd /opt/home_dir/outputs
ffmpeg -f yuv4mpegpipe -i receiver_unlimited-60s.yuv -r 30 -c:v libx264 -preset slow -crf 18 output.mp4