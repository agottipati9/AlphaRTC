#!/bin/bash

cd /opt/home_dir/outputs
ffmpeg -f rawvideo -vcodec rawvideo -s 1920x1080 -r 30 -pix_fmt yuv420p -i receiver_unlimited-60s.yuv -c:v libx264 -preset slow -crf 18 output.mp4
