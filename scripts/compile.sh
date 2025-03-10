#!/bin/bash

cd /opt/home_dir/AlphaRTC
gn gen out/Default
ninja -C out/Default peerconnection_serverless