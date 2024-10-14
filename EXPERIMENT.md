# EXPERIMENT

## Build

1. Install [depot_tools](https://commondatastorage.googleapis.com/chrome-infra-docs/flat/depot_tools/docs/html/depot_tools_tutorial.html#_setting_up)
2. `sudo apt install python-is-python3` if don't have executable `python`
3. `make docker-sync` and wait tens of minutes

WebRTC defaultly uses VP8 as the video codec. You can modify `media/engine/internal_encoder_factory.cc` to select video codec. We modified it to use VP9 as the video codec by default.

### Build without H264

Without H264, we will use VP9 as the video codec.

We can directly `make docker-peerconnection` to build artifacts in `out/Default/` directory.

Or we can manually build artifacts like this:

1. `gn gen out/VP9 --args='is_debug=false'`
2. `ninja -C out/VP9 peerconnection_gcc`

Artifacts are in `out/VP9/` directory. We can use `peerconnection_gcc` to run with gcc.

`mkdir run` and `cp out/VP9/peerconnection_gcc run/ && cd run`, then run it.

## Run

### Prepare Media

#### Related Tools

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Download video from Youtube
- [ffmpeg](https://ffmpeg.org/) - Convert video to raw format

**Attention:** [YUView](https://github.com/IENT/YUView) cannot properly play YUV videos in experiments. It is recommended to use a player based on ffmpeg instead.

#### Example Steps

##### Our Experiment Example

``` bash
# Download video with frame number
wget https://media.xiph.org/video/derf/twitch/H264/GTAV.mp4

# Cut video to 60 seconds
ffmpeg -i GTAV.mp4 -ss 0 -t 60 -c copy GTAV-60.mp4

# Convert video to YUV format with 30fps frame rate
# ATTENTION: After this, the frame number will only have even numbers
ffmpeg -i GTAV-60.mp4 -filter:v fps=30 -f yuv4mpegpipe -pix_fmt yuv420p 1080p.yuv

# --- If you don't want to use audio, you can generate a silent audio file like this ---
# Generate a 1 second silent audio file.
ffmpeg -f lavfi -i anullsrc -t 1 -c:a pcm_s16le silent-1s.wav
```

**Attention:**

1. When converting the video to YUV format, the video format should be **yuv4mpegpipe**, and the pixel format should be **yuv420p**.
2. The configuration file below only supports **integer** FPS.

### Configure

Refer to [README](./README.md) for more details.

In our experiment, we use `receiver_gcc.json` and `sender_gcc.json` as configuration files.

Notes:

- Please change the `dest_ip` of the sender to the IP address of the receiver.
- `autoclose` could be set larger than video or audio duration. Video or audio will be repeated automatically.

`receiver_gcc.json`

``` json
{
  "serverless_connection": {
      "autoclose": 60,
      "sender": {
          "enabled": false
      },
      "receiver": {
          "enabled": true,
          "listening_ip": "0.0.0.0",
          "listening_port": 8000
      }
  },
  "bwe_feedback_duration": 200,
  "video_source": {
      "video_disabled": {
          "enabled": true
      },
      "webcam": {
          "enabled": false
      },
      "video_file": {
          "enabled": false
      }
  },
  "audio_source": {
      "microphone": {
          "enabled": false
      },
      "audio_file": {
          "enabled": true,
          "file_path": "./videos/silent-1s.wav"
      }
  },
  "save_to_file": {
      "enabled": true,
      "audio": {
          "file_path": "./outputs/outaudio.wav"
      },
      "video": {
          "width": 1920,
          "height": 1080,
          "fps": 30,
          "file_path": "./outputs/unlimited-60s.yuv"
      }
  },
  "logging": {
      "enabled": true,
      "log_output_path": "./outputs/receiver.log"
  }
}
```

`sender_gcc.json`

``` json
{
  "serverless_connection": {
      "autoclose": 60,
      "sender": {
          "enabled": true,
          "dest_ip": "100.64.0.1",
          "dest_port": 8000
      },
      "receiver": {
          "enabled": false
      }
  },
  "bwe_feedback_duration": 200,
  "video_source": {
      "video_disabled": {
          "enabled": false
      },
      "webcam": {
          "enabled": false
      },
      "video_file": {
          "enabled": true,
          "width": 1920,
          "height": 1080,
          "fps": 30,
          "file_path": "./videos/1080p.yuv"
      }
  },
  "audio_source": {
      "microphone": {
          "enabled": false
      },
      "audio_file": {
          "enabled": true,
          "file_path": "./videos/silent-1s.wav"
      }
  },
  "save_to_file": {
      "enabled": false
  },
  "logging": {
      "enabled": true,
      "log_output_path": "./outputs/sender.log"
  }
}
```

### Run GCC Experiment

Run receiver first, then run sender.

Receiver: `rm ./receiver.log; ./peerconnection_gcc receiver_gcc.json 2>receiver_warn.log`

Sender: `rm ./sender.log; ./peerconnection_gcc sender_gcc.json 2>sender_warn.log`

After 60 seconds, the program will exit automatically.

`outvideo.yuv` and `outaudio.wav` are the output files.

`receiver.log` and `sender.log` are the log files.

`receiver_warn.log` and `sender_warn.log` are the warning log files.

YUV video could be very large. If you want to play it, you can use `ffmpeg` to convert it to mp4 format like this: `ffmpeg -i outvideo.yuv outvideo.mp4`. Keep the YUV file if you want to do the evaluation.

For mahimahi, run the sender from within the mahimahi shell: 
``` bash
# Start Mahimahi shell
mm-delay 40 mm-loss uplink 0 mm-loss downlink 0 mm-link ../traces/Starlink/starlink_trace_14.log ../traces/12mbps

# Run sender from within Mahimahi shell
./peerconnection_gcc sender_gcc.json 2>sender_warn.log
```



## Evaluate

`evaluate.py` is a script to evaluate the QoE metrics, including network performance and video quality.

Network score comes from logs in `SENDER_LOG` and `RECEIVER_LOG`. You may need to set `--max_delay` and `--ground_recv_rate` accdording to your situation.

Video quality score includes [VMAF](https://github.com/Netflix/vmaf) and other metrics. Our script will align the frame according to the log file `SENDER_LOG` and use the binary specified by `--vmaf` to calculate the VMAF score. VMAF score will be in `--vmaf_output` file, defaultly `vmaf.json`.

Build VMAF binary from source code [libvmaf](https://github.com/Netflix/vmaf/blob/master/libvmaf/README.md) or download the pre-built binary from [VMAF Releases](https://github.com/Netflix/vmaf/releases).

QoE metrics refer to [PACC: Perception Aware Congestion Control for Real-time Communication](https://ieeexplore.ieee.org/document/10219915).

Our script refers to [Challenge-Environment](https://github.com/OpenNetLab/Challenge-Environment), [e2e_delay.py](https://github.com/OpenNetLab/AlphaRTC/blob/gcc_baseline/examples/peerconnection/gcc/corpus/e2e_delay.py) and [lossless_frame_delay.py](https://github.com/OpenNetLab/AlphaRTC/blob/gcc_baseline/examples/peerconnection/gcc/corpus/lossless_frame_delay.py).

### Usage

``` text
usage: evaluate.py [-h] [-o OUTPUT] -s SENDER_LOG -r RECEIVER_LOG [--max_delay MAX_DELAY] [--ground_recv_rate GROUND_RECV_RATE] --vmaf VMAF --sender_video SENDER_VIDEO --receiver_video RECEIVER_VIDEO
                   [--vmaf_output VMAF_OUTPUT] [--threads THREADS]

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        the path of output file. It will print the result in terminal if you don't specify its value.
  -s SENDER_LOG, --sender_log SENDER_LOG
                        the path of sender log.
  -r RECEIVER_LOG, --receiver_log RECEIVER_LOG
                        the path of receiver log.
  --max_delay MAX_DELAY
                        the max packet delay.
  --ground_recv_rate GROUND_RECV_RATE
                        the receive rate of a special scenario ground truth.
  --vmaf VMAF           the path of vmaf
  --sender_video SENDER_VIDEO
                        the path of sender video
  --receiver_video RECEIVER_VIDEO
                        the path of receiver video
  --vmaf_output VMAF_OUTPUT
                        the path to save vmaf json output
  --threads THREADS     the number of threads for vmaf
```

Example:

``` bash
./evaluate.py -r ./outputs/receiver.log -s ./outputs/sender.log --vmaf ./vmaf --sender_video ./videos/1080p.yuv --receiver_video ./outputs/unlimited-60s.yuv
```

### Notes

You should set correct `--max_delay` and `--ground_recv_rate` according to your situation.

You had better check the vmaf json output to make sure the vmaf score is correct.
