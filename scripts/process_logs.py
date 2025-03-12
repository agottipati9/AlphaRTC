#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import numpy as np
import sys
import subprocess
import re

import pandas as pd

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return_code = process.returncode
    return out, err, return_code

class NetInfo(object):
    def __init__(self, net_path):
        self.net_path = net_path
        self.net_data = None

        # TODO: we need to parse the state info we just added to logs
        self.state_info = None
        self.frame_timestamps = None
        self.interpolated_mos = None
        self.aligned_packet_timestamps = None
        self.all_data = None

        self.parse_net_log()
        

    def parse_net_log(self):
        if not self.net_path or not os.path.exists(self.net_path):
            raise ValueError("Error net path")

        json_data = []
        frame_timestamps = []
        state_info = {}
        aligned_packet_timestamps = []
        
        with open(self.net_path, 'r') as f:
            for line in f.readlines():
                if "remote_estimator_proxy.cc" in line:
                
                    # Extract the line number and content
                    match = re.search(r'\(remote_estimator_proxy\.cc:(\d+)\):\s*(.*)', line)
                    if not match:
                        continue
                    
                    line_num = match.group(1)
                    content = match.group(2).strip()
                    
                    # Try to parse as JSON (packet info)
                    if content.startswith('{'):
                        try:
                            json_obj = json.loads(content)
                            aligned_packet_timestamps.append(json_obj["packetInfo"]["arrivalTimeMs"])
                            # We'll preserve mediaInfo but mark it as parsed
                            json_data.append(json_obj)
                        except ValueError:
                            pass
                        except Exception as e:
                            raise ValueError(f"Exception when parsing JSON log: {str(e)}")
                    
                    # Parse array metrics
                    elif ":" in content and "[" in content and "]" in content:
                        try:
                            metric_name, values_str = content.split(':', 1)
                            metric_name = metric_name.strip()
                            
                            # Extract values from array format [x, y, z]
                            values_str = values_str.strip()
                            if values_str.startswith('[') and values_str.endswith(']'):
                                values_str = values_str[1:-1].strip()
                                # Convert values to appropriate type (float or int)
                                values = []
                                for val in values_str.split(','):
                                    val = val.strip()
                                    if not val:
                                        continue
                                    try:
                                        # Try to convert to float first
                                        if '.' in val:
                                            values.append(float(val))
                                        else:
                                            values.append(int(val))
                                    except ValueError:
                                        # Keep as string if conversion fails
                                        values.append(val)
                                
                                if metric_name not in state_info:
                                    state_info[metric_name] = []
                                state_info[metric_name].append(values)
                        except Exception as e:
                            print(f"Warning: Failed to parse metric line: {content}, error: {str(e)}")
                elif "frame_generator.cc" in line and "FRAME READ:" in line:
                    # (frame_generator.cc:253): FRAME READ: 1333915904572
                    match = re.search(r'\(frame_generator\.cc:(\d+)\):\s*FRAME READ:\s*(\d+)', line)
                    if match:
                        frame_timestamps.append(int(match.group(2)) / 1000)  # convert microseconds to milliseconds

        self.net_data = json_data
        self.state_info = state_info
        self.frame_timestamps = frame_timestamps
        # align packet timestamps to the first packet
        aligned_packet_timestamps = sorted(aligned_packet_timestamps)
        aligned_packet_timestamps = [ts - aligned_packet_timestamps[0] for ts in aligned_packet_timestamps]
        self.aligned_packet_timestamps = aligned_packet_timestamps
        self.all_data = {
            "net_data": json_data,
            "state_info": state_info,
            "frame_timestamps": frame_timestamps,
            "aligned_packet_timestamps": aligned_packet_timestamps
        }


def eval_network(dst_audio_info: NetInfo):
    net_data = dst_audio_info.net_data
    ssrc_info = {}

    delay_list = []
    loss_count = 0
    last_seqNo = {}
    for item in net_data:
        ssrc = item["packetInfo"]["header"]["ssrc"]
        sequence_number = item["packetInfo"]["header"]["sequenceNumber"]
        tmp_delay = item["packetInfo"]["arrivalTimeMs"] - \
            item["packetInfo"]["header"]["sendTimestamp"]
        if (ssrc not in ssrc_info):
            ssrc_info[ssrc] = {
                "time_delta": -tmp_delay,
                "delay_list": [],
                "received_nbytes": 0,
                "start_recv_time": item["packetInfo"]["arrivalTimeMs"],
                "avg_recv_rate": 0
            }
        if ssrc in last_seqNo:
            loss_count += max(0, sequence_number -
                              last_seqNo[ssrc] - 1)
        last_seqNo[ssrc] = sequence_number

        ssrc_info[ssrc]["delay_list"].append(
            ssrc_info[ssrc]["time_delta"] + tmp_delay)
        ssrc_info[ssrc]["received_nbytes"] += item["packetInfo"]["payloadSize"]
        if item["packetInfo"]["arrivalTimeMs"] != ssrc_info[ssrc]["start_recv_time"]:
            ssrc_info[ssrc]["avg_recv_rate"] = ssrc_info[ssrc]["received_nbytes"] / \
                (item["packetInfo"]["arrivalTimeMs"] -
                    ssrc_info[ssrc]["start_recv_time"])

    # filter short stream
    ssrc_info = {key: val for key,
                 val in ssrc_info.items() if len(val["delay_list"]) >= 10}

    # avg delay
    avg_delay = np.mean(
        [np.mean(ssrc_info[ssrc]["delay_list"]) for ssrc in ssrc_info])

    # receive rate score
    recv_rate_list = [ssrc_info[ssrc]["avg_recv_rate"]
                      for ssrc in ssrc_info if ssrc_info[ssrc]["avg_recv_rate"] > 0]
    # avg recv rate
    avg_recv_rate = np.mean(recv_rate_list)

    print("Avg Delay (ms): {}".format(avg_delay))
    print("Avg Receive Rate (KB/s): {}".format(avg_recv_rate))
    print("Loss Count: {}".format(loss_count))
    return net_data


def get_network_data(args):
    print("----- Network Statistics -----")
    dst_network_info = NetInfo(args.receiver_log)
    dst_net_data = eval_network(dst_network_info)
    src_network_info = NetInfo(args.sender_log)
    src_net_data = eval_network(src_network_info)
    print("")
    return dst_network_info.all_data, src_network_info.all_data


def get_video_score(args):
    # Quality Score (VMOS)
    out, err, ret = run_command(f"/opt/home_dir/AlphaRTC/scripts/process_videos.sh --results_dir {args.output_dir}")

    if out:
        print(out.decode('utf-8'))
    if err:
        print(err.decode('utf-8'))

def interpolate_video_score(output_dir, receiver_packet_info, sender_packet_info):
    # open receiver video MOS file
    receiver_mos_path = os.path.join(output_dir, "receiver_output.csv")
    receiver_mos_df = pd.read_csv(receiver_mos_path)
    receiver_packet_info = _interpolate_video_score(receiver_packet_info, receiver_mos_df)
    # open sender video MOS file
    sender_mos_path = os.path.join(output_dir, "sender_output.csv")
    sender_mos_df = pd.read_csv(sender_mos_path)
    sender_packet_info = _interpolate_video_score(sender_packet_info, sender_mos_df)
    return receiver_packet_info, sender_packet_info


def _interpolate_video_score(packet_info, mos_df):
    # align frame timestamps to the start of the video (treat first frame as 0)
    # video_start_time_in_ms = 12000  # 12 seconds (video packets start at 12 seconds)  
    frame_timestamps = packet_info['frame_timestamps']
    frame_timestamps = [ts - frame_timestamps[0] for ts in frame_timestamps]
    # frame_timestamps = [0] + frame_timestamps
    packet_timestamps = packet_info['aligned_packet_timestamps']
    mos_values = mos_df["video_call_mos"].values
    # Interpolate MOS values to frame timestamps -- evenly distribute MOS values across frames
    num_frames = len(frame_timestamps)
    num_mos_values = len(mos_values)
    mos_indices = np.linspace(0, num_frames - 1, num_mos_values)
    mos_timestamps = np.interp(mos_indices, np.arange(num_frames), frame_timestamps)
    # interpolate video MOS values to packet timestamps
    interpolated_mos_packet_level = np.interp(packet_timestamps, mos_timestamps, mos_values)
    packet_info['interpolated_mos_packet_level'] = interpolated_mos_packet_level.tolist()
    # interpolate video MOS values to every 50 ms (inference interval)
    inference_interval = 50
    inference_timestamps = np.arange(0, frame_timestamps[-1], inference_interval)
    interpolated_mos_inference_level = np.interp(inference_timestamps, mos_timestamps, mos_values)
    packet_info['interpolated_mos_inference_level'] = interpolated_mos_inference_level.tolist()
    return packet_info

def init_network_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="path to output artifacts.")
    parser.add_argument("--sender_log", type=str,
                        default=None, help="the path of sender log.")
    parser.add_argument("--receiver_log", type=str,
                        default=None, help="the path of receiver log.")
    return parser


if __name__ == "__main__":
    parser = init_network_argparse()
    args = parser.parse_args()
    if args.sender_log is None:
        args.sender_log = os.path.join(args.output_dir, "sender.log")
    if args.receiver_log is None:
        args.receiver_log = os.path.join(args.output_dir, "receiver.log")

    out_dict = {}
    receiver_packet_info, sender_packet_info = get_network_data(args)

    print("Computing Video MOS...")
    get_video_score(args)
    print("Video evaluation completed.")

    receiver_packet_info, sender_packet_info = interpolate_video_score(args.output_dir, receiver_packet_info, sender_packet_info)

    out_dict["receiver_packet_info"] = receiver_packet_info
    out_dict["sender_packet_info"] = sender_packet_info
    out_path = os.path.join(args.output_dir, "call_metrics.json")
    with open(out_path, 'w') as f:
        f.write(json.dumps(out_dict))

    print("Processed call logs.")
    print("Output written to", out_path)
    print("")