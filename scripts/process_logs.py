#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import numpy as np
import sys
import subprocess
import re

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
        self.net_states = None

        self.parse_net_log()
        

    def parse_net_log(self):
        if not self.net_path or not os.path.exists(self.net_path):
            raise ValueError("Error net path")

        json_data = []
        current_packet = None
        
        with open(self.net_path, 'r') as f:
            for line in f.readlines():
                net_states = {}
                if "remote_estimator_proxy.cc" not in line:
                    continue
                
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
                        # We'll preserve mediaInfo but mark it as parsed
                        current_packet = json_obj
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
                            
                            # Store in net_states
                            net_states[metric_name] = values

                            # Also attach to the current packet if it exists
                            if current_packet is not None:
                                if "state_metrics" not in current_packet:
                                    current_packet["state_metrics"] = {}
                                current_packet["state_metrics"][metric_name] = values
                    except Exception as e:
                        print(f"Warning: Failed to parse metric line: {content}, error: {str(e)}")
        self.net_data = json_data


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

    # scale delay list
    # for ssrc in ssrc_info:
    #     min_delay = min(ssrc_info[ssrc]["delay_list"])
    #     # ssrc_info[ssrc]["scale_delay_list"] = [
    #     #     min(max_delay, delay) for delay in ssrc_info[ssrc]["delay_list"]]
    #     delay_pencentile_95 = np.percentile(
    #         ssrc_info[ssrc]["scale_delay_list"], 95)
    #     ssrc_info[ssrc]["delay_score"] = (
    #         max_delay - delay_pencentile_95) / (max_delay - min_delay)
    #     print("Queue Delay_min (ms): {}".format(min_delay))
    #     print("Queue Delay_95 (ms): {}".format(delay_pencentile_95))
    #     print("Queue Delay_max (ms): {}".format(max_delay))
    #     print("Receive Rate (KB/s): {}".format(
    #         ssrc_info[ssrc]["avg_recv_rate"]))
    #     print("Ground Truth Receive Rate (KB/s): {}".format(ground_recv_rate))
    #     print("Loss Count: {}".format(loss_count))

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
    return dst_net_data, src_net_data


def get_video_score(args):
    # print("----- Frame Statistics -----")

    # Frame Delay Score
    # delays = []
    # with open(args.receiver_log, 'r') as receiver_log:
    #     for line in receiver_log:
    #         if 'E2E FRAME DELAY' in line:
    #             delays.append(int(line.split()[-1]))

    # print("Frame delay measurement count:", len(delays))
    # if len(delays) == 0:
    #     delays = [0]
    # print("Mean per-frame delay (ms):", np.mean(delays))
    # print("Median per-frame delay (ms):", np.median(delays))
    # print("P95 per-frame delay (ms):", np.percentile(delays, 95))

    # frame_delay_score = max(100 - np.mean(delays) / 3, 0)
    # print("")

    # Quality Score (VMOS)
    out, err, ret = run_command(f"/opt/home_dir/AlphaRTC/scripts/process_videos.sh --results_dir {args.output_dir}")

    # # Drop frames from sender log
    # frame_numbers=$(cat $log_file | grep "Framedropped for reason" | awk '{print "eq(n,"$8")"}' | paste -sd "+")
    # frame_numbers = subprocess.check_output('cat ' + args.sender_log +
    #                                         ' | grep "Framedropped for reason" | awk \'{print "eq(n,"$8")"}\' | paste -sd "+"', shell=True).decode('utf-8').strip()

    # # ffmpeg -v error -i $source -vf "select='not($frame_numbers)',setpts=N/FRAME_RATE/TB" -f yuv4mpegpipe -pix_fmt yuv420p source_dropped1.yuv
    # print("Dropping {} frames from sender log...".format(
    #     frame_numbers.count('eq')))
    # if frame_numbers.count('eq') > 0:
    #     subprocess.check_call(
    #         'ffmpeg -v error -i {} -vf "select=\'not({})\',setpts=N/FRAME_RATE/TB" -f yuv4mpegpipe -pix_fmt yuv420p ./source_dropped1.yuv'.format(
    #             args.sender_video, frame_numbers),
    #         shell=True)
    # else:
    #     subprocess.check_call('cp {} ./source_dropped1.yuv'.format(
    #         args.sender_video), shell=True)

    # # Drop frames from receiver log
    # frame_numbers = subprocess.check_output('cat ' + args.receiver_log +
    #                                         ' | grep "Framedropped:" | awk \'{print "eq(n,"$5")"}\' | paste -sd "+"', shell=True).decode('utf-8').strip()

    # print("Dropping {} frames from receiver log...".format(
    #     frame_numbers.count('eq')))
    # if frame_numbers.count('eq') > 0:
    #     subprocess.check_call(
    #         'ffmpeg -v error -i ./source_dropped1.yuv -vf "select=\'not({})\',setpts=N/FRAME_RATE/TB" -f yuv4mpegpipe -pix_fmt yuv420p ./source_dropped2.yuv'.format(
    #             frame_numbers),
    #         shell=True)
    # else:
    #     subprocess.check_call('cp ./source_dropped1.yuv ./source_dropped2.yuv',
    #                           shell=True)

    # # ./vmaf -r ./source_dropped2.yuv -d $target -o $vmaf_output --json --threads $threads
    # subprocess.check_call(
    #     '{} -r source_dropped2.yuv -d {} -o {} --json --threads {}'.format(
    #         args.vmaf, args.receiver_video, args.vmaf_output, args.threads),
    #     shell=True)
    # vmaf_json = json.load(open('vmaf.json', 'r'))
    # vmaf_score = vmaf_json['pooled_metrics']['vmaf']['mean']

    # # rm ./source_dropped1.yuv
    # # rm ./source_dropped2.yuv
    # try:
    #     subprocess.call('rm ./source_dropped1.yuv', shell=True)
    # except:
    #     pass
    # try:
    #     subprocess.call('rm ./source_dropped2.yuv', shell=True)
    # except:
    #     pass
    # print("")

    # # Frame Drop Score
    # read_frames = []
    # with open(args.sender_log, 'r') as sender_log:
    #     for line in sender_log:
    #         if 'FRAME READ' in line:
    #             read_frames.append(int(line.split()[-1]))

    # write_frames = []
    # with open(args.receiver_log, 'r') as receiver_log:
    #     for line in receiver_log:
    #         if 'FRAME WRITE' in line:
    #             write_frames.append(int(line.split()[-1]))

    # print("Frame Drop Rate: {:.2f}% ({}/{})".format(
    #     (1 - len(write_frames) / len(read_frames)) * 100, len(read_frames) - len(write_frames), len(read_frames)))

    # frame_drop_score = 100 * len(write_frames) / len(read_frames)
    # print("")

    # video_score = 0.2 * frame_delay_score + \
    #     0.2 * vmaf_score + 0.3 * frame_drop_score

    # print("Frame Delay Score: {:.2f}".format(frame_delay_score))
    # print("VMAF Score: {:.2f}".format(float(vmaf_score)))
    # print("Frame Drop Score: {:.2f}".format(frame_drop_score))
    # print("Video Score: {:.2f}".format(video_score))
    # print("")

    if out:
        print(out.decode('utf-8'))
    if err:
        print(err.decode('utf-8'))

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
    out_dict["receiver_packet_info"] = receiver_packet_info
    out_dict["sender_packet_info"] = sender_packet_info

    out_path = os.path.join(args.output_dir, "call_metrics.json")
    with open(out_path, 'w') as f:
        f.write(json.dumps(out_dict))
    

    print("Processed call logs.")
    print("Output written to", out_path)
    print("")

    print("Computing Video MOS...")
    get_video_score(args)
    print("Video evaluation completed.")