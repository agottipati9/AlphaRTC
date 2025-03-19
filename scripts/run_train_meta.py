import os
import sys
import glob
import json
import time
import subprocess
import random

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return_code = process.returncode
    return out, err, return_code

def load_traces(trace_path):
    traces = glob.glob(os.path.join(trace_path, "*"))
    traces = sorted(traces)  # ensure traces are sorted
    return traces

def set_configuration_files(proc_id, results_path, port=5000):
    # set configuration file for sender and receiver
    sender_config = json.load(open("/opt/home_dir/AlphaRTC/configs/sender_template.json"))
    receiver_config = json.load(open("/opt/home_dir/AlphaRTC/configs/receiver_template.json"))

    # update log output paths and port numbers
    sender_config["serverless_connection"]["sender"]["dest_port"] = port
    sender_config["save_to_file"]["audio"]["file_path"] = results_path + "sender_outaudio.wav"
    sender_config["save_to_file"]["video"]["file_path"] = results_path + "sender_unlimited-60s.yuv"
    sender_config["logging"]["log_output_path"] = results_path + "sender.log"

    receiver_config["serverless_connection"]["receiver"]["listening_port"] = port
    receiver_config["save_to_file"]["audio"]["file_path"] = results_path + "receiver_outaudio.wav"
    receiver_config["save_to_file"]["video"]["file_path"] = results_path + "receiver_unlimited-60s.yuv"
    receiver_config["logging"]["log_output_path"] = results_path + "receiver.log"

    # save configuration files
    CONFIG_DIR = "/opt/home_dir/AlphaRTC/configs/"
    sender_config = json.dump(sender_config, open(CONFIG_DIR + f"tmp_sender_config_proc_{proc_id}.json", "w"))
    receiver_config = json.dump(receiver_config, open(CONFIG_DIR + f"tmp_receiver_config_proc_{proc_id}.json", "w"))
    return CONFIG_DIR + f"tmp_sender_config_proc_{proc_id}.json", CONFIG_DIR + f"tmp_receiver_config_proc_{proc_id}.json"

def setup():
    # compile code
    cmd = "/opt/home_dir/AlphaRTC/scripts/compile.sh"
    run_command(cmd)
    # reset meta model
    cmd = "cp /opt/home_dir/AlphaRTC/scripts/online-train/ppo/checkpoints/initial_meta.pth /opt/home_dir/AlphaRTC/scripts/meta_model/meta.pth"
    run_command(cmd)
    # reset output directory
    cmd = "mkdir -p /mydata/outputs/test_artifacts/"
    run_command(cmd)
    cmd = "rm -rf /mydata/outputs/test_artifacts/*"
    run_command(cmd)
    # reset online train log dir
    cmd = "mkdir -p /mydata/online_train/"
    run_command(cmd)
    cmd = "rm -rf /mydata/online_train/*"
    run_command(cmd)
    # create meta_trajectories dir if it does not exist
    cmd = "mkdir -p /mydata/meta_trajectories/"
    run_command(cmd)

def main():
    # PARAMETERS
    # - trace_path: path to the directory containing traces
    # - n_runs: number of times to run each trace
    # load traces
    trace_path = "/opt/home_dir/network_traces/traces/train/"
    # trace_path = "/opt/home_dir/toy_trace/"
    # trace_path = "/opt/home_dir/network_traces/debug/"
    traces = load_traces(trace_path)
    n_runs = 10

    # clear directories and compile code
    setup()
    
    # create unique folder path to store all results
    proc_id = int(time.time())
    for i in range(0, n_runs):
        print(f"Epoch {i+1}")
        for j, trace_file in enumerate(traces):
            print(f"Step {j+1}, Running trace: {trace_file}")
            # create output directory
            trace_name = os.path.basename(trace_file) # without extension
            if '.' in trace_file:
                trace_name = trace_name.split('.')[0]
            results_path = f"/mydata/outputs/test_artifacts/RESULTS_RUN_{i}_STEP_{j}_TRACE_{trace_name}_ID_{proc_id}/"
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            # update log output paths and port numbers
            port = 5000 + (i * len(traces) + j)
            sender_path, receiver_path = set_configuration_files(proc_id, results_path, port=port)
            # sample delay
            delay = random.randint(40, 61)
            # run test script
            test_script = f"bash /opt/home_dir/AlphaRTC/scripts/run_mahimahi_one_trace_pyinfer.sh --trace {trace_file} --receiver_config {receiver_path} --sender_config {sender_path} --output_dir {results_path} --delay {delay}"
            run_command(test_script)
            # post processing
            post_processing_cmd = f"python /opt/home_dir/AlphaRTC/scripts/process_logs.py --output_dir {results_path}"
            run_command(post_processing_cmd)
            # process meta trajectories
            meta_processing_cmd = f"python /opt/home_dir/AlphaRTC/scripts/online-train/process_meta_output_logs.py --path {results_path} --output /mydata/online_train/"
            run_command(meta_processing_cmd)
            # # train agent with meta trajectories
            train_meta_cmd = f"python /opt/home_dir/AlphaRTC/scripts/online-train/train.py --epoch {i * len(traces) + j} --traj_path /mydata/online_train/meta_trajectories.pkl"
            out, _, __ = run_command(train_meta_cmd)
            print(out)
            # remove trajectory files after update to avoid training on stale data            
            os.remove(f"/mydata/online_train/meta_trajectories.pkl")
            # delete configuration files
            os.remove(sender_path)
            os.remove(receiver_path)
            # wait for 5 seconds for FDs to close
            time.sleep(5)

if __name__ == '__main__':
    main()