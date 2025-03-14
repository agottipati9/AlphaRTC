import os
import json
import pickle
import re
from tqdm import tqdm

import pandas as pd

def get_all_call_metric_paths(path):
    call_metric_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".log") or file.endswith(".csv"):
                call_metric_paths.append(os.path.join(root, file))
    return sorted(call_metric_paths)  # ensure paths are sorted

def process_log_file(path):
    state_info = {}
    
    with open(path, 'r') as f:
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
            
            # Parse array metrics
            if ":" in content and "[" in content and "]" in content:
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
    return state_info

def process_csv_file(path):
    mos_df = pd.read_csv(path)
    mos_values = mos_df["video_call_mos"].values
    return list(mos_values)

def process_data(paths):
    data = {}
    for path in tqdm(paths):
        path_key = '/'.join(path.split('/')[:-1])
        if data.get(path_key, None) is None:
            data[path_key] = {}
        if path.endswith('.log'):
            log_data = process_log_file(path)
            dict_key = 'sender_data' if 'sender' in path else 'receiver_data'
            data[path_key][dict_key] = log_data
        elif path.endswith('.csv'):
            mos_values = process_csv_file(path)
            dict_key = 'sender_mos' if 'sender' in path else 'receiver_mos'
            data[path_key][dict_key] = mos_values
    # save data to a file
    with open('/mydata/gcc_baselines/data.pkl', 'wb') as f:
        pickle.dump(data, f)

paths = get_all_call_metric_paths('/mydata/gcc_baselines/test_artifacts/')
process_data(paths)


