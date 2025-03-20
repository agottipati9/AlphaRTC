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
            if file.endswith('.pkl') and ('sender' in file or 'receiver' in file):
                call_metric_paths.append(os.path.join(root, file))
    return sorted(call_metric_paths)  # ensure paths are sorted

def process_meta_trajectory(path):
    with open(path, 'rb') as f:
        meta_trajectory = pickle.load(f)
    return meta_trajectory

def process_data(paths):
    data = {}
    for path in tqdm(paths):
        path_key = '/'.join(path.split('/')[:-1])
        if data.get(path_key, None) is None:
            data[path_key] = {}
        meta_trajectory = process_meta_trajectory(path)
        dict_key = 'sender_data' if 'sender' in path else 'receiver_data'
        data[path_key][dict_key] = meta_trajectory
    # save data to a file
    with open('/mydata/outputs/meta_trajectories.pkl', 'wb') as f:
        pickle.dump(data, f)

paths = get_all_call_metric_paths('/mydata/outputs/test_artifacts/')
process_data(paths)


