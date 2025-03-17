import os
import json
import pickle
import re
from tqdm import tqdm
import argparse

import pandas as pd

def get_all_call_metric_paths(args):
    path = args.path
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

def process_data(paths, args):
    data = {}
    for path in tqdm(paths):
        path_key = '/'.join(path.split('/')[:-1])
        if data.get(path_key, None) is None:
            data[path_key] = {}
        meta_trajectory = process_meta_trajectory(path)
        dict_key = 'sender_data' if 'sender' in path else 'receiver_data'
        data[path_key][dict_key] = meta_trajectory
    # save data to a file
    output_file = args.output + 'meta_trajectories.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    # remove all files in the path
    # for path in paths:
    #     os.remove(path)

def parse_args():
    # add cmd line argument for path
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/mydata/outputs/test_artifacts/')
    parser.add_argument('--output', type=str, default='/mydata/online_train/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    paths = get_all_call_metric_paths(args)
    process_data(paths, args)


