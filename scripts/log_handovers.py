import json
import time
import numpy as np
from pathlib import Path
import datetime
import cv2  # OpenCV for video writing
import sys

import argparse

# Import the Starlink gRPC client from your provided library

sys.path.insert(0,str(Path('/opt/home_dir/grpc/starlink-grpc-tools').resolve())) # /Users/silver/Desktop/alphartc_dockers/grpc/starlink-grpc-tools
import starlink_grpc

# --- Configuration ---
# The IP address of the Starlink dish.
DISH_IP = "192.168.100.1:9200"
# The minimum distance (in pixels) the centroid must jump to be considered a handover.
HANDOVER_JUMP_THRESHOLD = 5.0
# Video output settings
VIDEO_FILENAME = "obstruction_map_feed.mp4"
VIDEO_FPS = 20

# --- gRPC Helper Functions ---

def get_obstruction_map(context):
    """
    Fetches the obstruction map and returns it as a 123x123 numpy array.
    """
    try:
        resp = starlink_grpc.obstruction_map(context)
        if resp:
            snr = np.array(resp).reshape(123, 123)
            snr = (snr > 0).astype(int)
            return snr
    except Exception as e:
        print(f"Error getting obstruction map: {e}")
    return None

def clear_obstruction_map(context):
    """Sends a command to the dish to clear the current obstruction map."""
    try:
        print("Handover detected! Sending command to clear obstruction map...")
        starlink_grpc.reset_obstruction_map(context)
    except Exception as e:
        print(f"Error clearing obstruction map: {e}")

# --- Analysis Helper Functions ---

def xor_diff(current_frame, prev_frame):
    """Calculates the bitwise XOR to find differences between two frames."""
    diff = np.bitwise_xor(current_frame, prev_frame)
    return diff

def find_centroid(mask):
    """Calculates the geometric center (centroid) of non-zero pixels in a mask."""
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())) if len(xs) > 0 else None

def pixel_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])

# --- Main Application Logic ---

def main(run_time):
    """
    Main loop to monitor the obstruction map, detect handovers based on centroid jumps,
    log satellite trajectories, and record a video of the map feed.
    """
    print("Starting Starlink handover detector and visualizer...")

    # Setup output directory and files
    output_dir = Path("/opt/home_dir/outputs") # /Users/silver/Desktop/alphartc_dockers/outputs 
    output_dir.mkdir(exist_ok=True)
    # arcs_file = output_dir / "satellite_arcs.json"
    # video_file = output_dir / VIDEO_FILENAME
    handover_indicator = output_dir / "handovers.npy"
    handover_indicator_ts = output_dir / "handovers_ts.npy"

    # # Initialize video writer using OpenCV
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # frame_size = (123, 123)
    # video_writer = cv2.VideoWriter(str(video_file), fourcc, VIDEO_FPS, frame_size)
    # print(f"Recording obstruction map video to: {video_file}")

    prev_frame = None
    prev_second = None
    last_centroid = None
    current_arc = []
    all_arcs = []
    handover_bits = []
    handover_timestamps = []

    # Connection context
    context = starlink_grpc.ChannelContext(target=DISH_IP)

    # Start with a clean map
    clear_obstruction_map(context)
    handover_bit = 0
    time.sleep(1)

    # NOTE: for debugging
    start_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp())

    try:
        while True:
            timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
            if int(timestamp) - start_time >= run_time:
                break
            seconds = int(timestamp) % 60
            frame = get_obstruction_map(context)

            # Assume we maintain the same connection
            if frame is None:
                frame = prev_frame

            # # --- Video Recording Logic ---
            # frame_uint8 = frame.astype(np.uint8) * 255
            # vis_frame = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
            # video_writer.write(vis_frame)

            # --- Handover Detection Logic ---
            if prev_frame is not None:
                diff_mask = xor_diff(frame, prev_frame)
                centroid = find_centroid(diff_mask)

                if centroid:
                    print(f"Satellite centroid detected at: ({centroid[0]:.2f}, {centroid[1]:.2f}) T={seconds}")

                    if last_centroid and pixel_distance(centroid, last_centroid) > HANDOVER_JUMP_THRESHOLD:
                        all_arcs.append(current_arc)

                        # Save the collected trajectories
                        # with open(arcs_file, "w") as f:
                        #     json.dump(all_arcs, f, indent=2)
                        print(f"Handover detected at T={seconds}!")
                        handover_bit = 1

                        # Reset for the next satellite
                        current_arc = []
                        clear_obstruction_map(context)
                        # time.sleep(1)

                    current_arc.append((timestamp, seconds, centroid))
                    last_centroid = centroid

            # log handover time series every second
            if prev_second is None or abs(seconds - prev_second) >= 1:
                prev_second = seconds
                handover_bits.append(handover_bit)
                handover_timestamps.append(timestamp)
                handover_bit = 0

            prev_frame = frame
            # time.sleep(1 / VIDEO_FPS)

    except KeyboardInterrupt:
        print("\nScript terminated by user.")
    finally:
        np.save(handover_indicator, np.array(handover_bits))
        np.save(handover_indicator_ts, np.array(handover_timestamps))
        # video_writer.release()
        # print(f"Video recording stopped. File saved to {video_file}")
        # NOTE: for debugging
        print(f"{len(handover_bits)/60:.2f} minutes. {len(handover_timestamps)/60:.2f} minutes.")
        end_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        print(f"{(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect handover data.")
    parser.add_argument("--run_time", type=int, default=120,
                        help="set the duration this script should run for.")
    args = parser.parse_args()
    run_time = int(args.run_time)
    main(run_time)
