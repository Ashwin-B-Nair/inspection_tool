import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os
import time 
import argparse

# ctx = rs.context()
# devices = ctx.query_devices()
# for dev in devices:
#     dev.hardware_reset()

# Argument parser
parser = argparse.ArgumentParser(description="RealSense RGB-D Capture Script")
parser.add_argument('--mode', type=str, choices=['manual', 'auto'], required=True,
                    help="Choose between 'manual' or 'automatic' mode")
parser.add_argument('--fps', type=int, default=5,
                    help="Frames per second (only used in automatic mode)")
parser.add_argument('--base_data_folder', type=str, required=True,
                    help="Base folder to store rgb, depth, and pcd data")
args = parser.parse_args()

# Define subfolders
rgb_folder = os.path.join(args.base_data_folder, 'rgb')
depth_folder = os.path.join(args.base_data_folder, 'depth')
pcd_folder = os.path.join(args.base_data_folder, 'pcd')

# Create folders to save
os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)
os.makedirs(pcd_folder, exist_ok=True)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()


"""
Config Parameters

config.enable_stream(
    stream_type,   # e.g., rs.stream.depth or rs.stream.color
    width,         # image width in pixels
    height,        # image height in pixels
    format,        # pixel format (e.g., z16, bgr8)
    framerate      # frames per second (fps)
)

""" 
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, args.fps)  # can increase resolution if needed
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, args.fps) # 848 x 480 OR 1280 x 720
print("Enabled Streams")
align = rs.align(rs.stream.color)   # Align depth to color
pipeline.start(config)  # Start pipeline
frame_counter = 0


def save_frames(color_image, depth_image, frame_counter):
    rgb_path = os.path.join(rgb_folder, f'rgb_{frame_counter:03d}.png')
    depth_path = os.path.join(depth_folder, f'depth_{frame_counter:03d}.npy')    
    cv2.imwrite(rgb_path, color_image)
    np.save(depth_path, depth_image)
    print(f"Saved rgb {frame_counter} and depth {frame_counter}")

try:
    if args.mode == 'auto':
        print(f"Running in automatic mode at {args.fps} FPS")
        while True:
            start_time = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("Skipping frame due to missing data")
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow('Color', color_image)
            cv2.imshow('Depth', depth_colormap)

            save_frames(color_image, depth_image, frame_counter)
            frame_counter += 1

            elapsed_time = time.time() - start_time
            delay = max(int((1.0 / args.fps - elapsed_time) * 1000), 1)
            key = cv2.waitKey(delay) & 0xFF
            if key == 27:  # ESC
                print("Exiting automatic mode...")
                break

    elif args.mode == 'manual':
        print("Running in manual mode (press SPACE to capture, ESC to exit)")
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("Skipping frame due to missing data")
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow('Color', color_image)
            cv2.imshow('Depth', depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Exiting manual mode...")
                break
            elif key == 32:  # SPACE
                save_frames(color_image, depth_image, frame_counter)
                frame_counter += 1

finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    
    
