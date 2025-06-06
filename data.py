import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os
import time 

ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()
    
# Create folders to save
os.makedirs('rgb', exist_ok=True)
os.makedirs('depth', exist_ok=True)
os.makedirs('pcd', exist_ok=True)

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
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # can increase resolution if needed
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
print("Enabled Streams")
# Start pipeline
pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# # Capture loop - Manual - Space to capture 
# frame_counter = 0

# try:
#     while True:
#         print("Waiting for frames")
#         frames = pipeline.wait_for_frames()
#         aligned_frames = align.process(frames)

#         depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()

#         if not depth_frame or not color_frame:
#             print("Skipping frame due to missing data")
#             continue

#         # Convert to numpy arrays
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())

#         # Show color and depth images
#         depth_colormap = cv2.applyColorMap(
#             cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
#         )
#         cv2.imshow('Color', color_image)
#         cv2.imshow('Depth', depth_colormap)

#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # ESC
#             print("Exiting...")
#             break
#         elif key == 32:  # SPACE
#             # Save RGB and depth
#             rgb_path = f'rgb/rgb_{frame_counter:03d}.png'
#             depth_path = f'depth/depth_{frame_counter:03d}.npy'
#             cv2.imwrite(rgb_path, color_image)
#             np.save(depth_path, depth_image)
#             print(f"Saved {rgb_path} and {depth_path}")

#             # Convert to point cloud
#             pc = rs.pointcloud()
#             points = pc.calculate(depth_frame)
#             pc.map_to(color_frame)
#             vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
#             tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

#             # Get color image as float in [0, 1] for Open3D
#             color_image_float = color_image.astype(np.float32) / 255.0
#             h, w, _ = color_image.shape

#             # Map texture coords (u,v) to pixel indices (x,y)
#             u = tex[:, 0] * w
#             v = tex[:, 1] * h

#             # Clamp values to image bounds
#             u = np.clip(u, 0, w - 1).astype(np.int32)
#             v = np.clip(v, 0, h - 1).astype(np.int32)

#             # Get RGB color for each vertex
#             colors = color_image_float[v, u]  # Shape: (N, 3)

#             # Save as Open3D point cloud
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(vtx)
#             pcd.colors = o3d.utility.Vector3dVector(colors)
#             pcd_path = f'pcd/frame_{frame_counter:03d}.ply'
#             o3d.io.write_point_cloud(pcd_path, pcd)
#             print(f"Saved {pcd_path}")

#             frame_counter += 1

# finally:
#     cv2.destroyAllWindows()
#     pipeline.stop()


#------------------------------------------------------------------------
# Capture loop - automated
num_frames = 10           # Number of frames to capture
interval_sec = 1          # Interval between frames in seconds (change as needed)

try:
    for i in range(num_frames):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            print(f"Frame {i}: missing data, skipping.")
            continue

        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Save RGB and depth
        cv2.imwrite(f'rgb/rgb_{i:03d}.png', color_image)
        np.save(f'depth/depth_{i:03d}.npy', depth_image)

        # Convert to point cloud
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        # Get color image as float in [0, 1] for Open3D
        color_image_float = color_image.astype(np.float32) / 255.0
        h, w, _ = color_image.shape

        # Map texture coords (u,v) to pixel indices (x,y)
        u = tex[:, 0] * w
        v = tex[:, 1] * h

        # Clamp values to image bounds
        u = np.clip(u, 0, w - 1).astype(np.int32)
        v = np.clip(v, 0, h - 1).astype(np.int32)

        # Get RGB color for each vertex
        colors = color_image_float[v, u]  # Shape: (N, 3)

        # Save as Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(f'pcd/frame_{i:03d}.ply', pcd)

        print(f"Captured frame {i+1}/{num_frames}")
        time.sleep(interval_sec)  # Wait before capturing next frame

finally:
    pipeline.stop()
    
# try:
#     for i in range(30):  # capture 30 frames
#         frames = pipeline.wait_for_frames()
#         aligned_frames = align.process(frames)

#         depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()

#         if not depth_frame or not color_frame:
#             continue

#         # Convert to numpy arrays
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())

#         cv2.imshow('Color', color_image)
#         key = cv2.waitKey(0)  # Wait for a key press
#         if key == 27:  # ESC to exit early
#             break
#         # Save RGB and depth
#         cv2.imwrite(f'rgb/rgb_{i:03d}.png', color_image)
#         np.save(f'depth/depth_{i:03d}.npy', depth_image)

#         # Convert to point cloud
#         pc = rs.pointcloud()
#         points = pc.calculate(depth_frame)
#         pc.map_to(color_frame)
#         vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
#         tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

#         # Get color image as float in [0, 1] for Open3D
#         color_image_float = color_image.astype(np.float32) / 255.0
#         h, w, _ = color_image.shape

#         # Map texture coords (u,v) to pixel indices (x,y)
#         u = tex[:, 0] * w
#         v = tex[:, 1] * h

#         # Clamp values to image bounds
#         u = np.clip(u, 0, w - 1).astype(np.int32)
#         v = np.clip(v, 0, h - 1).astype(np.int32)

#         # Get RGB color for each vertex
#         colors = color_image_float[v, u]  # Shape: (N, 3)

#         # Save as Open3D point cloud
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(vtx)
#         pcd.colors = o3d.utility.Vector3dVector(colors)
#         o3d.io.write_point_cloud(f'pcd/frame_{i:03d}.ply', pcd)
        
# finally:
#     cv2.destroyAllWindows()
#     pipeline.stop()
