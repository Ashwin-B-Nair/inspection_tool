import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os

# Create folders to save
os.makedirs('rgb', exist_ok=True)
os.makedirs('depth', exist_ok=True)
os.makedirs('pcd', exist_ok=True)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start pipeline
pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Capture loop
try:
    for i in range(30):  # capture 30 frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
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

        # Save as Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)
        o3d.io.write_point_cloud(f'pcd/frame_{i:03d}.ply', pcd)

finally:
    pipeline.stop()
