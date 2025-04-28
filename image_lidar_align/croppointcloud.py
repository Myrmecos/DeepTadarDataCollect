import open3d as o3d 
import numpy as np 
import matplotlib.pyplot as plt
import cv2

def readPcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print("readPcd received empty point cloud")
        return
    return pcd

#pcd = readPcd("/home/astar/dart_ws/single_scene_calibration/0.pcd")
pcd = readPcd("/home/astar/dart_ws/calib/calibpointcloud/calibscene.pcd")
points = np.asarray(pcd.points)

# Define HFOV and VFOV in degrees
hfov_deg = 30  # Horizontal field of view
vfov_deg = 20  # Vertical field of view

# Convert to radians for calculations
hfov_rad = np.deg2rad(hfov_deg / 2)  # ±15 degrees
vfov_rad = np.deg2rad(vfov_deg / 2)  # ±10 degrees

# Compute spherical coordinates
# Assuming the sensor is at (0, 0, 0) facing along +z
x, y, z = points[:, 0], points[:, 1], points[:, 2]
rho = np.sqrt(x**2 + y**2 + z**2)  # Distance from origin
theta = np.arctan2(y, x)  # Azimuthal angle (horizontal)
phi = np.arccos(z / rho)  # Elevation angle from xy-plane

# Filter points within HFOV and VFOV
mask = (np.abs(theta) <= hfov_rad) & (np.abs(phi - np.pi/2) <= vfov_rad)
filtered_points = points[mask]

# Create a new point cloud with filtered points
cropped_pcd = o3d.geometry.PointCloud()
cropped_pcd.points = o3d.utility.Vector3dVector(filtered_points)

# Save the cropped point cloud
o3d.io.write_point_cloud("cropped_point_cloud.pcd", cropped_pcd)

# Optional: Visualize the result
o3d.visualization.draw_geometries([cropped_pcd])