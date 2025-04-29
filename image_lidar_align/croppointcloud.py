import open3d as o3d 
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from pyntcloud import PyntCloud
from pypcd import pypcd
import pandas as pd

pcd_path = "/home/astar/dart_ws/single_scene_calibration/0.pcd"

def readPcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print("readPcd received empty point cloud")
        return
    return pcd

cloud = PyntCloud.from_file(pcd_path)
print("Available attributes:", cloud.points.columns)

points = np.asarray([cloud.points["x"], cloud.points["y"], cloud.points["z"]]).T
colors = np.asarray(cloud.points["intensity"]).T

# points = np.asarray(pcd.points)
# colors = np.asarray(pcd.normals)
print(f"Shape of points: {points.shape}; shape of colors: {colors.shape}")

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
filtered_intensity = colors[mask]


# # Save the cropped point cloud
# Combine XYZ and intensity
data = np.column_stack((filtered_points, filtered_intensity))  # Shape: (N, 4)

# Write PCD file
with open("cropped_point_cloud.pcd", "w") as f:
    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z intensity\n")
    f.write("SIZE 4 4 4 4\n")
    f.write("TYPE F F F F\n")
    f.write("COUNT 1 1 1 1\n")
    f.write(f"WIDTH {len(data)}\n")
    f.write("HEIGHT 1\n")
    f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
    f.write(f"POINTS {len(data)}\n")
    f.write("DATA ascii\n")
    for point in data:
        f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.6f}\n")