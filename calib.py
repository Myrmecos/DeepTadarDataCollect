import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2 as cv

# Load the YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract points
thermal_points = config['thermal_points']
depth_points = config['depth_points']

print(type(thermal_points[0]))

# Print the points
print("Thermal Points:", thermal_points)
print("Depth Points:", depth_points)

# Example corresponding points (depth and thermal)
P_d = np.array(depth_points)  # Depth image points
P_t = np.array(thermal_points)  # Thermal image points

# Step 1: Compute scaling factor
dist_d = np.linalg.norm(P_d[1] - P_d[0])  # Distance between two points in depth image
dist_t = np.linalg.norm(P_t[1] - P_t[0])  # Distance between two points in thermal image
s = dist_t / dist_d  # Scaling factor
print("scaling factor (thermal/depth) is:", s)

depth_points = P_d*s

# =============== now we calculate rotation and transformation ========
# Step 1: Center the points
thermal_centroid = np.mean(thermal_points, axis=0)
depth_centroid = np.mean(depth_points, axis=0)

thermal_centered = thermal_points - thermal_centroid
depth_centered = depth_points - depth_centroid


# Step 2: Compute the covariance matrix
H = depth_centered.T @ thermal_centered

# Step 3: Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(H)

# Step 4: Compute the rotation matrix R
R = Vt.T @ U.T

# Handle reflection case (ensure determinant of R is 1)
if np.linalg.det(R) < 0:
    print("determinant of R is negative, flip it.")
    Vt[-1, :] *= -1
    R = Vt.T @ U.T

# Step 5: Compute the translation vector T
T = thermal_centroid - R @ depth_centroid

# Print the results
print("Rotation Matrix (R):")
print(R)
print("Translation Vector (T):")
print(T)

#=============testing

# # Step 1: Center the points
# thermal_centroid = np.mean(thermal_points, axis=0)
# depth_centroid = np.mean(depth_points, axis=0)

# thermal_centered = thermal_points - thermal_centroid
# depth_centered = depth_points - depth_centroid


# # Step 2: Compute the covariance matrix
# H = thermal_centered.T @ depth_centered

# # Step 3: Perform Singular Value Decomposition (SVD)
# U, S, Vt = np.linalg.svd(H)

# # Step 4: Compute the rotation matrix R
# R = Vt.T @ U.T

# # Handle reflection case (ensure determinant of R is 1)
# if np.linalg.det(R) < 0:
#     print("determinant of R is negative, flip it.")
#     Vt[-1, :] *= -1
#     R = Vt.T @ U.T

# # Step 5: Compute the translation vector T
# T = depth_centroid - R @ thermal_centroid

# # Print the results
# print("Rotation Matrix (R):")
# print(R)
# print("Translation Vector (T):")
# print(T)

# =================== now we map the depth image to thermal array ===============

# Load the depth image
# thermal = np.load("color.npy")
thermal_image=np.load("color.npy")
thermal= cv.cvtColor(thermal_image, cv.COLOR_BGR2GRAY)
#depth_image = cv.rotate(depth_image, cv.ROTATE_90_CLOCKWISE)
# Define the padding size (top, bottom, left, right)
# top = 200
# bottom = 200
# left = 200
# right = 200

# Zero-pad the image using cv2.copyMakeBorder
# padded_image = cv.copyMakeBorder(thermal, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])

# Define the scaling factor, rotation matrix, and translation vector
# s = dist_t / dist_d  # Scaling factor (computed earlier)

# Get the shape of the depth image
height, width = thermal.shape

# Create a grid of coordinates for the depth image
x, y = np.meshgrid(np.arange(width), np.arange(height))
coords = np.vstack((x.flatten(), y.flatten()))  # Shape: (2, N), where N = height * width

# Step 1: Apply scaling
#coords_scaled = s * coords
coords_scaled = coords #DEBUG ONLY==============================================

# Step 2: Apply rotation and translation
coords_transformed = np.dot(R, coords_scaled) + T[:, np.newaxis]
print("after rotation: ", coords_transformed)

# Reshape the transformed coordinates back to the image shape
x_transformed = coords_transformed[0, :].reshape(height, width)
y_transformed = coords_transformed[1, :].reshape(height, width)

# Step 3: Interpolate the depth values at the transformed coordinates
from scipy.interpolate import griddata

# Create a grid of original coordinates
grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

# Flatten the depth image and original coordinates
points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
values = thermal.flatten()

# Interpolate the depth values at the transformed coordinates
transformed_depth = griddata(points, values, (x_transformed, y_transformed), method='linear')

#resize
# Compute the new dimensions
new_width = int(transformed_depth.shape[1] / s)
new_height = int(transformed_depth.shape[0] / s)

# Resize the distance image using interpolation
resized_distance = cv.resize(transformed_depth, (new_width, new_height), interpolation=cv.INTER_LINEAR)

# Step 4: Visualize the transformed depth image
print(resized_distance)
plt.imshow(resized_distance, cmap='gray')
plt.title("Transformed Depth Image")
plt.colorbar()
plt.show()