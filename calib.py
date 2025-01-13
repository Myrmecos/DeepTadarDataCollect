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

# Step 2: Rescale depth image
P_d_scaled = s * P_d

# Step 3: Compute rotation and translation
# Center the points
center_d = np.mean(P_d_scaled, axis=0)
center_t = np.mean(P_t, axis=0)
P_d_centered = P_d_scaled - center_d
P_t_centered = P_t - center_t

# Compute the covariance matrix
H = P_d_centered.T @ P_t_centered

# Singular Value Decomposition
U, S, Vt = np.linalg.svd(H)

# Rotation matrix
R = Vt.T @ U.T

# Translation vector
T = center_t - R @ center_d

# print("Rotation:\n", R)
# print("Translation:\n", T)

# # Step 4: Apply transformation
# P_d_transformed = (R @ P_d_scaled.T).T + T

# # Now P_d_transformed should be aligned with P_t
# print("before scaled: ")
# print("after scaled: ", P_d_scaled.shape)
# print("before transform: ", P_d_centered.shape)
# print("afte transform: ", P_d_transformed.shape)

depth_image=np.load("depth.npy")
#shape
depth_image = cv.cvtColor(depth_image, cv.COLOR_BGR2GRAY)
print(depth_image.shape)
height, width = depth_image.shape
#grid of coordinates for depth image
x, y = np.meshgrid(np.arange(width), np.arange(height))
coords = np.vstack((x.flatten(), y.flatten()))

# Step 1: Apply scaling
coords_scaled = s * coords

# Step 2: Apply rotation and translation
coords_transformed = np.dot(R, coords_scaled) + T[:, np.newaxis]

# Reshape the transformed coordinates back to the image shape
x_transformed = coords_transformed[0, :].reshape(height, width)
y_transformed = coords_transformed[1, :].reshape(height, width)

# Step 3: Interpolate the depth values at the transformed coordinates
from scipy.interpolate import griddata

# Create a grid of original coordinates
grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

# Flatten the depth image and original coordinates
points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
values = depth_image.flatten()

# Interpolate the depth values at the transformed coordinates
transformed_depth = griddata(points, values, (x_transformed, y_transformed), method='linear')

# Step 4: Visualize the transformed depth image
plt.imshow(transformed_depth, cmap='gray')
plt.title("Transformed Depth Image")
plt.colorbar()
plt.show()

