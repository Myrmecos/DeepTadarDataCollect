import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.interpolate import griddata

#
#根据需求，认为需要把distance对应到transform上，即将distance通过resize，平移和旋转贴合到transform上。
#下方示例是将transform通过转换对应到distance上，只要把transform和distance的文件交换即可。
#

# Load the YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract points
transform_points = config['transform_points']
reference_points = config['reference_points']

print(type(transform_points[0]))

# Print the points
print("transform Points:", transform_points)
print("reference Points:", reference_points)

# Example corresponding points (reference and transform)
P_d = np.array(reference_points)  # reference image points
P_t = np.array(transform_points)  # transform image points

# Step 1: Compute scaling factor
dist_d = np.linalg.norm(P_d[1] - P_d[0])  # Distance between two points in reference image
dist_t = np.linalg.norm(P_t[1] - P_t[0])  # Distance between two points in transform image
s = dist_t / dist_d  # Scaling factor
print("scaling factor (transform/reference) is:", s)

reference_points = P_d*s

# =============== now we calculate rotation and transformation ========================
# Step 1: Center the points
transform_centroid = np.mean(transform_points, axis=0)
reference_centroid = np.mean(reference_points, axis=0)

transform_centered = transform_points - transform_centroid
reference_centered = reference_points - reference_centroid


# Step 2: Compute the covariance matrix
H = reference_centered.T @ transform_centered

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
T = transform_centroid - R @ reference_centroid

# Print the results
print("Rotation Matrix (R):")
print(R)
print("Translation Vector (T):")
print(T)

# =================== now we map the transform array to reference image==========================

# Load the reference image
# transform = np.load("color.npy")
transform_image=np.load("ori.npy")

# Define the padding size (top, bottom, left, right)
print("image shape:", transform_image.shape)
if (transform_image.shape[1]-transform_image.shape[0]>0):
    top = 0 #abs(transform_image.shape[1]-transform_image.shape[0])
    bottom = abs(transform_image.shape[1]-transform_image.shape[0])
    left = 0
    right = 0
else: 
    # we will check later how to deal with a tall image
    exit(1)
#Zero-pad the image using cv2.copyMakeBorder
transform_image = cv.copyMakeBorder(transform_image, top, bottom, left, right, cv.BORDER_CONSTANT, value=np.nan)
# plt.imshow(transform_image , cmap='gray')
# plt.title("Transformed reference Image")
# plt.colorbar()
# plt.show()

#transform_image = cv.rotate(transform_image, cv.ROTATE_90_CLOCKWISE) #testing only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#transform= cv.cvtColor(transform_image, cv.COLOR_BGR2GRAY) #testing only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Get the shape of the reference image
transform = transform_image
height, width = transform.shape

# Create a grid of coordinates for the reference image
x, y = np.meshgrid(np.arange(width), np.arange(height))
coords = np.vstack((x.flatten(), y.flatten()))  # Shape: (2, N), where N = height * width

# Step 1: Apply scaling
coords_scaled = coords

# Step 2: Apply rotation and translation
coords_transformed = np.dot(R, coords_scaled) + T[:, np.newaxis]
print("after rotation: ", coords_transformed)

# Reshape the transformed coordinates back to the image shape
x_transformed = coords_transformed[0, :].reshape(height, width)
y_transformed = coords_transformed[1, :].reshape(height, width)

# Step 3: Interpolate the reference values at the transformed coordinates

# Create a grid of original coordinates
grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
print("meshgrid ok")

# Flatten the reference image and original coordinates
points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
values = transform.flatten()
print("flatten ok")

# Interpolate the reference values at the transformed coordinates
transformed_reference = griddata(points, values, (x_transformed, y_transformed), method='linear')
print("transform_reference ok")

#resize
# Compute the new dimensions
new_width = int(transformed_reference.shape[1] / s)
new_height = int(transformed_reference.shape[0] / s)

print("transform ok")

# Resize the distance image using interpolation
resized_distance = cv.resize(transformed_reference, (new_width, new_height), interpolation=cv.INTER_LINEAR)

print("resize ok")

# Step 4: Visualize the transformed reference image
print(resized_distance)
plt.imshow(resized_distance, cmap='gray')
plt.title("Transformed reference Image")
plt.colorbar()
plt.show()