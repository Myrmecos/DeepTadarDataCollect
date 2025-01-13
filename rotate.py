import numpy as np
import math

# Define the rotation function
def rotate_point(x, y, theta):
    # Convert theta from degrees to radians
    theta_rad = math.radians(theta)
    
    # Calculate the new coordinates after rotation
    x_new = x * math.cos(theta_rad) - y * math.sin(theta_rad)
    y_new = x * math.sin(theta_rad) + y * math.cos(theta_rad)
    
    return x_new, y_new

# Define the points
points = [
    [10, 140],
    [110, 280],
    [210, 300],
    [300, 50]
]

# Define the rotation angle in degrees
theta = 90  # Example: rotate by 45 degrees

# Rotate each point and print the new coordinates
rotated_points = []
for point in points:
    x, y = point
    x_new, y_new = rotate_point(x, y, theta)
    rotated_points.append([x_new, y_new])

# Print the rotated points
# print("Original Points:", points)
# print("Rotated Points:", rotated_points)
print("Original points: ")
for point in points:
    print("-", point)

for point in rotated_points:
    point[0] = point[0]+480
    print("-", point)