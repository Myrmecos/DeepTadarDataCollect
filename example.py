import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate the COLORMAP_JET lookup table
colormap_jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, 256), cv2.COLORMAP_JET)
colormap_jet = colormap_jet.squeeze()  # Shape: (256, 3)

# Step 2: Create a reverse mapping from RGB to depth
# We'll use a dictionary for fast lookup
rgb_to_depth = {tuple(colormap_jet[i]): i for i in range(256)}

# Step 3: Load the colored depth image
colored_depth_image = np.load("dep.npy")  # Replace with your image path

# Step 4: Recover the depth image
depth_image = np.zeros((colored_depth_image.shape[0], colored_depth_image.shape[1]), dtype=np.uint8)

for i in range(colored_depth_image.shape[0]):
    for j in range(colored_depth_image.shape[1]):
        pixel_color = tuple(colored_depth_image[i, j])
        if pixel_color in rgb_to_depth:
            depth_image[i, j] = rgb_to_depth[pixel_color]
        else:
            # Handle unknown colors (e.g., use nearest neighbor or default value)
            depth_image[i, j] = 0  # Default to 0 (or handle differently)

# Step 5: Scale the depth image back to original depth range
# Assuming the depth was scaled by alpha=0.03 during the original mapping
original_depth_image = depth_image / 0.03/1000

# Save or display the recovered depth image
cv2.imwrite("recov.png", original_depth_image)
plt.imshow(original_depth_image)
plt.show()
np.save("recov.npy", original_depth_image)
print("Depth image recovered successfully!")