import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load images (replace with your file paths)
#thermal_image = cv2.imread('thermal_image.png', cv2.IMREAD_GRAYSCALE)
#depth_image = cv2.imread('depth_image.png', cv2.IMREAD_GRAYSCALE)

thermal_image=np.load("color.npy")
thermal_image=depth_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
depth_image=np.load("depth.npy")
depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)

print(thermal_image.shape)
print(depth_image.shape)

# Lists to store clicked points
thermal_points = []
depth_points = []

# Function to handle mouse clicks
def onclick(event):
    if event.inaxes == ax1:
        x, y = int(event.xdata), int(event.ydata)
        thermal_points.append((x, y))
        print(f"Thermal Image: Clicked at ({x}, {y})")
        ax1.scatter(x, y, c='red', s=50)
        
    elif event.inaxes == ax2:
        x, y = int(event.xdata), int(event.ydata)
        depth_points.append((x, y))
        print(f"Depth Image: Clicked at ({x}, {y})")
        ax2.scatter(x, y, c='red', s=50)  # Mark the point
        
    plt.draw()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(thermal_image, cmap='gray')
ax1.set_title('Thermal Image')
ax2.imshow(depth_image, cmap='gray')
ax2.set_title('Depth Image')

# Connect click events to the images
fig.canvas.mpl_connect('button_press_event', onclick)
#fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, 'depth'))

plt.tight_layout()
plt.show()

# Print the selected points
print("Selected Thermal Points:", thermal_points)
print("Selected Depth Points:", depth_points)