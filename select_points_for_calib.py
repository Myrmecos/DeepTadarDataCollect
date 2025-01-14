import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load images (replace with your file paths)
#transform_image = cv2.imread('transform_image.png', cv2.IMREAD_GRAYSCALE)
#reference_image = cv2.imread('reference_image.png', cv2.IMREAD_GRAYSCALE)

transform_image=np.load("ori.npy") #image to be rotated, transform and resized
reference_image=np.load("trans.npy")
#reference_image= cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
#transform_image = cv2.rotate(transform_image, cv2.ROTATE_90_CLOCKWISE) #TESTING ONLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print(transform_image.shape)
print(reference_image.shape)

# Lists to store clicked points
transform_points = []
reference_points = []

# Function to handle mouse clicks
def onclick(event):
    if event.inaxes == ax1:
        x, y = int(event.xdata), int(event.ydata)
        transform_points.append((x, y))
        print(f"transform Image: Clicked at ({x}, {y})")
        ax1.scatter(x, y, c='red', s=50)
        
    elif event.inaxes == ax2:
        x, y = int(event.xdata), int(event.ydata)
        reference_points.append((x, y))
        print(f"reference Image: Clicked at ({x}, {y})")
        ax2.scatter(x, y, c='red', s=50)  # Mark the point
        
    plt.draw()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(transform_image, cmap='gray')
ax1.set_title('transform Image')
ax2.imshow(reference_image, cmap='gray')
ax2.set_title('reference Image')

# Connect click events to the images
fig.canvas.mpl_connect('button_press_event', onclick)
#fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, 'reference'))

plt.tight_layout()
plt.show()

# Print the selected points
# print("Selected transform Points:", transform_points) #transform refers to those points to be transformed and mapped (should be depth)
# print("Selected reference Points:", reference_points)
print("transform points: ")
for p in transform_points:
    print("- [",p[0] ,", ", p[1], "]", sep = "")
print("reference points:")
for p in reference_points:
    print("- [",p[0], ", ", p[1], "]", sep = "")
