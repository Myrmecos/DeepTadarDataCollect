import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

ax1 = None
ax2 = None
transform_points = []
reference_points = []


# Function to handle mouse clicks
def onclick(event):
    global ax1, ax2
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


def showImagePanels():
    # Lists to store clicked points
    global ax1, ax2
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

if __name__=='__main__':
    baseDir = "RawData/exp04/"
    transform_files = os.listdir(baseDir+"realsense_depth/")
    reference_files = os.listdir(baseDir+"MLX/")
    for i in range(5):
        # load image
        ind = i
        transform_image=np.load(baseDir+"realsense_depth/"+transform_files[ind]) #image to be rotated, transform and resized
        reference_image=np.load(baseDir+"seek_thermal/"+reference_files[ind])
        #transform_image = np.load("ori.npy")
        #reference_image = np.load("trans.npy")
        #reference_image = cv2.imread("shifted.png")

        #to grayscale and normalize the images
        reference_image= cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        reference_image = cv2.normalize(reference_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        transform_image = cv2.cvtColor(transform_image, cv2.COLOR_BGR2GRAY)
        transform_image = cv2.normalize(transform_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        print(transform_image.shape)
        print(reference_image.shape)

        showImagePanels()

    # Print the selected points
    # print("Selected transform Points:", transform_points) #transform refers to those points to be transformed and mapped (should be depth)
    # print("Selected reference Points:", reference_points)
    print("transform_points: ")
    for p in transform_points:
        print("- [",p[0] ,", ", p[1], "]", sep = "")
    print("reference_points:")
    for p in reference_points:
        print("- [",p[0], ", ", p[1], "]", sep = "")
