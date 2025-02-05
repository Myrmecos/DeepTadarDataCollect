import yaml
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

matplotlib.use('TkAgg')

#example usage: 
# python select_points_for_calib.py --transform realsense_depth/ --target MLX/ --distance 1

#goal: select points on both corresponding images__
transform_points = []
reference_points = []

# Load the YAML file
with open('calibresults/image.yaml', 'r') as file:
    data = yaml.safe_load(file)

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument("--transform", type=str, help="the name of the folder containing the images to be transformed")
parser.add_argument("--target", type=str, help="the name of the folder containing the reference images")
parser.add_argument("--distance", type=str, help="the distance of the images")

args = parser.parse_args()

transform = args.transform
target = args.target
distance = args.distance

# Class to select points on two images
class PointSelector:
    ax1 = None
    ax2 = None
    cursor1 = None
    cursor2 = None
    transform_points = []
    reference_points = []

    def __init__(self):
        pass


    # Function to handle mouse clicks
    def onclick(self, event):
        global ax1, ax2
        cursor1, = ax1.plot([], [], 'r+')
        cursor2, = ax2.plot([], [], 'r+')
        if event.inaxes == ax1:
            x, y = int(event.xdata), int(event.ydata)
            self.transform_points.append([x, y])
            print(f"transform Image: Clicked at ({x}, {y})")
            ax1.scatter(x, y, c='red', s=50)
            
        elif event.inaxes == ax2:
            x, y = int(event.xdata), int(event.ydata)
            self.reference_points.append([x, y])
            print(f"reference Image: Clicked at ({x}, {y})")
            ax2.scatter(x, y, c='red', s=50)  # Mark the point
            
        plt.draw()

    # show two images
    def showImagePanels(self, transform_image, reference_image):
        # Lists to store clicked points
        global ax1, ax2
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(transform_image, cmap='gray')
        ax1.set_title('transform Image')
        ax2.imshow(reference_image, cmap='gray')
        ax2.set_title('reference Image')

        # Connect click events to the images
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        #fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, 'reference'))

        plt.tight_layout()
        
        plt.show()
        return self.return_list()

    # return the two lists of corresponding points
    def return_list(self):
        return self.transform_points, self.reference_points

    # print the two lists of corresponding points
    def print_list(self):
        print("transform_points: ")
        for p in self.transform_points:
            print(" - [",p[0] ,", ", p[1], "]", sep = "")
        print("reference_points:")
        for p in self.reference_points:
            print(" - [",p[0], ", ", p[1], "]", sep = "")

    # clear the two lists
    def clear_list(self):
        transform_points = []
        reference_points = []

    # callback function, for moving cursor
    def on_mouse_move(self, event):
        if event.inaxes == ax1:
            self.cursor1.set_data(event.xdata, event.ydata)
        elif event.inaxes == ax2:
            self.cursor2.set_data(event.xdata, event.ydata)
        plt.draw()

# first obtain the filenames of images
# then asks users to select corresponding points for a specific distance, return two lists of points (points on transform image and points on reference image)
# transform image: depth (to be mapped onto thermal), reference image: thermal
def calib_for_distance_m(ps, transform_dir, target_dir, distance_str):
    #dirinds = [distance_str, "1"+distance_str, "2"+distance_str] #1, 11, 21
    
    transform_image_names = []
    target_image_names = []
    
    dirname = f"MSC/calibImages/{distance}/"
    imageNamesTrans=os.listdir(dirname+transform_dir)
    imageNamesTarget=os.listdir(dirname+target_dir)
    # add the names to image_names
        
    for i in range(len(imageNamesTrans)):
        transform_image_names.append(f"MSC/calibImages/{distance}/{transform_dir}{imageNamesTrans[i]}")
        target_image_names.append(f"MSC/calibImages/{distance}/{target_dir}{imageNamesTarget[i]}")

    print(len(transform_image_names))

    #===================================calib start!==============================================
    for i in range(6):
        # load image
        ind = i + 0
        transform_image=np.load(transform_image_names[ind]) #image to be rotated, transform and resized
        reference_image=np.load(target_image_names[ind])
        ps.showImagePanels(transform_image, reference_image)

    return ps.return_list()

if __name__=="__main__":
    # indices of images that are used for selecting points for calibration are stored in calibresults/image.yaml
    ps = PointSelector()

    transform_points, reference_points = calib_for_distance_m(ps, transform, target, distance)
    ps.print_list()

    data = {
        "transform_points": transform_points,
        "reference_points": reference_points
    }

    to_write = input("Write to yaml file? y/n")
    if (to_write=="y"):
        print("write stuffs back!")
        with open(f'MSC/calib1points/{target}{distance}.yaml', 'w') as file:
            yaml.dump(data, file, default_flow_style=True)
            
        