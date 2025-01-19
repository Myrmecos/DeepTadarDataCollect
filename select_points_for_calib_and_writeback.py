import yaml
import select_points_for_calib as sc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

transform_points = []
reference_points = []

# Load the YAML file
with open('image.yaml', 'r') as file:
    data = yaml.safe_load(file)

# with open('transform.yaml', 'w') as outfile:
    
baseDir = "RawData/"

transform = "realsense_depth/"
target = "seek_thermal/"

image_names = []
# goal: exp1/***.npy, exp11/***.npy, exp21/***.npy
distance = "4"
dirinds = [distance, "1"+distance, "2"+distance] #1, 11, 21

transform_image_names = []
target_image_names = []
for dirind in dirinds:
    dirname = "exp"+dirind+"/"
    validInd = data[dirind]
    #list all files under exp**, and select the names corresponding to validInd
    imageNamesTrans=os.listdir(baseDir+dirname+transform)
    imageNamesTarget=os.listdir(baseDir+dirname+target)
    # add the names to image_names
    
    for i in validInd:
        transform_image_names.append(baseDir+dirname+transform+imageNamesTrans[i])
        target_image_names.append(baseDir+dirname+target+imageNamesTarget[i])
    

print(len(transform_image_names))


#===================================calib start!==============================================

for i in range(6):
    # load image
    ind = i + 0
    transform_image=np.load(transform_image_names[ind]) #image to be rotated, transform and resized
    reference_image=np.load(target_image_names[ind])

    sc.showImagePanels(transform_image, reference_image)

sc.print_list()

# Print the selected points
# print("Selected transform Points:", transform_points) #transform refers to those points to be transformed and mapped (should be depth)
# print("Selected reference Points:", reference_points)
    
    

