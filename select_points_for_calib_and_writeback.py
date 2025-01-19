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
distance = "2"
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

















# for i in range(5):
#     # load image
#     ind = i + 16
#     transform_image=np.load(baseDir+"realsense_depth/"+transform_files[ind]) #image to be rotated, transform and resized
#     reference_image=np.load(baseDir+"seek_thermal/"+reference_files[ind])
#     #transform_image = np.load("ori.npy")
#     #reference_image = np.load("trans.npy")
#     #reference_image = cv2.imread("shifted.png")

#     #to grayscale and normalize the images
#     reference_image= cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
#     reference_image = cv2.normalize(reference_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#     transform_image = cv2.cvtColor(transform_image, cv2.COLOR_BGR2GRAY)
#     transform_image = cv2.normalize(transform_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

#     print(transform_image.shape)
#     print(reference_image.shape)

#     sc.showImagePanels()

# # Print the selected points
# # print("Selected transform Points:", transform_points) #transform refers to those points to be transformed and mapped (should be depth)
# # print("Selected reference Points:", reference_points)
# print("transform_points: ")
# for p in transform_points:
#     print("- [",p[0] ,", ", p[1], "]", sep = "")
# print("reference_points:")
# for p in reference_points:
#     print("- [",p[0], ", ", p[1], "]", sep = "")
    

