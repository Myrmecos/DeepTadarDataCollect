import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import copy
import calib

# TODO: map a depth image to a thermal image, give the R, T, S at different distance
# 0. output image: copy original image


#0.1. read image
# load all valid image names
# transform_points = []
# reference_points = []
# with open('image.yaml', 'r') as file:
#     data = yaml.safe_load(file)
# baseDir = "RawData/"
# transform_dir = "realsense_depth/"
# target_dir = "seek_thermal/"
# distance_str = "1"
# dirinds = [distance_str, "1"+distance_str, "2"+distance_str] #1, 11, 21
# transform_image_names = []
# target_image_names = []
# for dirind in dirinds:
#     dirname = "exp"+dirind+"/"
#     validInd = data[dirind]
#     #list all files under exp**, and select the names corresponding to validInd
#     imageNamesTrans=os.listdir(baseDir+dirname+transform_dir)
#     imageNamesTarget=os.listdir(baseDir+dirname+target_dir)
#     # add the names to image_names
#     for i in validInd:
#         transform_image_names.append(baseDir+dirname+transform_dir+imageNamesTrans[i])
#         target_image_names.append(baseDir+dirname+target_dir+imageNamesTarget[i])
# print(len(transform_image_names))

# # load the image
# depth_ori = np.load(transform_image_names[0])
# depth_res = np.load(transform_image_names[0])

# print(depth_ori)
# 1. separate out values at 1.5m-2.5m, make other values invalid (such as nan?)
def read_yaml(filename):
    with open(filename) as file:
        data = yaml.safe_load(file)
    return np.array(data["R"]), np.array(data["T"]), np.float64(data["s"])

R2, T2, s2 = read_yaml("calibresults/seek_thermal/2.yaml")
R3, T3, s3 = read_yaml("calibresults/seek_thermal/3.yaml")

depth_ori = np.load("recov.npy")
depth_ori1 = copy.copy(depth_ori)
depth_ori2 = copy.copy(depth_ori)

#1.1. at 1.5-2.5m
depth_ori1[depth_ori<1.5]=np.nan
depth_ori1[depth_ori>2.5]=np.nan
#2.2. at 2.5-3.5m
depth_ori2[depth_ori<2.5]=np.nan
depth_ori2[depth_ori>3.5]=np.nan

# 2. apply transformation (RTS) to the image
transformed_image1 = calib.transform_img(depth_ori1, R2, T2, s2)
transformed_image2 = calib.transform_img(depth_ori2, R3, T3, s3)

plt.imshow(transformed_image1, alpha=1)
plt.imshow(transformed_image2, alpha=1)
plt.show()

# 3. mask the image back to original image. Use valid values to cover up original values. Invalid values are ignored.


