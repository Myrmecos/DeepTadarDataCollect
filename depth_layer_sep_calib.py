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

R7, T7, s7 = read_yaml("calibresults/seek_thermal/3.yaml")


depth_ori = np.load("recov.npy")
# plt.imshow(depth_ori)
# plt.show()

background = calib.transform_img(depth_ori, R7, T7, s7)
#background[(background<7.5) & (background>=0.5)]=np.nan


for i in range(6):
    ind = i+1
    depth_ori1 = copy.copy(depth_ori)
    

    #1.1. at 1.5-2.5m
    depth_ori1[depth_ori<ind-0.5]=np.nan
    depth_ori1[depth_ori>=ind+0.5]=np.nan

    R2, T2, s2 = read_yaml(f"calibresults/seek_thermal/{ind}.yaml")
    

    # 2. apply transformation (RTS) to the image
    transformed_image1 = calib.transform_img(depth_ori1, R2, T2, s2)

    mask = ~np.isnan(transformed_image1)
    background[mask] = transformed_image1[mask]


#plt.imshow(transformed_image1, alpha=1)
plt.imshow(background)
plt.show()

# 3. mask the image back to original image. Use valid values to cover up original values. Invalid values are ignored.


