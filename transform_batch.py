# TODO: transform depth images in a folder (realsense_depth) to match thermal images (e.g. senxor_m08); save result in another folder (depth_senxor_m08)
# Step1: obtain file names of all images in transform image folder (realsense depth)
# Step2: load R, T, s for the transformation from realsense depth to senxor_m08
# step3: apply R, T, s to each image in realsense depth folder
# step 4: save the transformed images in a new folder (depth_senxor_m08)

import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import os

# Step1: obtain file names of all images in transform image folder (realsense depth)
def get_tansform_images_names(dirbase, transform_folder_name):
    transform_image_names = os.listdir(dirbase+transform_folder_name)
    return [dirbase+transform_folder_name+image_name for image_name in transform_image_names]
    



if __name__=="__main__":
    dirbase = "/media/zx/zx-data/RawData/exp06/"
    sensor_name = "senxor_m08/"
    transform_name = "realsense_depth/"



    transform_image_names = get_tansform_images_names(dirbase, transform_name)
    cv.imshow("depth:", np.load(transform_image_names[-1]))
    cv.waitKey(1000)