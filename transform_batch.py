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

# step 2: load R, T, scale
def load_RTS(yaml_base_dir, sensor_name, max_dis):
    # the return format: [[R1, T1, s1], [R2, T2, s2], ...]
    RTS = []
    for i in range(1, max_dis+1):
        yaml_file_name = yaml_base_dir+sensor_name+str(i)+".yaml"
        with open(yaml_file_name, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                R = data['R']
                T = data['T']
                s = data['s']
                RTS.append([R, T, s])
            except yaml.YAMLError as exc:
                print(exc)
    return RTS

    



if __name__=="__main__":
    dirbase = "/media/zx/zx-data/RawData/exp06/"
    sensor_name = "senxor_m08/"
    transform_name = "realsense_depth/"
    yaml_base_dir = "calibresults/"
    max_dis = 6

    # step 1: load image names
    transform_image_names = get_tansform_images_names(dirbase, transform_name)
    
    # step 2: load R, T, scale
    RTS = load_RTS(yaml_base_dir, sensor_name, max_dis)
    # for i in range(len(RTS)):
    #     print(f"This is the RTS of {i+1}m distance")
    #     print(RTS[i])

    
