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
from scipy.ndimage import map_coordinates
import copy
matplotlib.use('TkAgg')

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
                R = np.array(data['R'])
                T = np.array(data['T'])
                s = np.array(data['s'])
                RTS.append([R, T, s])
            except yaml.YAMLError as exc:
                print(exc)
    return RTS

# step 4. apply R, T, s to each image in realsense depth folder
# transform image give the rotation matrix R, translation matrix T and scale s
def transform_img(transform_image, R, T, scale):
    print("starting normal calib")
    # Load the reference image
    # transform = np.load("color.npy")
    transform_image = transform_image.astype(np.float32)
    # Define the padding size (top, bottom, left, right)
    print("image shape:", transform_image.shape)
    if (transform_image.shape[1]-transform_image.shape[0]>=0):
        top = 0 #abs(transform_image.shape[1]-transform_image.shape[0])
        bottom = abs(transform_image.shape[1]-transform_image.shape[0])
        left = 0
        right = 0
    else: 
        # we will check later how to deal with a tall image
        print("it's a tall image")
        #exit(1)
        top = 0 #abs(transform_image.shape[1]-transform_image.shape[0])
        bottom = 0#abs(transform_image.shape[1]-transform_image.shape[0])
        left = 0
        right = 0 #abs(transform_image.shape[1]-transform_image.shape[0])
    #Zero-pad the image using cv2.copyMakeBorder
    transform_image = cv.copyMakeBorder(transform_image, top, bottom, left, right, cv.BORDER_CONSTANT, value=-1)
    #plt.imshow(transform_image)
    #plt.show()
    # Get the shape of the reference image
    transform = transform_image
    height, width = transform.shape

    # Create a grid of coordinates for the reference image
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.vstack((x.flatten(), y.flatten()))  # Shape: (2, N), where N = height * width

    # Step 1: Apply scaling
    coords_scaled = coords

    # Step 2: Apply rotation and translation
    coords_transformed = np.dot(R, coords_scaled) + T[:, np.newaxis]

    # Reshape the transformed coordinates back to the image shape
    x_transformed = coords_transformed[0, :].reshape(height, width)
    y_transformed = coords_transformed[1, :].reshape(height, width)

    # Resize the distance image using interpolation
    #resized_distance = cv.resize(transformed_reference, (new_width, new_height), interpolation=cv.INTER_LINEAR)
    transformed_reference = map_coordinates(transform_image, [y_transformed, x_transformed], order=1, mode='constant', cval=np.nan)
    transformed_reference = cv.resize(transformed_reference, (int(transform.shape[0]/scale), int(transform.shape[1]/scale)), interpolation=cv.INTER_AREA)

    return transformed_reference

def transform_image_layered(RTS=[[],[],[]], max_dist=6, depth_ori=None, padding = True):
    print("starting multi-layer calib")
    background = transform_img(depth_ori, RTS[0][0], RTS[0][1], RTS[0][2])
    if not padding:
        background[(background<6.5) | (background>6.5)] = np.nan
    # calib for each distance range
    for i in range(max_dist, 0, -1):
        ind = i
        depth_copy = copy.copy(depth_ori)
        #1.1. make all areas outside the range nan
        min_ind = depth_ori<(ind-0.5)*1000
        max_ind = depth_ori>=(ind+0.5)*1000
        if ind==1: 
            min_ind = depth_ori<(0)*1000
        if ind==max_dist:
            max_ind = depth_ori>=(ind+42424242)*1000
        depth_copy[min_ind]=np.nan
        depth_copy[max_ind]=np.nan
        
        #get RTS and transform
        R, T, s = RTS[i-1]
        transformed_image1 = transform_img(depth_copy, R, T, s)

        mask = ~np.isnan(transformed_image1)
        background[mask] = transformed_image1[mask]
    return background
    


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

    # step 3: apply R, T, s to each image in realsense depth folder
    for image_name in transform_image_names:
        print("name of the image:", image_name)
        image = np.load(image_name)
        image = image.astype(np.float32)
        image = transform_image_layered(RTS, max_dis, image, padding=True)
        plt.imshow(image)
        plt.show()
        break
        # plt.imshow(image)
        # plt.show()
        # step 4: save the transformed images in a new folder (depth_senxor_m08)


