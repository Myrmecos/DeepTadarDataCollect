import os
import yaml
import numpy as np
import cv2 as cv
import argparse
from scipy.ndimage import map_coordinates
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

sensor_size_dic = {"seek_thermal":(150, 200), "senxor_m16":(120, 160), "senxor_m08":(62, 80), "MLX":(24, 32)}

class Aligner:

    #TODO: align a depth image to a thermal image give the R,T,scale
    RTS_base_dir = "MSC/3calibParams/"
    #init: takes the sensor type and make
    def __init__(self, sensor_name):
        self.sensor_name = sensor_name
        self.load_RTS()
        self.image_shape = sensor_size_dic[sensor_name]
        
    #load R, T, scale in the form: 1:[R1, T1, s1], 2:[R2, T2, s2], ...
    def load_RTS(self):
        all_files = os.listdir(self.RTS_base_dir+self.sensor_name)
        dist_list = []
        dist_RTS_dic = {}
        for i in all_files:
            yaml_file_name = self.RTS_base_dir+self.sensor_name+"/"+i
            dist = i.split(".")[:-1]
            dist = ".".join(dist)
            dist_list.append(int(dist))
            dist_RTS_dic[int(dist)] = self._load_RTS(yaml_file_name)
        self.RTS_dic = dist_RTS_dic
        dist_list.sort()
        self.dist_list = dist_list

    #helper method for loading R,T,scale from one yaml file
    def _load_RTS(self, yaml_file_name):
        with open(yaml_file_name, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                R = np.array(data['R'])
                T = np.array(data['T'])
                s = np.array(data['s'])
                return (R, T, s)
            except yaml.YAMLError as exc:
                print(exc)
                return None
            
    #apply R, T, s to each image in realsense depth folder
    # transform image give the rotation matrix R, translation matrix T and scale s
    def transform_img(self, transform_image, R, T, scale):
        print("starting normal calib")
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
        transform_image = cv.copyMakeBorder(transform_image, top, bottom, left, right, cv.BORDER_CONSTANT, value=np.nan)
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

    
    # takes a depth image and transform it in multiple layers
    def transform_image_layered(self, depth_ori=None, padding = True):
        depth_ori = depth_ori.astype(np.float32)
        add_padding(depth_ori, right=50, bottom=50)
        print("=============starting multi-layer calib. will include multiple normal calibs==============")
        R, T, scale = self.RTS_dic[self.dist_list[0]]
        background = self.transform_img(depth_ori, R, T, scale)
        if not padding:
            background[background==background] = np.nan
        # calib for each distance range
        for i in range(len(self.dist_list)):
            ind = self.dist_list[i]
            depth_copy = copy.copy(depth_ori)
            #1.1. make all areas outside the range nan
            min_ind = depth_ori<(ind-0.5)*1000
            max_ind = depth_ori>=(ind+0.5)*1000
            if i==0: 
                min_ind = depth_ori<(0)*1000
            if i==len(self.dist_list)-1:
                max_ind = depth_ori>=(ind+42424242)*1000
            depth_copy[min_ind]=np.nan
            depth_copy[max_ind]=np.nan
            
            #get RTS and transform
            R, T, s = self.RTS_dic[ind]
            transformed_image1 = self.transform_img(depth_copy, R, T, s)

            mask = ~np.isnan(transformed_image1)
            background[mask] = transformed_image1[mask]
        return background[0:self.image_shape[0], 0:self.image_shape[1]]

    def transform_images_batch(self, depth_ori_list):
        transformed_images = []
        for depth_ori in depth_ori_list:
            transformed_images.append(self.transform_image_layered(depth_ori))
        return transformed_images

# add padding to the image. avoid clipping during transformation
def add_padding(image, top = 0, bottom = 0, left = 0, right = 0):
    # Define the padding size (top, bottom, left, right)
    # print("image shape:", image.shape)
    image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=np.nan)
    return image

sensor_size_dic = {"seek_thermal":(150, 200), "senxor_m16":(120, 160), "senxor_m08":(62, 80), "MLX":(24, 32)}

if __name__ == "__main__":

    #1. load image
    transform_image = np.load("/media/zx/zx-data/RawData/exp06/realsense_depth/1737684595.6384075.npy")
    transform_image1 = np.load("/media/zx/zx-data/RawData/exp06/realsense_depth/1737684596.0393853.npy")
    print(transform_image.shape)
    # transform_image = transform_image.astype(np.float32) #make sure image is of np.float32 type
    # transform_image1 = transform_image1.astype(np.float32) #make sure image is of np.float32 type

    # TODO: add padding to right and bottom to avoid clipping
    # transform_image = add_padding(transform_image, right=50, bottom=50)
    # transform_image1 = add_padding(transform_image1, right=50, bottom=50)
    print(transform_image.shape)

    # TODO: initialize an Aligner
    a = Aligner("senxor_m16")

    # TODO: transform one image
    # padding: use the transform parameters at furthest calibrated distance to fill empty space
    #transform_image = a.transform_image_layered(transform_image, padding=True)
    transform_images = a.transform_images_batch([transform_image, transform_image1, transform_image, transform_image1])

    #4. visualize image
    plt.imshow(transform_images[0])
    plt.show()
    plt.imshow(transform_images[1])
    plt.show()
    
    
    