import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
import argparse

def normalize_thermal_array(array, lower = 20, upper = 25):
    array = array.astype(np.uint8)
    #senxor_temperature_map_m08 = cv2.normalize(senxor_temperature_map_m08, None, 0, 255, cv2.NORM_MINMAX)
    array = np.clip(array, lower, upper)
    array = ((array-lower)/(upper-lower)*255).astype(np.uint8)
    return array
        

dirbase = "/media/zx/zx-data/RawData/exp06/"
sensors = ["MLX", "realsense_color", "realsense_depth", "seek_thermal", "senxor_m08", "senxor_m08_1"]

parser = argparse.ArgumentParser()
parser.add_argument("--dirbase", type=str, help="the base directory of the dataset") #dirbase

args = parser.parse_args()

dirbase = args.dirbase

# load the dirnames of each sensor type
mlx_dir = dirbase+sensors[0]
realsense_color_dir = dirbase+sensors[1]
realsense_depth_dir = dirbase+sensors[2]
seek_thermal_dir = dirbase+sensors[3]
senxor_m08_dir = dirbase+sensors[4]
senxor_m08_1_dir = dirbase+sensors[5]

num_of_frames = len(os.listdir(mlx_dir))

# load the images of each sensor type
mlx_images = os.listdir(mlx_dir)
realsense_color_images = os.listdir(realsense_color_dir)
realsense_depth_images = os.listdir(realsense_depth_dir)
seek_thermal_images = os.listdir(seek_thermal_dir)
senxor_m08_images = os.listdir(senxor_m08_dir)
senxor_m08_1_images = os.listdir(senxor_m08_1_dir)

for i in range(0, num_of_frames):
    MLX_temperature_map = np.load(mlx_dir+"/"+mlx_images[i])
    realsense_color_image = np.load(realsense_color_dir+"/"+realsense_color_images[i])
    realsense_depth_image = np.load(realsense_depth_dir+"/"+realsense_depth_images[i])
    seek_camera_frame = np.load(seek_thermal_dir+"/"+seek_thermal_images[i])
    senxor_temperature_map_m08 = np.load(senxor_m08_dir+"/"+senxor_m08_images[i])
    senxor_temperature_map_m08_1 = np.load(senxor_m08_1_dir+"/"+senxor_m08_1_images[i])


    
    if realsense_depth_image is not None:
        realsense_depth_image = cv2.applyColorMap(cv2.convertScaleAbs(realsense_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #realsense_color_image = cv2.resize(realsense_color_image, (320, 240))
        realsense_depth_image = cv2.resize(realsense_depth_image, (320, 240))  
    else:
        realsense_depth_image = np.zeros((240, 320, 3), dtype=np.uint8)

    if realsense_color_image is not None:
        realsense_color_image = cv2.resize(realsense_color_image, (320, 240), interpolation=cv2.INTER_NEAREST)
    else:
        realsense_color_image = np.zeros((240, 320, 3), dtype=np.uint8)

    if seek_camera_frame is not None:
        print(f"seek min: {np.min(seek_camera_frame)}; max: {np.max(seek_camera_frame)}")

        # if seek_camera_frame.shape[0] > 201:
        #     seek_camera_frame = cv2.resize(argb2bgr(seek_camera_fqrame), (320, 240))
        # else:
        #     seek_camera_frame = np.flip(np.flip(cv2.resize(argb2bgr(seek_camera_frame), (320, 240)),0),1)
        # seek_camera_frame = np.flip(seek_camera_frame, 0)
        # seek_camera_frame = np.flip(seek_camera_frame, 1)
        seek_camera_frame = seek_camera_frame.astype(np.uint8)
        seek_camera_frame = cv2.normalize(seek_camera_frame, None, 0, 255, cv2.NORM_MINMAX)
        seek_camera_frame = cv2.resize(seek_camera_frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        seek_camera_frame = cv2.applyColorMap(seek_camera_frame, cv2.COLORMAP_JET)
    else:
        seek_camera_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
    if MLX_temperature_map is not None:
        MLX_temperature_map = MLX_temperature_map.reshape(24, 32)
        #MLX_temperature_map = np.flip(MLX_temperature_map, 0)
        # MLX_temperature_map = np.flip(MLX_temperature_map, 1)
        #MLX_temperature_map = mlx_sensor.SubpageInterpolating(MLX_temperature_map)
        MLX_temperature_map = MLX_temperature_map.astype(np.uint8)
        # min_temp, max_temp = np.min(MLX_temperature_map), np.max(MLX_temperature_map)
        MLX_temperature_map = cv2.normalize(MLX_temperature_map, None, 0, 255, cv2.NORM_MINMAX)
        MLX_temperature_map = cv2.resize(MLX_temperature_map, (320, 240), interpolation=cv2.INTER_NEAREST)
        MLX_temperature_map = cv2.applyColorMap(MLX_temperature_map, cv2.COLORMAP_JET)
    else:
        MLX_temperature_map = np.zeros((240, 320, 3), dtype=np.uint8)
        
    if senxor_temperature_map_m08 is not None:
        print(f"senxor_m08 min: {np.min(senxor_temperature_map_m08)}; max: {np.max(senxor_temperature_map_m08)}")

        #senxor_temperature_map_m08 = senxor_temperature_map_m08.reshape(num_cols_m08, num_rows_m08)
        # senxor_temperature_map_m08 = np.flip(senxor_temperature_map_m08, 0)
        senxor_temperature_map_m08 = senxor_temperature_map_m08.astype(np.uint8)
        #senxor_temperature_map_m08 = cv2.normalize(senxor_temperature_map_m08, None, 0, 255, cv2.NORM_MINMAX)
        senxor_temperature_map_m08 = normalize_thermal_array(senxor_temperature_map_m08, 10, 25)
        senxor_temperature_map_m08 = cv2.resize(senxor_temperature_map_m08, (320, 240), interpolation=cv2.INTER_NEAREST)
        senxor_temperature_map_m08 = cv2.applyColorMap(senxor_temperature_map_m08, cv2.COLORMAP_JET)
    else:
        senxor_temperature_map_m08 = np.zeros((240, 320, 3), dtype=np.uint8)

    if senxor_temperature_map_m08_1 is not None:
        print(f"senxor_m16 min: {np.min(senxor_temperature_map_m08_1)}; max: {np.max(senxor_temperature_map_m08_1)}")

        #senxor_temperature_map_m08_1 = senxor_temperature_map_m08_1.reshape(num_cols_m08_1, num_rows_m08_1)
        # senxor_temperature_map_m08_1 = np.flip(senxor_temperature_map_m08_1, 0)
        #senxor_temperature_map_m08_1 = senxor_temperature_map_m08_1.astype(np.uint8)
        #senxor_temperature_map_m08_1 = cv2.normalize(senxor_temperature_map_m08_1, None, 0, 255, cv2.NORM_MINMAX)
        senxor_temperature_map_m08_1 = normalize_thermal_array(senxor_temperature_map_m08_1, 15, 29)
        senxor_temperature_map_m08_1 = cv2.resize(senxor_temperature_map_m08_1, (320, 240), interpolation=cv2.INTER_NEAREST)
        senxor_temperature_map_m08_1 = cv2.applyColorMap(senxor_temperature_map_m08_1, cv2.COLORMAP_JET)
    else:
        senxor_temperature_map_m08_1 = np.zeros((240, 320, 3), dtype=np.uint8)
        
    #print(realsense_depth_image.shape, realsense_color_image.shape, seek_camera_frame.shape,  senxor_temperature_map_m08.shape, senxor_temperature_map_m08_1.shape, MLX_temperature_map.shape)
    print("=============================================================")
    interm1 = np.concatenate((realsense_depth_image, realsense_color_image, seek_camera_frame), axis=1)
    interm2 = np.concatenate((senxor_temperature_map_m08, MLX_temperature_map, senxor_temperature_map_m16), axis=1)
    final_image = np.concatenate((interm1, interm2), axis=0)
    #print(final_image)
    cv2.imshow("Final Image", final_image)
    cv2.waitKey(100)