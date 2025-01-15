import serial
import time
import ast
import numpy as np
import cv2
import sys
import os
import signal
import logging
import cv2 as cv
from pprint import pprint
import argparse
import pyrealsense2 as rs
from PIL import Image

#initialization of realsense camera
# rs.log_to_file(rs.log_severity.debug, file_path="librealsense.log")
# rs.log_to_console(rs.log_severity.debug)
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
#below for testing only ====
# device = profile.get_device()
# device.hardware_reset()
#above for testing only ====
align_to = rs.stream.color
align = rs.align(align_to)
cnt = 0
while (True):
    print("looping: ", cnt)
    cnt+=1
    #obtain frames from realsense camera
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        print("No frames received")
        exit(-1)
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    #display frames
    realsense_color_image = cv2.resize(color_image, (320, 240))
    realsense_depth_image = cv2.resize(depth_image, (320, 240)) 
    realsense_depth_image = cv2.applyColorMap(cv2.convertScaleAbs(realsense_depth_image, alpha=0.03), cv2.COLORMAP_JET)

    realsense_color_image = cv2.resize(realsense_color_image, (320, 240))
    print("size of color img: ", realsense_color_image.shape)
    #final_img = np.concatenate(realsense_color_image, realsense_depth_image)
    print(type(realsense_color_image))
    cv2.imshow('RealSense', realsense_color_image)
    key = cv.waitKey(1)
    if key in [ord("q"), ord('Q'), 27]:
        break



