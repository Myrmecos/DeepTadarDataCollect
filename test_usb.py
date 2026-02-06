#!/usr/bin/env python
import rospy
import message_filters
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import threading
import copy
import yaml
from dart_lidar_image_utils import imagelidaraligner, imageprocessor
import traceback
import matplotlib.pyplot as plt
import time
import serial
import struct
import traceback
import math

MAX_PCD_MESSAGES = 50 # how many pcd messages we want to pool for processing
NUM_OF_POINTS = 40 #how many number of points we want to cluster for the target
CAMERA_PARAM_PATH = "/home/astar/dart_ws/src/lidar_image_align/calib/calib.yaml"
BAUD_RATE=115200
PORTX="/dev/ttyACM0"
TIMEX=5

def make_data(yaw, pitch=0.0, found=0, shoot_or_not=0, done_fitting=0, patrolling=0, updated=0, base_dis=0.0, checksum=0):
    # Pack the data according to the struct format
    # Constants
    SOF = 0xA3  # Start of Frame marker
    data = struct.pack(
        '<BffBBBBBf',  # Format: < for little-endian, B=uint8, f=float32
        SOF,
        yaw,
        pitch,
        found,
        shoot_or_not,
        done_fitting,
        patrolling,
        updated,
        base_dis
    )
    return data

def make_data0(yaw, pitch=0.0, found=0, shoot_or_not=0, done_fitting=0, patrolling=0, updated=0, base_dis=0.0, checksum=0):
    # Pack the data according to the struct format
    # Constants
    SOF = 0xA3  # Start of Frame marker
    data = struct.pack(
        '<BffBBBBBf',  # Format: < for little-endian, B=uint8, f=float32
        SOF,
        0xFF,
        pitch,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        base_dis
    )
    return data

'''
send the yaw angle and distance via uart
'''
def send_via_uart(angleX, distance):
    # data = struct.pack("ff", angleX, distance)
    data = make_data(yaw=angleX, base_dis=distance)
    # Print the packet data in hex format for debugging
    print("Packet data (hex):", data.hex())
    # Calculate and print the size of the packet
    packet_size = len(data)
    print("Packet size (bytes):", packet_size)
    try:
        with serial.Serial(PORTX, BAUD_RATE, timeout=TIMEX) as ser:
            ser.write(data)
            print("angle and distance sent")
    except Exception as e:
        traceback.print_exc()
        print("error occurred: ", e)


while True:
    for i in range(1, 100, 1):
        send_via_uart(i, 100-i)
        time.sleep(0.01)
