import open3d as o3d 
import numpy as np 
import matplotlib.pyplot as plt
import cv2

def readPcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        print("readPcd received empty point cloud")
        return
    return pcd

#pcd = readPcd("/home/astar/dart_ws/single_scene_calibration/0.pcd")
pcd = readPcd("/home/astar/dart_ws/single_scene_calibration/0.pcd")
# Save the cropped point cloud
o3d.io.write_point_cloud("res.pcd", pcd)
