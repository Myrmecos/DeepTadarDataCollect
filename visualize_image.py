import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import argparse

matplotlib.use('TkAgg')


# dirbase = "/media/zx/zx-data/RawData/exp06/"
parser = argparse.ArgumentParser()
parser.add_argument("--dirbase", type=str, help="the base directory of the dataset") #dirbase
args = parser.parse_args()
dirbase = args.dirbase

#dirbase = "RawData/exp01/"
for dir in os.listdir(dirbase):
    if ("meta.log" == dir or "depth_map" == dir):
        continue
    dirname = dirbase+dir
    #print(dirname)
    print(len(os.listdir(dirname)))
    image = np.load(dirbase+dir+"/"+os.listdir(dirname)[0])
    print(image.shape)
    plt.imshow(image)
    plt.title(dir)
    plt.show()
    # image = np.load("RawData/exp01/"+dir+"/"+os.listdir(dirname)[1])
    # print(image.shape)
    # plt.imshow(image)
    # plt.title(dir)
    # plt.show()
   
    print(dirname)
    print("=============================================================")