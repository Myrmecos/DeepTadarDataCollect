import numpy as np
import os
import time
#import cv2 as cv
import matplotlib.pyplot as plt
 
base_path = "RawData/exp09/"
for dir in os.listdir(base_path):
    if ("meta.log" == dir):
        continue
    dirname = base_path+dir
    #print(dirname)
    path=base_path+dir+"/"+os.listdir(dirname)[0]
    image = np.load(path)
    #image = np.load("RawData/exp03/MLX/1736941282.512067.npy")
    plt.imshow(image)
    plt.title(dir)
    plt.show()
    #image = np.rint(image)
    print(path, image)
    # cv.imshow(path, image)
    # cv.waitKey()
    
    # image = np.load("RawData/exp01/"+dir+"/"+os.listdir(dirname)[1])
    # print(image.shape)
    # plt.imshow(image)
    # plt.title(dir)
    # plt.show()
    print("=============================================================")