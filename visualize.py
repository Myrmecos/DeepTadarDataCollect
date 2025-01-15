import numpy as np
import os
import time
import cv2 as cv
#import matplotlib.pyplot as plt
 
base_path = "RawData/exp03/"
for dir in os.listdir(base_path):
    if ("meta.log" == dir):
        continue
    dirname = base_path+dir
    #print(dirname)
    path=base_path+dir+"/"+os.listdir(dirname)[5]
    image = np.load(path)
    # plt.imshow(image)
    # plt.title(dir)
    # plt.show()
    #image = np.rint(image)
    print(path, image)
    cv.imshow(path, image)
    cv.waitKey()
    
    # image = np.load("RawData/exp01/"+dir+"/"+os.listdir(dirname)[1])
    # print(image.shape)
    # plt.imshow(image)
    # plt.title(dir)
    # plt.show()
    print("=============================================================")