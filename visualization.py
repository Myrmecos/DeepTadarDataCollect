import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


dirbase = "/media/zx/zx-data/RawData/exp06/"
#dirbase = "RawData/exp01/"
for dir in os.listdir(dirbase):
    if ("meta.log" == dir):
        continue
    dirname = dirbase+dir
    #print(dirname)
    print(len(os.listdir(dirname)))
    image = np.load(dirbase+dir+"/"+os.listdir(dirname)[1])
    print(image.shape)
    # plt.imshow(image)
    # plt.title(dir)
    # plt.show()
    # image = np.load("RawData/exp01/"+dir+"/"+os.listdir(dirname)[1])
    # print(image.shape)
    # plt.imshow(image)
    # plt.title(dir)
    # plt.show()
    print(dirname)
    print("=============================================================")