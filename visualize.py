import numpy as np
import os
import time
import matplotlib.pyplot as plt
 
for dir in os.listdir("RawData/exp01"):
    if ("meta.log" == dir):
        continue
    dirname = "RawData/exp01/"+dir
    #print(dirname)
    image = np.load("RawData/exp01/"+dir+"/"+os.listdir(dirname)[20])
    plt.imshow(image)
    plt.title(dir)
    plt.show()
    # image = np.load("RawData/exp01/"+dir+"/"+os.listdir(dirname)[1])
    # print(image.shape)
    # plt.imshow(image)
    # plt.title(dir)
    # plt.show()
    print("=============================================================")
    print(dirname)
    print(image)

    if (dir=="MLX"):
        print(image)