import cv2 as cv
import numpy as np

img = cv.imread("grid.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img1 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
img1 = cv.resize(img1, (img1.shape[1]//2, img1.shape[0]//2))
cv.imshow("image ori", img)
cv.waitKey()
cv.imshow("image transformed", img1)
cv.waitKey()

img_arr_ori = np.array(img)
img_arr_transf = np.array(img1)

np.save("ori.npy", img_arr_ori)
np.save("trans.npy", img_arr_transf)