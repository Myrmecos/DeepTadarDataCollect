import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import yaml
'''
Green Light Position Determination
A class to determine the relative position between camera and green light
'''
class GLPosition():

    '''
    image_widht and image_height is the width and height of images
    taken by our camera.
    The width and height will be used to determine the relative angle between camera and green light.
    camera_param_path is the path to yaml file containing camera intrinsic matrix and distortion coefficient
    '''
    def __init__(self, image_width=1280, image_height=1024, camera_param_path = "camparam.yaml"):
        
        
        self.image_width = image_width
        self.image_height = image_height
        self.get_camera_intrinsic_distortion(camera_param_path)

    '''
    task: load camera intrinsic from a yaml file
    '''
    def get_camera_intrinsic_distortion(self, yaml_file_name):
        with open(yaml_file_name, 'r') as file:
            contents = yaml.safe_load(file)
    
        self.IM = np.matrix(contents["intrinsic_matrix"])
        self.distort = np.matrix(contents["distortion_coefficient"])

        self.lower_color = np.matrix(contents["lower_color"])
        self.upper_color = np.matrix(contents["upper_color"])

    '''
    Task: given the image that contains a green dot
    report its center coordinate in the image
    input: image
    output, tuple (x, y). x and y can be float
    '''
    def find_green_light(self, image):
        mask = cv.inRange(image, self.lower_color, self.upper_color)
        plt.imshow(mask)
        #plt.show()
        #return
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours: 
            largest_contour = max(contours, key=cv.contourArea)
            M = cv.moments(largest_contour)
            if M["m00"] !=0:
                cx = M["m10"]/M["m00"]
                cy = M["m01"]/M["m00"]
                center = (cx, cy)
            else:
                center = None
        else: 
            print("found none")
            center=None
        return center
    '''
    Task: given a position in the image
    return its position (coordinate) relative to center
    input: pos = [x_coordinate, y_coordinate]
    output: tuple (x, y). x and y can be float
    '''
    def pos_relative_to_center(self, pos):
        relative_x = pos[0] - self.image_width/2
        relative_y = pos[1] - self.image_height/2
        return (relative_x, relative_y)

    '''
    Task: given a pixel position to image center
    calculate the angle differnence between camera direction and green light
    input: relative position of light to camera, 2-tuple (x_dev_from_center, y_dev_from_center)
    output: relative position of light to camera, 2-tuple (x_angle_from_center, y_angle_from_center)
    '''
    def get_GL_angle_relative(self, pixel_pos):
        # first, prepare the pixel coordinate
        x, y = pixel_pos
        pts = np.array([[[x, y]]], dtype=np.float32)

        #print("position relative to center before un-distortion:", pts[0][0])
        # then, undistort and normalize coordinates
        undistorted_pts = cv.undistortPoints(pts, self.IM, self.distort)
        #print("position of circle center after un-distortion:", undistorted_pts[0][0])
        x_norm, y_norm = undistorted_pts[0][0]

        # finally, get the tangent value of each side
        angle_x = np.arctan2(x_norm, 1)
        angle_y = np.arctan2(y_norm, 1)

        return (angle_x, -angle_y)




# image = cv.imread("img_dir/0.png")
# glp = GLPosition()
# pos = glp.find_green_light(image)
# #rel_pos = glp.pos_relative_to_center(pos)
# pos = (1024, 1280)
# print("relative position to center: ", pos)
# glp.get_camera_intrinsic_distortion("camparam.yaml")
# print("distortion coefficient: \n", glp.distort)
# print("intrinsic matrix: \n", glp.IM)
# print("angle relative to camera center: ", glp.get_GL_angle_relative(rel_pos))