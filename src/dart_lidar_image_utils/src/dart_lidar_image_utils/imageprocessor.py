import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import yaml
import math
'''
Green Light Position Determination
A class to determine the relative position between camera and green light
'''
class GLPosition():
    
    '''Debug only'''
    def _show_image(self, image, title="image"):
        height, width = image.shape[:2]

        # Resize to half size
        resized_image = cv.resize(
            image, 
            (width // 2, height // 2),  # New dimensions (half of original)
            interpolation=cv.INTER_LINEAR  # Or INTER_AREA for downscaling
        )

        cv.imshow(title, resized_image)
        cv.waitKey(0)
    
    '''Debug only'''
    def _compare_images(self, image1, image2):
        fig, axes = plt.subplots(1, 2, figsize=(8, 8))  # 2x2 grid

        axes[0].imshow(image1)
        axes[0].set_title("Image 1")
        axes[0].axis('off')

        axes[1].imshow(image2)
        axes[1].set_title("Image 2")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()


    '''
    image_widht and image_height is the width and height of images
    taken by our camera.
    The width and height will be used to determine the relative angle between camera and green light.
    camera_param_path is the path to yaml file containing camera intrinsic matrix and distortion coefficient
    '''
    def __init__(self, image_width=2448, image_height=2048, camera_param_path = "camparam.yaml"):
        
        
        self.image_width = image_width
        self.image_height = image_height
        self.get_camera_intrinsic_distortion(camera_param_path)

    '''
    task: load camera intrinsic from a yaml file
    '''
    def get_camera_intrinsic_distortion(self, yaml_file_name):
        with open(yaml_file_name, 'r') as file:
            contents = yaml.safe_load(file)
    
        self.IM = np.matrix(contents['camera']["camera_matrix"]).reshape(3, 3)
        self.distort = np.matrix(contents['camera']["dist_coeffs"])

        self.lower_color = np.asarray(contents['colors']["lower_color"])
        self.upper_color = np.asarray(contents['colors']["upper_color"])

    '''
    given a list of contours
    find the contour that is roundest
    circularity: 4\pi area divided by perimeter squared
    for circle, circularity is exactly 1.
    '''
    def find_roundest_contour(self, contours):
        # Calculate circularity for each contour
        roundest_contour = None
        max_circularity = 0  # Circularity ranges from 0 (not round) to 1 (perfect circle)

        for contour in contours:
            area = cv.contourArea(contour)
            perimeter = cv.arcLength(contour, closed=True)
            
            # Avoid division by zero for tiny contours
            if perimeter == 0:
                continue
            
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # Update roundest contour if current one is more circular
            if circularity > max_circularity:
                max_circularity = circularity
                roundest_contour = contour
        return roundest_contour
    '''
    Task: given the image that contains a green dot
    report its center coordinate in the image
    input: image
    output, tuple (x, y). x and y can be float
    '''
    def find_green_light(self, image):
        mask = cv.inRange(image, self.lower_color, self.upper_color)
        cv.imwrite("mask_DEBUG.jpg", mask)
        print("DEBUG: lower color: ", self.lower_color)
        print("DEBUG: upper color: ", self.upper_color)
        #self._compare_images(image, mask)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours: 
            roundest_contour = self.find_roundest_contour(contours)
            M = cv.moments(roundest_contour)
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
    def get_GL_angle(self, pixel_pos):
        # first, prepare the pixel coordinate
        x, y = pixel_pos
        pts = np.array([[[x, y]]], dtype=np.float32)

        #print("position relative to center before un-distortion:", pts[0][0])
        # then, undistort and normalize coordinates
        undistorted_pts = cv.undistortPoints(pts, self.IM, self.distort)
        #print("position of circle center after un-distortion:", undistorted_pts[0][0])
        x_norm, y_norm = undistorted_pts[0][0]

        # finally, get the tangent value of each side
        angle_x = np.arctan2(x_norm, 1)*180/math.pi
        angle_y = np.arctan2(y_norm, 1)*180/math.pi

        return (angle_x, -angle_y)
    



if __name__=="__main__":
    #image = cv.imread("testImg/frame0000.jpg")
    image = cv.imread("/home/astar/Desktop/testing/test1_dotdot_rec.jpg")

    glp = GLPosition(camera_param_path="/home/astar/dart_ws/src/lidar_image_align/calib/calib.yaml")
    glp.lower_color = np.array([245, 245, 20])
    glp.upper_color = np.array([255, 255, 90])
    pos = glp.find_green_light(image)
    #rel_pos = glp.pos_relative_to_center(pos)
    print("pixel coord: ", pos)
    # glp.get_camera_intrinsic_distortion()
    # print("distortion coefficient: \n", glp.distort)
    # print("intrinsic matrix: \n", glp.IM)
    #print("angle relative to camera center: ", glp.get_GL_angle_relative(rel_pos))
    print(glp.get_GL_angle([1374.860012, 1083.353097]))

