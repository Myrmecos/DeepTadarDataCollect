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

'''
Read camera intrinsic, distortion, and extrinsic parameters from yaml file
@param yaml_file_name: the path of yaml file that contains camera parameters
@return IM: camera intrinsic matrix; distort: camera distortion coefficients; EM: camera extrinsic matrix
'''
def get_camera_intrinsic_distortion_extrinsic(yaml_file_name):
        with open(yaml_file_name, 'r') as file:
            contents = yaml.safe_load(file)
    
        IM = np.matrix(contents['camera']["camera_matrix"]).reshape((3, 3))
        distort = np.matrix(contents['camera']["dist_coeffs"])
        EM = np.matrix(contents['camera']['ex_matrix']).reshape((4, 4))

        return IM, distort, EM
        

MAX_PCD_MESSAGES = 50 # how many pcd messages we want to pool for processing
NUM_OF_POINTS = 40 #how many number of points we want to cluster for the target
CAMERA_PARAM_PATH = "/home/astar/dart_ws/src/lidar_image_align/calib/calib.yaml" # the path of yaml file that contains camera parameters
BAUD_RATE=115200 # baud rate for uart communication
PORTX="/dev/ttyACM0"
TIMEX=5
ok_to_send = True
ok_to_send_lock = threading.Lock()

# '''
# listening to messages from usb port
# if the message indicates we can start to send (the format of message: ?)
# we set the variable ok_to_send to true.
# this function is non-blocking
# '''
# def recv_uart():
#     thread = threading.Thread(target = _recv_uart)
#     thread.daemon = True
#     thread.start()

# '''
# blocking version of recv_uart
# '''
# def _recv_uart():
#     global ok_to_send, ok_to_send_lock

#     # Open serial port (adjust parameters as needed)
#     with serial.Serial(PORTX, BAUD_RATE, timeout=TIMEX) as ser:  # Change 'COM3' to your port
    
#         while True:
#             print("waiting...")
#             # Read one byte
#             received_byte = ser.read(1)

#             # Convert to boolean (assuming 0x00=False, 0x01=True)
#             bool_value = bool(int.from_bytes(received_byte, 'big')) if received_byte else None
#             if bool_value:
#                 with ok_to_send_lock:
#                     ok_to_send = True
#                 print(f"Received: {bool_value}")
#                 return
# #_recv_uart() #uncomment to wait for start signal


im, distort, em = get_camera_intrinsic_distortion_extrinsic(CAMERA_PARAM_PATH)
cam_rotor_em = imagelidaraligner.get_cam_rotor_matrix(CAMERA_PARAM_PATH)

class Listener:
    '''
    Initialize listener class by:
    1. initialize image subscriber, image variable (holding most recent image) and a lock protecting the image varialbe
    2. initialize point cloud subscriber, point cloud queue (holding a pre-defined number of point cloud message contents), and a lock protecting the queue
    3. initialize the processing classes to find green light position and find corresponding points
    '''
    def __init__(self):

        # for image
        ##  obtaining images
        self.bridge = CvBridge()
        image_sub = rospy.Subscriber('/hikrobot_camera/rgb', Image, self.image_callback)
        pc_sub = rospy.Subscriber('/livox/lidar', PointCloud2, self.pc_callback)
        self.image = None
        self.image_lock = threading.Lock()

        ## image processor for finding green light position
        self.glp = imageprocessor.GLPosition(camera_param_path = CAMERA_PARAM_PATH)
        # print("debug: ", glp.IM, glp.distort, glp.upper_color, glp.lower_color)

        # for point cloud
        ##  obtainging point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.point_queue = []
        self.pcd_lock = threading.Lock()
        ## processing point cloud
        self.ila = imagelidaraligner.ImageLidarAligner(em, im, num_of_points = NUM_OF_POINTS)

        # # for visualization
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window("Point Cloud", width=800, height=600)
        # self.first_frame = True
        #self.vis_pcd = o3d.geometry.PointCloud()
        #time.sleep(3)
        #self.periodic_callback(None)

        rospy.Timer(rospy.Duration(0.2), self.periodic_callback)
        
    '''
    callback function listening to incoming image messages
    it replaces self.image's previous variable with new image
    @param image_msg: the incoming image message
    '''
    def image_callback(self, image_msg):
        try:
            # Convert ROS Image to OpenCV (BGR8 format)
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            print("received image.")

            # store the image into self.image
            # since self.image will be accessed in another thread, we need to acquire lock before accessing it
            if not self.image_lock.acquire(blocking=False):
                rospy.loginfo("image locked in image_callback, skipping")
                return
            try:
                self.image = copy.deepcopy(cv_image)
            finally:
                self.image_lock.release()
            
            #rospy.loginfo("Image received and displayed")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    # task: read point cloud message and store data from last MAX_PCD_MESSAGES messages into self.pcd
    '''
    callback function listeneing to incoming point cloud message
    it plugs the point cloud into point_queue
    if the queue is overflown with the new point cloud
    we throw away the oldest point cloud
    '''
    def pc_callback(self, pc_msg):
        try:
            # Extract points from PointCloud2
            points = []
            for point in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append([point[0], point[1], point[2]])
            points = np.array(points, dtype=np.float64)
            # add to point queue
            self.point_queue.append(points)
            if (len(self.point_queue) > MAX_PCD_MESSAGES):
                self.point_queue=self.point_queue[1:]
            # prepare the point cloud that contains all recent points in queue
            points_in_last_half_second = np.concatenate(self.point_queue, axis=0)

            # lock and try to access pcd (since pcd will be accessed in another thread) 
            if not self.pcd_lock.acquire(blocking=False):
                rospy.loginfo("pcd locked in pc_callback, skipping")
                return
            try:
                self.pcd.points = o3d.utility.Vector3dVector(points_in_last_half_second)
            finally:
                self.pcd_lock.release()

        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {e}")
    
    # no longer used
    # directly displaying the image and visualize green light detection outcome
    def listen_image(self, myimg):
        lightpos = self.glp.find_green_light(myimg)
        print("green light position in image: ", lightpos)
        pos_rounded = [0, 0]
        if lightpos is not None:
            print(lightpos)
            pos_rounded[0] = round(lightpos[0])
            pos_rounded[1] = round(lightpos[1])
            lightpos=pos_rounded
            height, width = myimg.shape[:2]
            cv2.line(myimg, (0, lightpos[1]), (width-1, lightpos[1]), (255, 255, 255), 3)
            cv2.line(myimg, (lightpos[0], 0), (lightpos[0], height-1), (255, 255, 255), 3)
            myimg = cv2.cvtColor(myimg, cv2.COLOR_BGR2RGB)
            myimg = cv2.resize(
                myimg, 
                None, 
                fx=1/3,  # Scale factor for width
                fy=1/3,  # Scale factor for height
                interpolation=cv2.INTER_AREA  # Best for downscaling
            )
            cv2.imshow("Hikrobot Camera (detected)", myimg)
            cv2.waitKey(1)
        else: 
            cv2.imshow("Hikrobot Camera (empty detection)", myimg)
    
    # task: retrieve the most recent images (self.image) and point cloud (self.pcd)
    # do some operations on them
    '''
    callback function that is periodically called (period according to pre-defined interval time)
    it obtains image in self.image and point clouds in self.point_queue
    it then finds green light in the image and use the light's image coord to find corresponding points in point clouds.
    '''
    def periodic_callback(self, event):
        try:
            mypts = None
            myimg = None

            # Wait to acquire lock and obtain pts
            with self.pcd_lock:
                rospy.loginfo("=====Periodic callback accessed pcd======")
                # Example: Access pcd points (modify as needed)
                if len(self.pcd.points) > 0:
                    mypts = copy.deepcopy(self.pcd)
            # wait to acquire lock and obtain images
            with self.image_lock:
                rospy.loginfo("=====Periodic callback accessed image======")
                if self.image is not None:
                    myimg = copy.deepcopy(self.image)
            
            # Start processing if both mypts and myimg are not empty
            if mypts != None and myimg is not None:
                print("\n\n=====================================")
                # plt.imshow(myimg)
                # plt.show()
                # # example: show both image and point cloud
                # # show image
                print(myimg.shape)
                # cv2.imshow("Hikrobot Camera", myimg)
                # cv2.waitKey(1)

                # # show point cloud
                # self.vis_pcd.points = mypts.points
                # if self.first_frame:
                #     self.vis.add_geometry(self.vis_pcd)
                #     self.first_frame = False
                # else:
                #     self.vis.update_geometry(self.vis_pcd)
                # self.vis.poll_events()
                # self.vis.update_renderer()
                lightpos = self.glp.find_green_light(myimg)
                print("green light position in image: ", lightpos)
                if lightpos is not None:
                    closest_pts,pts_2d, valid_pts,_ = self.ila.reportPoints1(lightpos, mypts)
                    closest_pts_rotor = self.ila.to_rotor_coord(closest_pts, cam_rotor_em)
                    
                    # get angle
                    angle = self.ila.to_degree(closest_pts_rotor)
                    angle1 = self.glp.get_GL_angle(lightpos)[0]
                    D=25; d=0.704 # D is the distance from camera to green light; d is the distance from camera to rotation center (horizontal dist)
                    angle1 = math.atan(D*math.tan(angle1*math.pi/180)/(D+d))*180/math.pi
                    hori_dist = self.ila.calc_horizontal_dist(closest_pts_rotor)

                    self.show_img_and_target(myimg, lightpos, angle, angle1, hori_dist)
                    send_via_uart(angle, hori_dist)
                    
                    #save_im_pcd(image=myimg, point_cloud=mypts)
                    closest_pts = imagelidaraligner.array_to_pointcloud(closest_pts)
                    valid_pts = imagelidaraligner.array_to_pointcloud(valid_pts)
                    imagelidaraligner.visualize_point_clouds(valid_pts, closest_pts)
                else:
                    self.show_img(myimg)
        except Exception as e:
            rospy.logerr(f"Error in periodic callback: {e}")
            traceback.print_exc()

    '''
    visualize img and target info
    @param myimg: the image collected by camera that contains target
    @param lightpos: the detected pixel coordinate of light
    @param angle: the detected yaw angle
    @param dist: the distance of target from rotation center
    '''
    def show_img_and_target(self, myimg, lightpos, angle, angle1, dist):
        height, width = myimg.shape[:2]
        pos_rounded = [0, 0]
        pos_rounded[0] = round(lightpos[0])
        pos_rounded[1] = round(lightpos[1])
        lightpos=pos_rounded
        cv2.line(myimg, (0, lightpos[1]), (width-1, lightpos[1]), (255, 255, 255), 3)
        cv2.line(myimg, (lightpos[0], 0), (lightpos[0], height-1), (255, 255, 255), 3)
        myimg = cv2.cvtColor(myimg, cv2.COLOR_BGR2RGB)
        myimg = cv2.putText(myimg, f"yaw directly from image: {angle1-angle}", (lightpos[0]+30, lightpos[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        myimg = cv2.putText(myimg, f"yaw: {angle}", (lightpos[0]+30, lightpos[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        myimg = cv2.putText(myimg, f"dist: {dist}", (lightpos[0]+30, lightpos[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2) # for center
        print(f"DEBUG: distance is {dist}")
        cv2.line(myimg, (int(im[0,2]), 0), (int(im[0,2]), height-1), (255, 255, 0), 3)
        myimg = cv2.resize(
            myimg, 
            None, 
            fx=1/3,  # Scale factor for width
            fy=1/3,  # Scale factor for height
            interpolation=cv2.INTER_AREA  # Best for downscaling
        )
        cv2.imshow("Hikrobot Camera", myimg)
        cv2.waitKey(1)
    
    '''
    showing image in half of original size
    this image showing function is used when the original image is too large to fit into screen
    '''
    def show_img(self, myimg):
        myimg = cv2.resize(
            myimg, 
            None, 
            fx=1/3,  # Scale factor for width
            fy=1/3,  # Scale factor for height
            interpolation=cv2.INTER_AREA  # Best for downscaling
        )
        cv2.imshow("Hikrobot Camera", myimg)
        cv2.waitKey(1)

'''
save image and point cloud to specific pos for debugging
'''
def save_im_pcd(image, point_cloud):
    cv2.imwrite("/home/astar/dart_ws/src/dart_lidar_image_utils/src/dart_lidar_image_utils/test.jpg", image)

    # Write PCD file
    o3d.io.write_point_cloud("/home/astar/dart_ws/src/dart_lidar_image_utils/src/dart_lidar_image_utils/test.pcd", point_cloud, write_ascii=True)

'''
compose data to be sent via uart through usb port
@param yaw: yaw angle
@param base_dis: distance from rotational center
'''
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

'''
send the yaw angle and distance via uart
'''
def send_via_uart(angleX, distance):
    with ok_to_send_lock:
        if not ok_to_send:
            return
    # data = struct.pack("ff", angleX, distance)
    data = make_data(yaw=angleX, base_dis=distance)
    try: 
        with serial.Serial(PORTX, BAUD_RATE, timeout=TIMEX) as ser:
            ser.write(data)
            print("angle and distance sent")
    except Exception as e:
        traceback.print_exc()
        print("error occurred: ", e)

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener = Listener()
    rospy.spin()
