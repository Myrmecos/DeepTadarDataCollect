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

def get_camera_intrinsic_distortion_extrinsic(yaml_file_name):
        with open(yaml_file_name, 'r') as file:
            contents = yaml.safe_load(file)
    
        IM = np.matrix(contents['camera']["camera_matrix"]).reshape((3, 3))
        distort = np.matrix(contents['camera']["dist_coeffs"])
        EM = np.matrix(contents['camera']['ex_matrix']).reshape((4, 4))

        return IM, distort, EM
        

MAX_PCD_MESSAGES = 6
CAMERA_PARAM_PATH = "/home/astar/dart_ws/src/livox_camera_calib/config/calib_ori.yaml"

im, distort, em = get_camera_intrinsic_distortion_extrinsic(CAMERA_PARAM_PATH)
# print(im)
# print(distort)
# print(em)

class Listener:
    def __init__(self):
        # for image
        ##  obtaining images
        self.bridge = CvBridge()
        image_sub = rospy.Subscriber('/hikrobot_camera/rgb', Image, self.image_callback)
        pc_sub = rospy.Subscriber('/livox/lidar', PointCloud2, self.pc_callback)
        self.image = None
        self.image_lock = threading.Lock()
        ##  processing images
        self.glp = imageprocessor.GLPosition(camera_param_path = CAMERA_PARAM_PATH)
        # print("debug: ", glp.IM, glp.distort, glp.upper_color, glp.lower_color)

        # for point cloud
        ##  obtainging point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.point_queue = []
        self.pcd_lock = threading.Lock()
        ## processing point cloud
        self.ila = imagelidaraligner.ImageLidarAligner(em, im)

        #for visualization
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window("Point Cloud", width=800, height=600)
        # self.first_frame = True
        self.vis_pcd = o3d.geometry.PointCloud()
        
        rospy.Timer(rospy.Duration(0.2), self.periodic_callback)
        

    def image_callback(self, image_msg):
        try:
            # Convert ROS Image to OpenCV (BGR8 format)
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # store the image
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
        
    # task: retrieve the most recent images (self.image) and point cloud (self.pcd)
    # do some operations on them
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
                myimg = copy.deepcopy(self.image)
                myimg = cv2.cvtColor(myimg, cv2.COLOR_BGR2RGB)
            
            # Start processing if both mypts and myimg are not empty
            if mypts != None and myimg is not None:
                print("\n\n=====================================")
                # # example: show both image and point cloud
                # # show image
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
                
                closest_pts,valid_pts,dist = self.ila.reportPoints(lightpos, mypts)
                print("average distance from origin is: ", dist)


        except Exception as e:
            rospy.logerr(f"Error in periodic callback: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener = Listener()
    rospy.spin()
