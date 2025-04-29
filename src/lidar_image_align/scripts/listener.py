#!/usr/bin/env python
import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

class Listener:
    def __init__(self):
        self.bridge = CvBridge()
        image_sub = rospy.Subscriber('/hikrobot_camera/rgb', Image, self.image_callback)
        pc_sub = rospy.Subscriber('/livox/lidar', PointCloud2, self.pc_callback)

        #for visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Point Cloud", width=800, height=600)
        self.pcd = o3d.geometry.PointCloud()
        self.first_frame = True
        self.point_queue = []

    def image_callback(self, image_msg):
        try:
            # Convert ROS Image to OpenCV (BGR8 format)
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv2.imshow("Hikrobot Camera", cv_image)
            cv2.waitKey(1)
            rospy.loginfo("Image received and displayed")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def pc_callback(self, pc_msg):
        try:
            # Extract points from PointCloud2
            points = []
            for point in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append([point[0], point[1], point[2]])
            points = np.array(points, dtype=np.float64)
            # add to point queue
            self.point_queue.append(points)
            if (len(self.point_queue) > 5):
                self.point_queue.pop(0)
            
            # Update Open3D point cloud
            points_in_last_half_second = np.concatenate(self.point_queue, axis=0)
            self.pcd.points = o3d.utility.Vector3dVector(points_in_last_half_second)
            
            if self.first_frame:
                self.vis.add_geometry(self.pcd)
                self.first_frame = False
            else:
                self.vis.update_geometry(self.pcd)
            
            self.vis.poll_events()
            self.vis.update_renderer()
            rospy.loginfo("Point cloud received and displayed")
        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {e}")
        

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener = Listener()
    rospy.spin()
