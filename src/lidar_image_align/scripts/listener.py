#!/usr/bin/env python
import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2

class Listener:
    def __init__(self):
        self.bridge = CvBridge()
        image_sub = message_filters.Subscriber('/hikrobot_camera/rgb', Image, self.callback)
        pc_sub = message_filters.Subscriber('/livox/lidar', PointCloud2)

    def callback(self, msg):
        try:
            # Convert ROS Image to OpenCV (BGR8 format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Display the image
            cv2.imshow("Hikrobot Camera", cv_image)
            cv2.waitKey(1)  # Refresh display
            
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

        

if __name__ == '__main__':
    listener = Listener()
    rospy.init_node('listener', anonymous=True)
    rospy.spin()
