#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

class Listener:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.Subscriber('/hikrobot_camera/rgb', Image, self.callback)

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
