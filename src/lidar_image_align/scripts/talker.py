#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def talker():
    pub = rospy.Publisher('/hikrobot_camera/rgb', Image, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    image = cv2.imread("/home/astar/Desktop/testing/output_with_dot_pillow.jpg")
    bridge = CvBridge()
    while not rospy.is_shutdown():
        cv_msg = bridge.cv2_to_imgmsg(image, "bgr8")
        pub.publish(cv_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
