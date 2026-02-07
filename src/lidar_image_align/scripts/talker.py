#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
from dart_lidar_image_utils import imagelidaraligner
import cv2
import numpy as np

def numpy_to_pointcloud2(points, frame_id="map"):
    """Convert NumPy array to ROS PointCloud2."""
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    
    header = Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now()

    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = points.shape[0]
    cloud_msg.fields = fields
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = 12  # 3x float32 (12 bytes)
    cloud_msg.row_step = points.shape[0] * 12
    cloud_msg.is_dense = False
    cloud_msg.data = points.astype(np.float32).tobytes()  # Ensure float32
    
    return cloud_msg

def talker():
    pub_img = rospy.Publisher('/hikrobot_camera/rgb', Image, queue_size=10)
    pub_pts = rospy.Publisher('/livox/lidar', PointCloud2, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    # image = np.asarray(cv2.imread("/home/astar/Desktop/testing/test0_dot.jpg"))
    # pointcloud = imagelidaraligner.readPcd("/home/astar/dart_ws/single_scene_calibration/0.pcd")

    # image = np.asarray(cv2.imread("/home/astar/Desktop/testing/test1_dotdot_rec.jpg"))
    image = np.asarray(cv2.imread("/home/astar/dart_ws/calib/temp/img/000000.jpg"))
    #pointcloud = imagelidaraligner.readPcd("/home/astar/dart_ws/single_scene_calibration/0.pcd")

    #points = np.asarray(pointcloud.points)

    bridge = CvBridge()
    while not rospy.is_shutdown():
        cv_msg = bridge.cv2_to_imgmsg(image, "bgr8")
        pub_img.publish(cv_msg)
        #pt_msg = numpy_to_pointcloud2(points, frame_id="camera")
        #pub_pts.publish(pt_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
