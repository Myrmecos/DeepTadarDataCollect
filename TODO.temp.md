# May 26
1. run ros to display bag file (simulate real running scenario)
    `rosbag play /home/astar/dart_ws/testing_data/test2.bag`
2. run listener to obtain images
    `rosrun image_view image_saver image:=/hikrobot_camera/rgb _filename_format:="target/frame%04d.jpg" _save_all_image:=true`
3. modify lidar_image_align/scripts/listener.py and dart_lidar_image_utils/src/dart_lidar_image_utils/imagealigner.py to obtain stable greenlight recognition performance.
    1. save all images in bag file to dart_lidar_image_utils/src/dart_lidar_image_utils/images
    2. visualize the recognition result of one image, modify it until:
        1. we have a clear circle after color filter (you may apply hsl)
        2. we can stably choose the center of the circle

## shortcut to some paths/actions

`roscore`
`rosbag play /home/astar/dart_ws/calib/calibpointcloud/calibscene_test.bag`
`rosrun lidar_image_align listener.py`

# May 27th
1. extrinsic matrix adjustment: between pointcloud and image (adjusting the extrinsic param manually)
    given 
    1. image coordinates (imcoords)
    2. corresponding lidar points (lidarcoords)
    3. the original extrinsic matrix that is slightly inaccurate (self.extrinsicMatrix)
    4. the camera matrix (self.cameraMatrix) 
    make slight adjustment to extrinsicMatrix to make it more accurate


    
    

2. check alignment: use mouse to pinpoint a position, then project to point cloud and visualize
    <!-- 1. input: pointcloud, points in pointcloud that is target, image
    2. output: overlaid image of point cloud (transformed using extrinsic matrix) and image -->
    1. input: image, let user select position; then point cloud, highlight selected points

    utils: merge point clouds: `pcl_concatenate_points_pcd /home/astar/dart_ws/testing_data/test2/* && mv output.pcd /home/astar/dart_ws/testing_data/test2.pcd`

3. testing: take greenlight out and test the recognition
    result: reduce exposure time to 5000 and shrink guangquan to obtain optimal green light detection result

4. working together
    `sudo ip addr add 192.168.1.100/24 dev enp100s0`
    `roslaunch livox_ros_driver livox_lidar_rviz.launch`
    `roslaunch hikrobot_camera hikrobot_camera.launch`
    `rosrun lidar_image_align listener.py`
    `source devel/setup.sh`

# May 28th
1. streamline extrinsic adjustment procedures
    1. record images and pcd into bag file
    2. extract point clouds`rosrun pcl_ros bag_to_pcd xxx.bag /livox_points pcd`. For example, `rosrun pcl_ros bag_to_pcd data/test.bag /livox/lidar target/test`
    3. merge into one point cloud: `pcl_concatenate_points_pcd target/test/* && mv output.pcd target/test.pcd`
2. full run

typedef __packed struct
{
    uint8_t SOF;
    fp32 yaw_angle;
    fp32 pitch_angle;
    uint8_t found;
    uint8_t shoot_or_not;
    uint_8_t done fitting;
    uint8_t patrolling;
    uint9_t is_updated;
    fp32 base_dis;
    uint8_t checksum;
} vision_rx_t;
