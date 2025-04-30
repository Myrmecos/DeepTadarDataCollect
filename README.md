# camera 

## camera calibration
To run calibration, do: `rosrun camera_calibration cameracalibrator.py --size 11x8 --square 0.108 image:=/hikrobot_camera/rgb`

## save camera image
To save image, run: `roslaunch hikrobot_camera hikrobot_camera_save.launch`

## all 0 output situation
change from: `camera::frame = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, m_pBufForSaveImage).clone(); //tmp.clone();`
to `camera::frame = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, m_pBufForDriver).clone();`
#========================================================================================================================

# lidar 

## livox point cloud publish
To start publishing livox point cloud, do: 
`sudo ip addr add 192.168.1.100/24 dev enp100s0` to configure the ip
`roslaunch livox_ros_driver livox_lidar_rviz.launch` to launch

## record point cloud
terminal one: `roscore`
terminal two: `roslaunch livox_ros_driver livox_lidar_rviz.launch`
terminal three: `rosbag record -a` or: `rosbag record -O calib/calibpointcloud/calibscene.bag`
control + C to stop recording

## visualize point cloud
terminal one: `roscore`
terminal two: `rosrun rviz rviz`
in RVIZ GUI: in lower left corner, select "add" option, choose "PointCloud2", change the "Topic" field to "/livox/lidar"
In top left, Global Options' "fixed frame" field, change contents to "livox_frame"
terminal three: `rosbag play ***.bag`

## change to pcd file:
1. first, transform topic in /livox_points to pcd into a directory`rosrun pcl_ros bag_to_pcd xxx.bag /livox_points pcd`. 
For example, `rosrun pcl_ros bag_to_pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene.bag /livox/lidar /home/astar/dart_ws/calib/calibpointcloud/calibscene`
2. then, merge all pcd files into one: `pcl_concatenate_points_pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene/* && mv output.pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene.pcd `
3. visualize the final pcd file: `pcl_viewer /home/astar/dart_ws/calib/calibpointcloud/calibscene.pcd`

`pcl_converter input.pcd output.txt -format txt`

#-------------------------------------------------------------------------------
### side note
if you installed Hikrobot camera SDK on this machine, you may encounter a problem when running bag_to_pcd:`undefined symbol: libusb_set_option`
solution: add this line at the end of ~/.bashrc: `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`. This ensures we can use the system's library.
## change to readable form
`rostopic echo -b xxx.bag /livox_points > xxx.txt`.
## cropping point cloud
**Important!!!!!!! If you wish to manipulate point cloud, remember to include the intensity field. Otherwise, a point cloud without intensity field will lead to livox_camera_calib reporting error.**
To read the intensity of point cloud, you can use pyntcloud. It allows you to extract the intensity field.
There does not seem to be a library that can directly write point cloud with "intensity" field, and you have to implement writing functionality on your own. For reference, see the end of the file croppointcloud.py.

#=========================================================================================================================
# Lidar camera calib
transform point cloud to ascii encoding: `pcl_convert_pcd_ascii_binary /home/astar/dart_ws/calib/calibpointcloud/calibscene.pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene_ascii.pcd 0`

modify contents in livox_camera_calib/config/calib.yaml
run `roslaunch livox_camera_calib calib.launch`

$_{L}^{C}T = (_{L}^{C}R, _{L}^{C}t)\in SE$

#=======================================================================================================================
# Working together


## Mock test (with fake data)
point cloud: `rosbag play /home/astar/dart_ws/calib/calibpointcloud/calibscene.bag`
image: `rosrun lidar_image_align talker.py`
run main program: `rosrun lidar_image_align listener.py`

## actual run (with real data):
point cloud: `roslaunch livox_ros_driver livox_lidar_rviz.launch`
image: `roslaunch hikrobot_camera hikrobot_camera.launch`
run main program: `rosrun lidar_image_align listener.py`