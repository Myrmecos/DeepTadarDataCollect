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
1. To start publishing livox point cloud, do: 
`sudo ip addr add 192.168.1.100/24 dev enp100s0` to configure the ip (the addr is an example. You have to make sure lidar and your computer is in the same subnet)

2. `roslaunch livox_ros_driver livox_lidar_rviz.launch` to launch

## record point cloud
1. terminal one: `roscore`

2. terminal two: `roslaunch livox_ros_driver livox_lidar_rviz.launch`

3. terminal three: `rosbag record -a` or: `rosbag record -O calib/calibpointcloud/calibscene.bag`

4. control + C to stop recording

## visualize collected point cloud in .bag file
1. terminal one: `roscore`

2. terminal two: `rosrun rviz rviz`

3. in RVIZ GUI: in lower left corner, select "add" option, choose "PointCloud2", change the "Topic" field to "/livox/lidar"
In top left, Global Options' "fixed frame" field, change contents to "livox_frame"

4. terminal three: `rosbag play ***.bag`

## change .bag file to .pcd file:
1. first, transform topic in /livox_points to pcd into a directory`rosrun pcl_ros bag_to_pcd xxx.bag /livox_points pcd`. 
For example, `rosrun pcl_ros bag_to_pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene.bag /livox/lidar /home/astar/dart_ws/calib/calibpointcloud/calibscene`

2. then, merge all pcd files into one: `pcl_concatenate_points_pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene/* && mv output.pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene.pcd `

3. visualize the final pcd file: `pcl_viewer /home/astar/dart_ws/calib/calibpointcloud/calibscene.pcd`

`pcl_converter input.pcd output.txt -format txt`

#-------------------------------------------------------------------------------
### side notes
1. if you installed Hikrobot camera SDK on this machine, you may encounter a problem when running bag_to_pcd:`undefined symbol: libusb_set_option`;\
solution: add this line at the end of ~/.bashrc: `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`. This ensures we can use the system's library.

## change to readable form
1. does not seem to work: `rostopic echo -b xxx.bag /livox_points > xxx.txt`.

## cropping point cloud
**Important!!!!!!! If you wish to manipulate point cloud, remember to include the intensity field. Otherwise, a point cloud without intensity field will lead to livox_camera_calib reporting error.**
1. set parameters in `/home/astar/dart_ws/image_lidar_align/croppointcloud.py`

2. run: `python3 /home/astar/dart_ws/image_lidar_align/croppointcloud.py`. 

Notes: To correctly use livox_camera_calib, you need to ensure intensity field exists in point cloud data. To read the intensity of point cloud, you can use pyntcloud. It allows you to extract the intensity field.\
There does not seem to be a library that can directly write point cloud with "intensity" field, and you have to implement writing functionality on your own. For reference, see the end of the file croppointcloud.py.

#=========================================================================================================================
# Lidar camera calib
1. transform point cloud to ascii encoding: `pcl_convert_pcd_ascii_binary /home/astar/dart_ws/calib/calibpointcloud/calibscene.pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene_ascii.pcd 0`

2. modify contents in livox_camera_calib/config/calib.yaml
run `roslaunch livox_camera_calib calib.launch`

$_{L}^{C}T = (_{L}^{C}R, _{L}^{C}t)\in SE$

#=======================================================================================================================
# Working together

## Notes. 
To test and modify the two classes for identifying object's pixel coordinate (GLPosition in imageprocessor.py) and matching pixel coordinate to point cloud points (ImageLidarAligner in imagelidaraligner.py), you can go to this directory to play around (this does not require the use of ros and should be faster): `/home/astar/dart_ws/image_lidar_align`

## Step 1: general setup
1. set camera param path in lidar_image_align's listener.py: `CAMERA_PARAM_PATH = "/home/astar/dart_ws/src/livox_camera_calib/config/calib_ori.yaml"` 

2. set how many frames of point cloud we want to aggregate together for resolving target distance by: `MAX_PCD_MESSAGES = 6` (we collect 6 frames, pool all points together, and find the points corresponding to target object among these points)

## Step 2: Mock test (with fake data)
1. point cloud: `rosbag play /home/astar/dart_ws/calib/calibpointcloud/calibscene.bag`

2. image: `rosrun lidar_image_align talker.py`

3. run main program: `rosrun lidar_image_align listener.py`

## Step 2: actual run (with real data):
1. point cloud: `roslaunch livox_ros_driver livox_lidar_rviz.launch`

2. image: `roslaunch hikrobot_camera hikrobot_camera.launch`

3. run main program: `rosrun lidar_image_align listener.py`