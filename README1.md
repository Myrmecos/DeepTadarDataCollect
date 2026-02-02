# camera 

## camera calibration
To run calibration, do: `rosrun camera_calibration cameracalibrator.py --size 11x8 --square 0.108 image:=/hikrobot_camera/rgb`

## save camera image
To save image, run: `roslaunch hikrobot_camera hikrobot_camera_save.launch`

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


#-------------------------------------------------------------------------------
### side note
if you installed Hikrobot camera SDK on this machine, you may encounter a problem when running bag_to_pcd:`undefined symbol: libusb_set_option`
solution: add this line at the end of ~/.bashrc: `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`. This ensures we can use the system's library.
## change to readable form
`rostopic echo -b xxx.bag /livox_points > xxx.txt`.

#=========================================================================================================================
# Lidar camera calib
modify contents in livox_camera_calib/config/calib.yaml
run `roslaunch livox_camera_calib calib.launch`

P.S. if you want to fine-tune parameters,
1. check which yaml file you are using for edge detection parameters, in: `/home/astar/dart_ws/src/livox_camera_calib/config/calib.yaml`'s `calib_config_file` field. It is usually: `/home/astar/dart_ws/src/livox_camera_calib/config/config_outdoor.yaml`
2. go to to adjust your parameters for edge detection.
and then run: `roslaunch livox_camera_calib adjust_calib_param.launch`
see if the lines from image (blue) and lines from point cloud (red) are good.
If they are good, you can go back to launch using calib.launch.

$_{L}^{C}T = (_{L}^{C}R, _{L}^{C}t)\in SE$