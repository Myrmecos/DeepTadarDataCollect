#rosbag record -a -O target/testX.bag

rosrun pcl_ros bag_to_pcd target/test${1}.bag /livox/lidar target/test${1}
pcl_concatenate_points_pcd target/test$1/* && mv output.pcd target/test$1.pcd
