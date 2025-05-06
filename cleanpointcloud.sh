rosrun pcl_ros bag_to_pcd calib/calibpointcloud/calibscene_test.bag /livox/lidar /home/astar/dart_ws/calib/calibpointcloud/calibscene_test
pcl_concatenate_points_pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene_test/* && mv output.pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene_test.pcd
pcl_viewer /home/astar/dart_ws/calib/calibpointcloud/calibscene_test.pcd
pcl_convert_pcd_ascii_binary /home/astar/dart_ws/calib/calibpointcloud/calibscene_test.pcd /home/astar/dart_ws/calib/calibpointcloud/calibscene_test_ascii.pcd 0











