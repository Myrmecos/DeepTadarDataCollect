# parse the name of the bag file from command line argument
echo $1
dirname_part="${1%.*}"
echo $dirname_part
target_dir="${dirname_part}/pcd"

# use pcl library to convert bag to pcd
rosrun pcl_ros bag_to_pcd $1 /livox/lidar $target_dir

# concatenate all pcd files extracted from bag into single pcd file
# the lidar needs to be stationary during the bag recording for this to work properly
pcd_dir="${target_dir}/*"
echo $pcd_dir
pcl_concatenate_points_pcd $pcd_dir && mv output.pcd $dirname_part

# convert binary pcd to ascii pcd
# this ensures correct reading of the pcd file in the cropping python script
output=${dirname_part}/output.pcd
echo $output
output_ascii=${dirname_part}/output_ascii.pcd
pcl_convert_pcd_ascii_binary $output $output_ascii 0

# crop the point cloud using the cropping script
python3 /home/astar/dart_ws/image_lidar_align/croppointcloud.py $output_ascii

# rosrun pcl_ros bag_to_pcd /home/astar/dart_ws/testing_data/test2.bag /livox/lidar /home/astar/dart_ws/testing_data/test0/pcd
# pcl_concatenate_points_pcd /home/astar/dart_ws/testing_data/test0/pcd/* && mv output.pcd /home/astar/dart_ws/testing_data/test0/output.pcd
# pcl_convert_pcd_ascii_binary /home/astar/dart_ws/testing_data/test0/output.pcd /home/astar/dart_ws/testing_data/test0/output_ascii.pcd 0
# python3 /home/astar/dart_ws/image_lidar_align/croppointcloud.py /home/astar/dart_ws/testing_data/test0/output_ascii.pcd
