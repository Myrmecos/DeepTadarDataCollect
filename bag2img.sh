# parse the name of the bag file from command line argument
echo $1
dirname_part="${1%.*}"
echo $dirname_part
target_dir="${dirname_part}/img"
mkdir -p $target_dir

rosbag play $1 & rosrun image_view extract_images image:=/hikrobot_camera/rgb _sec_per_frame:=0 _filename_format:="$target_dir/%06d.jpg"
