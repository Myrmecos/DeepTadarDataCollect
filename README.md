# instructions

1. obtain images\
    a. Visualization only:\
        `python data_collection.py --save_data 0`\
    b. Collecting data:\
        `python data_collection.py --collection_duration 600 --save_data 1 --save_path RawData/exp01 --sleep_time 3`\
    c. When collecting for calibration, save images in below destination: 
        MSC/calibImages/<distance in meters>/

2. visualize raw data \
    a. visualize raw data as a video:\
        `python3 visualize_videostream.py --dirbase /media/zx/zx-data/RawData/exp06/`\
    b. visualize raw data as images:\
        `python3 visualize_image.py --dirbase /media/zx/zx-data/RawData/exp06/`

3. select corresponding points in thermal array and depth image. points will be recorded in a yaml file. \
    note: read images from the base path: MSC/calibImages/ \
    example:\
        `python MSC1_corresponding_points.py --transform realsense_depth/ --target seek_thermal/ --distance 1`\

4. calibrate the thermal array and depth camera, obtain rotation and translation matrix and scaling factor\
    example: \
    point calib mode: use points 
        `python3 calib.py --src_distance 4 --dest_distance 4 --baseDir /media/zx/zx-data/RawData/exp06/ --transform_dir realsense_depth/ --reference_dir seek_thermal/ --ind 1 --mode pointcalib`

5. transform depth images to match thermal images. Example:\
    a. do not save, only visualize: \
        `python3 thermal_images_depth_align.py --dirbase /media/zx/zx-data/RawData/exp06/ --sensor_name MLX/ --transform_name realsense_depth/ --yaml_base_dir calibresults/ --max_dis 4 --save 0`\
    b. save:\
        `python3 thermal_images_depth_align.py --dirbase /media/zx/zx-data/RawData/exp06/ --sensor_name MLX/ --transform_name realsense_depth/ --yaml_base_dir calibresults/ --max_dis 4 --save 1`

6. visualize transformed depth images, with comparison to thermal images.\
    Example:\
        `python3 visualize_transformed_image.py --dirbase /media/zx/zx-data/RawData/exp06 --sensor_name senxor_m08 --image_index 0`\

# check; 
1. near-neighbor vs bilinear effect (near-neighbor is better)\
2. grayscale for thermal image (or normalize at a global range, clip off out-of-range pixels) (clipped off for visualize as videostream)\
3. m16 and m08 sense temperature differently. difference at upper bound stays around 7 degrees