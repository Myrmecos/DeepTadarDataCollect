# instructions

1. obtain images
    a. Visualization only:
        python data_collection.py --save_data 0
    b. Collecting data:
        python data_collection.py --collection_duration 600 --save_data 1 --save_path RawData/exp01 

2. visualize raw data
    a. visualize raw data in a folder:
        python3 videostream.py --dirbase /media/zx/zx-data/RawData/exp06/

3. transform images in batch. Example:
    a. do not save, only visualize: 
        python3 transform_batch.py --dirbase /media/zx/zx-data/RawData/exp06/ --sensor_name MLX/ --transform_name realsense_depth/ --yaml_base_dir calibresults/ --max_dis 4 --save 0
    b. save:
        python3 transform_batch.py --dirbase /media/zx/zx-data/RawData/exp06/ --sensor_name MLX/ --transform_name realsense_depth/ --yaml_base_dir calibresults/ --max_dis 4 --save 1

4. visualize transformed images. 
    Example:
        python3 visualize_transformed_image.py --dirbase /media/zx/zx-data/RawData/exp06 --sensor_name senxor_m08 --image_index 0\


