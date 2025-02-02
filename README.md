1. transform images in batch. Example:
    a. do not save, only visualize: 
        python3 transform_batch.py --dirbase /media/zx/zx-data/RawData/exp06/ --sensor_name MLX/ --transform_name realsense_depth/ --yaml_base_dir calibresults/ --max_dis 4 --save 0
    b. save:
        python3 transform_batch.py --dirbase /media/zx/zx-data/RawData/exp06/ --sensor_name MLX/ --transform_name realsense_depth/ --yaml_base_dir calibresults/ --max_dis 4 --save 1

2. visualize transformed images. Example:
    python3 visualize_transformed_image.py --dirbase /media/zx/zx-data/RawData/exp06 --sensor_name senxor_m08 --image_index 0