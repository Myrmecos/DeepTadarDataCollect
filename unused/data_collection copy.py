import serial
import time
import ast
import numpy as np
import cv2
import sys
import os
import signal
import logging
import cv2 as cv
from pprint import pprint
import argparse
import pyrealsense2 as rs
from seekcamera import (
    SeekCameraIOType,
    SeekCameraColorPalette,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekCamera,
    SeekFrame,
)
from collections import deque
import threading
import pickle
from time import sleep
from threading import Condition
from senxor.utils import connect_senxor, data_to_frame, remap
from senxor.utils import cv_filter, cv_render, RollingAverageFilter


class MLXSensor:
    def __init__(self, sensor_port):
        self.sensor_port = sensor_port
        self.ser = serial.Serial(self.sensor_port, 921600, timeout=1)

    def read_data(self):
        data = self.ser.readline().strip()
        if len(data) > 0:
            try:
                msg_str = str(data.decode('utf-8'))
                msg = ast.literal_eval(msg_str)
                return msg
            except:
                return None
        return None
    
    def get_temperature_map(self):
        data = self.read_data()
        if data is not None:
            temp = np.array(data["temperature"]) # 768
            if len(temp) == 768:
                temp = temp.reshape(24, 32)
                return temp
        return None   
    
    def get_ambient_temperature(self):
        data = self.read_data()
        if data:
            return data["at"]
        return None
    
    def close(self):
        self.ser.close()
    
    def SubpageInterpolating(self,subpage):
        shape = subpage.shape
        mat = subpage.copy()
        for i in range(shape[0]):
            for j in range(shape[1]):
                if mat[i,j] > 0.0:
                    continue
                num = 0
                try:
                    top = mat[i-1,j]
                    num = num+1
                except:
                    top = 0.0
                
                try:
                    down = mat[i+1,j]
                    num = num+1
                except:
                    down = 0.0
                
                try:
                    left = mat[i,j-1]
                    num = num+1
                except:
                    left = 0.0
                
                try:
                    right = mat[i,j+1]
                    num = num+1
                except:
                    right = 0.0
                mat[i,j] = (top + down + left + right)/num
        return mat


class senxor:
    def __init__(self, sensor_port = "/dev/ttyACM1"):
        self.sensor_port = sensor_port
        self.mi48 = connect_senxor(comport=self.sensor_port)
        self.setup_thermal_camera(fps_divisor=3) 
        
        self.mi48.set_data_type('temperature')
        self.mi48.set_temperature_units('Celsius')
        
        self.ncols, self.nrows = self.mi48.fpa_shape
        self.mi48.start(stream=True, with_header=True)

    def get_temperature_map(self):
        return self.mi48.read() # data, header 
    
    def get_temperature_map_shape(self):
        return self.ncols, self.nrows
    
    def setup_thermal_camera(self, fps_divisor = 3):
        self.mi48.regwrite(0xB4, fps_divisor)  #
        # Disable firmware filters and min/max stabilisation
        if self.mi48.ncols == 160:
            # no FW filtering for Panther in the mi48 for the moment
            self.mi48.regwrite(0xD0, 0x00)  # temporal
            self.mi48.regwrite(0x20, 0x00)  # stark
            self.mi48.regwrite(0x25, 0x00)  # MMS
            self.mi48.regwrite(0x30, 0x00)  # median
        else:
            # MMS and STARK are sufficient for Cougar
            self.mi48.regwrite(0xD0, 0x00)  # temporal
            self.mi48.regwrite(0x30, 0x00)  # median
            self.mi48.regwrite(0x20, 0x03)  # stark
            self.mi48.regwrite(0x25, 0x01)  # MMS
        self.mi48.set_fps(30)
        self.mi48.set_emissivity(0.95)  # emissivity to 0.95, as used in calibration,
                                       # so there is no sensitivity change
        self.mi48.set_sens_factor(1.0)  # sensitivity factor 1.0
        self.mi48.set_offset_corr(0.0)  # offset 0.0
        self.mi48.set_otf(0.0)          # otf = 0
        self.mi48.regwrite(0x02, 0x00)  # disable readout error compensation
    
    def close(self):
        self.mi48.stop()
        
        
class realsense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image


class Renderer:
    """Contains camera and image data required to render images to the screen."""
    def __init__(self):
        self.busy = False
        self.frame = SeekFrame()
        self.camera = SeekCamera()
        self.frame_condition = Condition()
        self.first_frame = True

def argb2bgr(frame):
    """Converts an RGBA8888 frame to a BGR frame."""
    if frame.shape[2] != 4:
        raise ValueError("Input frame must be RGBA8888")
    bgr_image = frame[:, :, 1:][:, :, ::-1]
    return bgr_image

class seekthermal:
    def __init__(self, data_format="color"):
        self.data_format = data_format
        self.manager = SeekCameraManager(SeekCameraIOType.USB)
        if self.data_format == "color": 
            self.renderer = Renderer()  
            self.manager.register_event_callback(self._on_event, self.renderer)      
            self.frame_condition = Condition()
        else:
            self.data_frame = None
            self.data_condition = False
            def on_frame2(camera, camera_frame, file):
                frame = camera_frame.thermography_float
                self.data_frame = frame.data
                self.data_condition = True
                # sleep(0.1)
            def on_event2(camera, event_type, event_status, user_data):
                print("{}: {}".format(str(event_type), camera.chipid))

                if event_type == SeekCameraManagerEvent.CONNECT:
                    camera.register_frame_available_callback(on_frame2, None)
                    camera.capture_session_start(SeekCameraFrameFormat.THERMOGRAPHY_FLOAT)

            self.manager.register_event_callback(on_event2)     

    def _on_event(self, camera, event_type, event_status, renderer):
        print("{}: {}".format(str(event_type), camera.chipid))

        def on_frame(_camera, camera_frame, renderer):
            with renderer.frame_condition:
                renderer.frame = camera_frame.color_argb8888
                renderer.frame_condition.notify()

        if event_type == SeekCameraManagerEvent.CONNECT:
            if renderer.busy:
                return
            renderer.busy = True
            renderer.camera = camera
            renderer.first_frame = True
            camera.color_palette = SeekCameraColorPalette.TYRIAN
            camera.register_frame_available_callback(on_frame, renderer)
            camera.capture_session_start(SeekCameraFrameFormat.COLOR_ARGB8888)


    def get_frame(self):
        if self.data_format == "color": 
            with self.renderer.frame_condition:
                if self.renderer.frame_condition.wait(150.0 / 1000.0):
                    frame = self.renderer.frame.data
                    if frame is not None:
                        return frame
        else:
            return self.data_frame
        return None

    def close(self):
        try:
            self.renderer.camera.capture_session_stop()
        except:
            pass
        self.manager.destroy()

if __name__ == "__main__":
    '''
    Visualization only:
        python data_collection.py --save_data 0
    Collecting data:
        python data_collection.py --collection_duration 600 --save_data 1 --save_path RawData/exp01 

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_duration", type=int, default=10, help="Duration to collect data, seconds")
    parser.add_argument("--save_data", type=int, default=0, help="0: not save, just visualize, 1: save to a pickle file without visualization")
    parser.add_argument("--save_path", type=str, default="data", help="path to save data")
    args = parser.parse_args()
    
    if args.save_data:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            os.makedirs(args.save_path + "/realsense_depth/")
            os.makedirs(args.save_path + "/realsense_color/")
            os.makedirs(args.save_path + "/seek_thermal/")
            os.makedirs(args.save_path + "/MLX/")
            os.makedirs(args.save_path + "/senxor_m08/")
            os.makedirs(args.save_path + "/senxor_m16/")
        else:
            print(f"The directory {args.save_path} already exists")
            exit(1)
    
    if args.save_data == 0:
        realsense_sensor = realsense()  
        seek_camera = seekthermal(data_format="color")
        mlx_sensor = MLXSensor("/dev/ttyUSB0")
        senxor_sensor_m08 = senxor(sensor_port="/dev/ttyACM0")
        senxor_sensor_m16 = senxor(sensor_port="/dev/ttyACM1")
        
        num_rows_m08, num_cols_m08 = senxor_sensor_m08.get_temperature_map_shape()
        print(num_rows_m08, num_cols_m08)
        num_rows_m16, num_cols_m16 = senxor_sensor_m16.get_temperature_map_shape()
        print(num_rows_m16, num_cols_m16)
        
        while True:
            realsense_depth_image, realsense_color_image = realsense_sensor.get_frame()
            seek_camera_frame = seek_camera.get_frame()
            MLX_temperature_map = mlx_sensor.get_temperature_map()
            senxor_temperature_map_m08, header1 = senxor_sensor_m08.get_temperature_map()
            senxor_temperature_map_m16, header2 = senxor_sensor_m16.get_temperature_map()

            realsense_color_image = cv2.resize(realsense_color_image, (320, 240))
            realsense_depth_image = cv2.resize(realsense_depth_image, (320, 240))   
            realsense_depth_image = cv2.applyColorMap(cv2.convertScaleAbs(realsense_depth_image, alpha=0.03), cv2.COLORMAP_JET)
            if seek_camera_frame is not None:
                if seek_camera_frame.shape[0] > 201:
                    seek_camera_frame = cv2.resize(argb2bgr(seek_camera_frame), (320, 240))
                else:
                    seek_camera_frame = np.flip(np.flip(cv2.resize(argb2bgr(seek_camera_frame), (320, 240)),0),1)
            else:
                seek_camera_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                
            if MLX_temperature_map is not None:
                MLX_temperature_map = MLX_temperature_map.reshape(24, 32)
                MLX_temperature_map = np.flip(MLX_temperature_map, 0)
                MLX_temperature_map = mlx_sensor.SubpageInterpolating(MLX_temperature_map)
                MLX_temperature_map = MLX_temperature_map.astype(np.uint8)
                # min_temp, max_temp = np.min(MLX_temperature_map), np.max(MLX_temperature_map)
                MLX_temperature_map = cv2.normalize(MLX_temperature_map, None, 0, 255, cv2.NORM_MINMAX)
                MLX_temperature_map = cv2.resize(MLX_temperature_map, (320, 240), interpolation=cv2.INTER_NEAREST)
                MLX_temperature_map = cv2.applyColorMap(MLX_temperature_map, cv2.COLORMAP_JET)
            else:
                MLX_temperature_map = np.zeros((240, 320, 3), dtype=np.uint8)
                
            if senxor_temperature_map_m08 is not None:
                senxor_temperature_map_m08 = senxor_temperature_map_m08.reshape(num_cols_m08, num_rows_m08)
                senxor_temperature_map_m08 = np.flip(senxor_temperature_map_m08, 0)
                senxor_temperature_map_m08 = senxor_temperature_map_m08.astype(np.uint8)
                senxor_temperature_map_m08 = cv2.normalize(senxor_temperature_map_m08, None, 0, 255, cv2.NORM_MINMAX)
                senxor_temperature_map_m08 = cv2.resize(senxor_temperature_map_m08, (320, 240), interpolation=cv2.INTER_NEAREST)
                senxor_temperature_map_m08 = cv2.applyColorMap(senxor_temperature_map_m08, cv2.COLORMAP_JET)
            else:
                senxor_temperature_map_m08 = np.zeros((240, 320, 3), dtype=np.uint8)
                
            if senxor_temperature_map_m16 is not None:
                senxor_temperature_map_m16 = senxor_temperature_map_m16.reshape(num_cols_m16, num_rows_m16)
                senxor_temperature_map_m16 = np.flip(senxor_temperature_map_m16, 0)
                senxor_temperature_map_m16 = senxor_temperature_map_m16.astype(np.uint8)
                senxor_temperature_map_m16 = cv2.normalize(senxor_temperature_map_m16, None, 0, 255, cv2.NORM_MINMAX)
                senxor_temperature_map_m16 = cv2.resize(senxor_temperature_map_m16, (320, 240), interpolation=cv2.INTER_NEAREST)
                senxor_temperature_map_m16 = cv2.applyColorMap(senxor_temperature_map_m16, cv2.COLORMAP_JET)
            else:
                senxor_temperature_map_m16 = np.zeros((240, 320, 3), dtype=np.uint8)
                
            # print(realsense_depth_image.shape, realsense_color_image.shape, seek_camera_frame.shape, MLX_temperature_map.shape, senxor_temperature_map_m08.shape, senxor_temperature_map_m16.shape)
            final_image = np.concatenate((realsense_depth_image, realsense_color_image, seek_camera_frame, senxor_temperature_map_m08, senxor_temperature_map_m16, MLX_temperature_map), axis=1)
            cv2.imshow("Final Image", final_image)
            key = cv.waitKey(1)
            if key in [ord("q"), ord('Q'), 27]:
                break
        seek_camera.close()
        mlx_sensor.close()
        senxor_sensor_m08.close()
        senxor_sensor_m16.close()
    else:
        collection_duration = args.collection_duration
        print("Start collecting data for {} seconds".format(collection_duration))
    
        realsense_sensor = realsense()  
        seek_camera = seekthermal(data_format= "temperature")
        mlx_sensor = MLXSensor("/dev/ttyUSB0")
        senxor_sensor_m08 = senxor(sensor_port="/dev/ttyACM0")
        senxor_sensor_m16 = senxor(sensor_port="/dev/ttyACM1")
        
        start_time = time.time()
        collection_counter = 0

        while True:
            realsense_depth_image, realsense_color_image = realsense_sensor.get_frame()
            seek_camera_frame = seek_camera.get_frame()
            MLX_temperature_map = mlx_sensor.get_temperature_map()
            senxor_temperature_map_m08, header1 = senxor_sensor_m08.get_temperature_map()
            senxor_temperature_map_m16, header2 = senxor_sensor_m16.get_temperature_map()
            num_cols_m08, num_rows_m08 = senxor_sensor_m08.get_temperature_map_shape()
            num_cols_m16, num_rows_m16 = senxor_sensor_m16.get_temperature_map_shape()

            if realsense_depth_image is None or realsense_color_image is None or seek_camera_frame is None or MLX_temperature_map is None or senxor_temperature_map_m08 is None or senxor_temperature_map_m16 is None:
                continue
            else:
                timestamp = time.time()
                # MLX_temperature_map = MLX_temperature_map.reshape(24, 32)
                # MLX_temperature_map = np.flip(MLX_temperature_map, 0)
                # senxor_temperature_map_m08 = senxor_temperature_map_m08.reshape(num_cols_m08, num_rows_m08)
                # senxor_temperature_map_m08 = np.flip(senxor_temperature_map_m08, 0)
                # senxor_temperature_map_m16 = senxor_temperature_map_m16.reshape(num_cols_m16, num_rows_m16)
                # senxor_temperature_map_m16 = np.flip(senxor_temperature_map_m16, 0)

                np.save(f"{args.save_path}/realsense_depth/{timestamp}.npy", realsense_depth_image)
                np.save(f"{args.save_path}/realsense_color/{timestamp}.npy", realsense_color_image)
                np.save(f"{args.save_path}/seek_thermal/{timestamp}.npy", seek_camera_frame)
                np.save(f"{args.save_path}/MLX/{timestamp}.npy", MLX_temperature_map)
                np.save(f"{args.save_path}/senxor_m08/{timestamp}.npy", senxor_temperature_map_m08)
                np.save(f"{args.save_path}/senxor_m16/{timestamp}.npy", senxor_temperature_map_m16)

                collection_counter += 1
                time_lasting = time.time() - start_time
                if collection_counter % 50 == 0:
                    print(f"Seek camera frame collected at {timestamp}", seek_camera_frame.shape)
                    print(f"Realsense depth and color image collected at {timestamp}", realsense_depth_image.shape, realsense_color_image.shape)
                    print(f"MLX temperature map collected at {timestamp}", MLX_temperature_map.shape)
                    print(f"Senxor temperature map m08 collected at {timestamp}", senxor_temperature_map_m08.shape)
                    print(f"Senxor temperature map m16 collected at {timestamp}", senxor_temperature_map_m16.shape)
                    print("-------------------------------------------------------------")
                if time_lasting > collection_duration:
                    print(f"Seek camera frame collected at {timestamp}", seek_camera_frame.shape)
                    print(f"Realsense depth and color image collected at {timestamp}", realsense_depth_image.shape, realsense_color_image.shape)
                    print(f"MLX temperature map collected at {timestamp}", MLX_temperature_map.shape)
                    print(f"Senxor temperature map m08 collected at {timestamp}", senxor_temperature_map_m08.shape)
                    print(f"Senxor temperature map m16 collected at {timestamp}", senxor_temperature_map_m16.shape)
                    print(f"Total frames collected: {collection_counter}")
                    print(f"Frame rate: {collection_counter / time_lasting} Hz")

                    # save the above as metadata in meta.log in the save_path
                    with open(f"{args.save_path}/meta.log", "w") as f:
                        f.write(f"Collecting time: {time_lasting} seconds\n")
                        f.write(f"Total frames collected: {collection_counter}\n")
                        f.write(f"Frame rate: {collection_counter / time_lasting} Hz\n")
                        f.write(f"Seek camera frame collected at {timestamp} {seek_camera_frame.shape}\n")
                        f.write(f"Realsense depth and color image collected at {timestamp} {realsense_depth_image.shape} {realsense_color_image.shape}\n")
                        f.write(f"MLX temperature map collected at {timestamp} {MLX_temperature_map.shape}\n")
                        f.write(f"Senxor temperature map m08 collected at {timestamp} {senxor_temperature_map_m08.shape}\n")
                        f.write(f"Senxor temperature map m16 collected at {timestamp} {senxor_temperature_map_m16.shape}\n")
                    break
        seek_camera.close()
        mlx_sensor.close()
        senxor_sensor_m08.close()
        senxor_sensor_m16.close()