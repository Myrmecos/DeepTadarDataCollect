# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019-2024. All rights reserved.
#
import sys
import os
import signal
import time
import logging
import serial
import numpy as np
import cv2 as cv
from pprint import pprint
import argparse

from senxor.utils import connect_senxor, data_to_frame, remap
from senxor.utils import cv_filter, cv_render, RollingAverageFilter
from senxor.commandargparser import parse_args

# This will enable mi48 logging debug messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

# what to do before quitting, whatever the cause
def close_app():
    mi48.stop()
    cv.destroyAllWindows()

# define a signal handler to ensure clean closure upon CTRL+C
# or kill from terminal
def signal_handler(sig, frame):
    """Ensure clean exit in case of SIGINT or SIGTERM"""
    logger.info("Exiting due to SIGINT or SIGTERM")
    close_app()
    logger.info("Done.")
    sys.exit(0)

# Define the signals that should be handled to ensure clean exit
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main_loop(mi48, img_scale, args):

    # Note that fpa_shape is as column x rows, is common in imagers.
    # This is is in reverse to numpy array shape.
    ncols, nrows = mi48.fpa_shape

    hflip = False
    vflip = True

    # initiate continuous frame acquisition
    # --------------------------------------
    mi48.start(stream=True, with_header=True)
    # frame counter to help calculating the true FPS
    fid0 = 0

    # loop
    while True:

        # read data: get a 1D array
        data, header = mi48.read()

        if data is None:
            logger.critical('NONE received instead of data; Check USB connection')
            close_app()
            # this situation typically requires reconnecting the senxor/mi48 and restart
            # therefore, no exception handling needed; quit straight away.
            sys.exit(1)

        min_temp, max_temp = data.min(), data.max()
        mean_temp = data.mean()

        # transform to 2D array
        frame = data_to_frame(data, (ncols,nrows), hflip=hflip);
        frame = np.flipud(frame) if vflip else frame

        # transform to uint8 for colormapping and rendering; only frame is mandatory
        # frame_u8 = remap(frame, curr_range=(min_temp, max_temp),
        #                         new_range=(0, 255), to_uint8=True)
        frame_u8 = remap(frame)

        # render a thermogram
        img_0 = cv_render(frame_u8, resize=img_scale,
                          interpolation=cv.INTER_NEAREST_EXACT, display=False,
                          colormap=args.colormap, with_colorbar=args.with_colorbar,
                          cbar_min=min_temp, cbar_max=max_temp)

        if fid0 == 0:
            t0 = header['timestamp']
            fid0 = header['frame_counter']
            fps_computed = 0
        else:
            tcurr = header['timestamp']
            fidcurr = header['frame_counter']
            fps_computed = (fidcurr - fid0) / (tcurr - t0) * 1.e3

        window_name = window_title = f'{mi48.sn} ({mi48.name})'
        window_title = f'{mi48.name}:{mi48.sn}'
        window_title += f' @{header["senxor_temperature"]:.1f}''°C'
        window_title += f' @{fps_computed:.1f} FPS,'
        window_title += f' Min {min_temp:.1f}'
        window_title += f' Mean {mean_temp:.1f}'
        window_title += f' Max {max_temp:.1f}'#  f'({frame.mean():.1f})''°C'

        cv.imshow(window_name, img_0)
        cv.setWindowTitle(window_name, window_title)

        # this wait is mandatory for opencv
        key = cv.waitKey(1)  # & 0xFF

        # check for user keyboard interaction, possible only if mouse is in focus
        if key in [ord("q"), ord('Q'), 27]:
            # use 'q' or 'ESC' to quit the program
            break
        if key == ord("f"):
            hflip = not hflip

def setup_thermal_camera(mi48, fps_divisor):
    # MI48 Settings
    # --------------------------------------------------
    # print out camera info
    # logger.info('Camera info:')
    # logger.info(mi48.camera_info)

    # Frame rate
    mi48.regwrite(0xB4, fps_divisor)  #

    # Disable firmware filters and min/max stabilisation
    if mi48.ncols == 160:
        # no FW filtering for Panther in the mi48 for the moment
        mi48.regwrite(0xD0, 0x00)  # temporal
        mi48.regwrite(0x20, 0x00)  # stark
        mi48.regwrite(0x25, 0x00)  # MMS
        mi48.regwrite(0x30, 0x00)  # median
    else:
        # MMS and STARK are sufficient for Cougar
        mi48.regwrite(0xD0, 0x00)  # temporal
        mi48.regwrite(0x30, 0x00)  # median
        mi48.regwrite(0x20, 0x03)  # stark
        mi48.regwrite(0x25, 0x01)  # MMS

    mi48.set_fps(30)
    mi48.set_emissivity(0.95)  # emissivity to 0.95, as used in calibration,
                               # so there is no sensitivity change
    mi48.set_sens_factor(1.0)  # sensitivity factor 1.0
    mi48.set_offset_corr(0.0)  # offset 0.0
    mi48.set_otf(0.0)          # otf = 0

    mi48.regwrite(0x02, 0x00)  # disable readout error compensation
    return mi48

def main(comport=None):

    args = parse_args()

    global mi48
    # mi48 = connect_senxor(comport=args.comport)
    mi48 = connect_senxor(comport="/dev/ttyACM1")
    mi48 = setup_thermal_camera(mi48, args.fps_divisor)

    # the following work within the python class, not the mi48 firmware
    mi48.set_data_type('temperature')
    mi48.set_temperature_units('Celsius')

    # img_scale controls how many times the default resolution we upscale
    # the default args.img_scale of 3 aims at an image size of around 480 x 320
    img_scale = {160: 1, 80: 2, 50: 3}
    img_scale = args.img_scale * img_scale[mi48.ncols]


    main_loop(mi48, img_scale, args)
    
    fps = mi48.get_fps()
    logger.info(f"FPS: {fps}")

    close_app()


if __name__ == "__main__":
    # ="/dev/ttyACM1"
    main()
