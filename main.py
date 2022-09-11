import numpy as np
import time
import cv2
import math
from datetime import datetime
import multiprocessing as mp
import pyrealsense2.pyrealsense2 as rs
from numpy import newaxis, zeros
import video_record, detector



if __name__ == "__main__":
    time_struct = time.gmtime()

    time_now = str(time_struct.tm_year)+'-'+str(time_struct.tm_mon)+'-'+str(time_struct.tm_mday)+'_'+str(time_struct.tm_hour+3)+'-'+str(time_struct.tm_min)+'-'+str(time_struct.tm_sec)
    
    videofile1 = '/home/pi/work/Depth_video_record/mvs/Video_%s.avi' % time_now#'mvs/Video_%s.bag' % time_now

    rgb_width = 1280
    rgb_height = 720
    rgb_fps = 15

    d_width = 640
    d_height = 360
    d_fps = 15

    
    try:
        manager = mp.Manager()
        colour_framebuffer = manager.Queue(100)
        detect_framebuffer = manager.Queue(100)
        detector_results = manager.Queue(100)

        processes = []
        # Start streaming
        p = mp.Process(target=video_record.Thread_read_d435_camera, args=(videofile1, rgb_width, rgb_height, rgb_fps, d_width, d_height, d_fps, colour_framebuffer, detect_framebuffer, detector_results, ), daemon=True)
        p.start()
        processes.append(p)

        
        p = mp.Process(target=video_record.Thread_record_camera, args=(videofile1, colour_framebuffer, rgb_fps, ), daemon=True)
        p.start()
        processes.append(p)

        #p = mp.Process(target=detector.Thread_detect, args=(rgb_width, rgb_height, detect_framebuffer, detector_results, ), daemon=True)
        #p.start()
        #processes.append(p)

        for process in processes:
            process.join()
    finally:
        for p in range(len(processes)):
            processes[p].terminate()