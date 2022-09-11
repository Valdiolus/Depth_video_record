import numpy as np
import time
import cv2
import math
from datetime import datetime
import multiprocessing as mp
import pyrealsense2.pyrealsense2 as rs
from numpy import newaxis, zeros

SHOW_MULTI = 0

def FPS_CHECK(stp, fps_int, framecount_int, elapsedTime_int, t1_int, t2_int):
	if(stp == 1):
		t1_int = time.perf_counter()

	if(stp == 2):
		t2_int = time.perf_counter()
		framecount_int += 1
		elapsedTime_int += t2_int - t1_int

		if elapsedTime_int > 0.3:
			fps_int = "{:.1f} FPS".format(framecount_int / elapsedTime_int)
			#print("fps = ", str(fps_int))
			framecount_int = 0
			elapsedTime_int = 0

	return fps_int, framecount_int, elapsedTime_int, t1_int, t2_int

# .avi 'M','J','P','G' t1=25ms, t2=55ms, vlc works

# .avi 'm', 'p', '4', 'v' t1=30ms, t2=30-60ms, vlc works

# .avi *'XVID' t1=30ms, t2=30-50ms, vlc works

# .mp4 'm', 'p', '4', 'v' t1=35ms, t2=30-50ms, vlc works

def Thread_record_camera(file_name, color_framebuffer, video_fps):
    #wait till the 
    while color_framebuffer.empty():
        b=1

    frame = cv2.imdecode(color_framebuffer.get()[1], cv2.IMREAD_UNCHANGED)
    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (frame.shape[1], frame.shape[0]))
    time_with_frames = time.time()
    while(True):
        t0 = time.time()
        if not color_framebuffer.empty():
            frame = cv2.imdecode(color_framebuffer.get()[1], cv2.IMREAD_UNCHANGED)
            #print("t1:", time.time() - t0)
            t0 = time.time()
            out.write(frame)
            #print("t2:", time.time() - t0)
            time_with_frames = time.time()

        if time.time() - time_with_frames > 10:
            break
    
    print("Closing recording thread")
    # Closes all the frames
    cv2.destroyAllWindows()
    out.release()
    exit(0)


def Thread_read_d435_camera(file_name, rgb_width, rgb_height, rgb_target_fps, d_width, d_height, d_target_fps, rgb_framebuffer, detect_framebuffer, detector_results):
    error_counts = 0
    started = 0
    try:
        fps = ""
        framecount = 0
        elapsedTime = 0
        t1 = 0
        t2 = 0

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()


        config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, rgb_target_fps)
        time.sleep(0.5)
        

        #time.sleep(1)
        config.enable_stream(rs.stream.depth, d_width, d_height, rs.format.z16, d_target_fps)
        time.sleep(0.5)

        #Not encrypted - 2GB/min !!!
        #enable saving to file .bag
        #config.enable_record_to_file(file_name)

        # Start streaming
        pipeline.start(config)
        started = 1
        print("Reading video inited")

        while True:#for i in range(color_buffer.shape[0]):
            
                # inc FPS
                fps, framecount, elapsedTime, t1, t2 = FPS_CHECK (1, fps, framecount, elapsedTime, t1, t2)
                
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()

                color_frame = frames.get_color_frame()

                if color_frame:
                    # Convert images to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    #print("t3", time.time()-t0)

                    #cv2.imwrite('mvs/img.jpg', color_image)
                    encoded = cv2.imencode('.jpg', color_image)

                    if rgb_framebuffer.full():
                        rgb_framebuffer.get()
                    rgb_framebuffer.put(encoded)

                    if detect_framebuffer.full():
                        detect_framebuffer.get()
                    detect_framebuffer.put(cv2.resize(color_image, (320, 180), interpolation = cv2.INTER_AREA))

                    # calc and add fps rate
                    #cv2.putText(color_image, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (38, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(color_image, fps, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

                else:
                    print("No color video data!")
                    #break

                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data()).astype('uint8')
                    
                else:
                    print("No depth video data!")
                    #break

                while not detector_results.empty():
                    det_results = detector_results.get()
                    #print("new results!")
                    for res in det_results:
                        #print(res)
                        if res[0] == 0:
                            color = (0,0,255)
                        else:
                            color = (255,0,0)
                        
                        cv2.rectangle(color_image, (res[2], res[3]), (res[2]+res[4],res[3]+res[5]), color, 2)
                        



                if SHOW_MULTI and color_frame and depth_frame:
                    color_resized = cv2.resize(color_image, (426, 240), interpolation = cv2.INTER_AREA)
                    depth_resized = cv2.resize(cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB), (426, 240), interpolation = cv2.INTER_AREA)
                    # SHOW on the screen
                    combined_img = cv2.hconcat([color_resized, depth_resized])
                    #combined_img = np.concatenate((color_resized, depth_resized), axis=0)

                    cv2.imshow('frame', combined_img)
                else:
                    # (747, 420) - ideal for rpi screen
                    cv2.imshow('frame', cv2.resize(color_image, (747, 420), interpolation = cv2.INTER_AREA))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Keyboard interrupt")
                        pipeline.stop()
                        print("Closing video thread")
                        exit(0)
                        break	

                # update the FPS counter
                fps, framecount, elapsedTime, t1, t2 = FPS_CHECK (2, fps, framecount, elapsedTime, t1, t2)
    except Exception as e:
        print(e, "Error while reading video data - try once again")
        error_counts += 1
        if started:
            pipeline.stop()
        if error_counts > 10:
            exit(0)




