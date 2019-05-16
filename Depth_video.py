#MIT license by Valdis Gerasymiak
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
#np.set_printoptions(threshold=np.inf)
import imutils
import time
import cv2
import math
from datetime import datetime

FPS_target = 15
frame_width_target = 640
frame_height_target = 480
total_seconds = 30
Frames_total = FPS_target*total_seconds

import pyrealsense2 as rs
from numpy import newaxis, zeros

def FPS_CHECK(stp, fps_int, framecount_int, elapsedTime_int, t1_int, t2_int):
	if(stp == 1):
		t1_int = time.perf_counter()

	if(stp == 2):
		t2_int = time.perf_counter()
		framecount_int += 1
		elapsedTime_int += t2_int - t1_int

		if elapsedTime_int > 0.3:
			fps_int = "{:.1f} FPS".format(framecount_int / elapsedTime_int)
			print("fps = ", str(fps_int))
			framecount_int = 0
			elapsedTime_int = 0

	return fps_int, framecount_int, elapsedTime_int, t1_int, t2_int

def VIDEO_WRITE(file1, file2):
	fps = ""
	framecount = 0
	elapsedTime = 0
	t1 = 0
	t2 = 0

	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()

	config.enable_stream(rs.stream.color, frame_width_target, frame_height_target, rs.format.bgr8, FPS_target)
	config.enable_stream(rs.stream.depth, frame_width_target, frame_height_target, rs.format.z16, FPS_target)

	#enable saving to file .bag
	config.enable_record_to_file(file1)

	# Start streaming
	pipeline.start(config)

	# create buffer for depth images
	#color_buffer = zeros((Frames_total,frame_height_target,frame_width_target,3)).astype('uint8')
	#depth_buffer = zeros((Frames_total,frame_height_target,frame_width_target,3)).astype('uint8')

	while True:#for i in range(color_buffer.shape[0]):
		# inc FPS
		fps, framecount, elapsedTime, t1, t2 = FPS_CHECK (1, fps, framecount, elapsedTime, t1, t2)
		
		# Wait for a coherent pair of frames: depth and color
		frames = pipeline.wait_for_frames()
		
		color_frame = frames.get_color_frame()
		if not color_frame:
			print("No color video data!")
			break

		depth_frame = frames.get_depth_frame()
		if not depth_frame:
			print("No depth video data!")
			break

		# Convert images to numpy arrays
		color_image = np.asanyarray(color_frame.get_data())
		#print(color_image.shape,color_buffer.shape)
		#color_buffer[i] = color_image
		#File_write(color_image)	
		
		depth_image = np.asanyarray(depth_frame.get_data()).astype('uint8')
		#depth_image = depth_image[:,:,newaxis]
		#depth_image = np.append(np.append(depth_image, depth_image, axis=2), depth_image, axis=2)
		#print(depth_image.shape, depth_buffer.shape)
		#depth_buffer[i] = depth_image

		# calc and add fps rate
		#cv2.putText(color_image, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (38, 0, 255), 1, cv2.LINE_AA)

		#SHOW COLOR FRAME
		cv2.imshow('frame', color_image)
		#cv2.imshow('frame2', depth_image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			print("Keyboard interrupt")
			break	
		
		# update the FPS counter
		fps, framecount, elapsedTime, t1, t2 = FPS_CHECK (2, fps, framecount, elapsedTime, t1, t2)

	print("Writing buffers to files!!!")
	# FILES TO WRITE INIT
	#color_file = cv2.VideoWriter(file1, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS_target, (frame_width_target, frame_height_target))
	#depth_file = cv2.VideoWriter(file2, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS_target, (frame_width_target, frame_height_target))

	#for i in range(color_buffer.shape[0]):
	#	color_file.write(color_buffer[i])
	#	depth_file.write(depth_buffer[i])
	#write depth buffer to file
	#np.savetxt('depth.txt', depth_buffer, fmt="%d", delimiter=',')

	# When everything done, release the video capture and video write objects
	pipeline.stop()

	# Closes all the frames
	cv2.destroyAllWindows()


def VIDEO_READ(file_in):
	file = open(file_in, 'r', encoding='ascii', errors='ignore')#cv2.VideoCapture(file_in)
	#if (file.isOpened() == False):
	#	print("Error opening file")

	time.sleep(1.0)
	
	time_period = 1/FPS_target
	time_old = time.perf_counter()
	print('Run file: %s' % file_in)

	lines = file.readlines()
	print(lines)
	for x in lines:
		if time.perf_counter() >= time_old:
			time_old = time_old + time_period

			#READ FRAME
			data = x
			if data is None:
				print("No data in frame")
				break

			#convert data to frame
			print(data)

			frame = 0
			#SHOW FRAME
			cv2.imshow('frame', frame)
	
			if cv2.waitKey(1) & 0xFF == ord('q'):
				print("Keyboard interrupt")
				break

	# When everything done, release the video capture and video write objects
	#file.release()
	file.close()
	#depth_file.release()

	# Closes all the frames
	cv2.destroyAllWindows()

def VIDEO_READ_RSD(file):
	fps = ""
	framecount = 0
	elapsedTime = 0
	t1 = 0
	t2 = 0

	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()

	# enable loading from .bag file
	rs.config.enable_device_from_file(config, file)

	config.enable_stream(rs.stream.color, frame_width_target, frame_height_target, rs.format.bgr8, FPS_target)
	config.enable_stream(rs.stream.depth, frame_width_target, frame_height_target, rs.format.z16, FPS_target)

	# Start streaming
	pipeline.start(config)

	# create buffer for depth images
	# color_buffer = zeros((Frames_total,frame_height_target,frame_width_target,3)).astype('uint8')
	# depth_buffer = zeros((Frames_total,frame_height_target,frame_width_target,3)).astype('uint8')

	while True:  # for i in range(color_buffer.shape[0]):
		# inc FPS
		fps, framecount, elapsedTime, t1, t2 = FPS_CHECK(1, fps, framecount, elapsedTime, t1, t2)

		# Wait for a coherent pair of frames: depth and color
		frames = pipeline.wait_for_frames()

		color_frame = frames.get_color_frame()
		if not color_frame:
			print("No color video data!")
			break

		depth_frame = frames.get_depth_frame()
		if not depth_frame:
			print("No depth video data!")
			break

		# Convert images to numpy arrays
		color_image = np.asanyarray(color_frame.get_data())
		# print(color_image.shape,color_buffer.shape)
		# color_buffer[i] = color_image
		# File_write(color_image)

		depth_image = np.asanyarray(depth_frame.get_data()).astype('uint8')
		# depth_image = depth_image[:,:,newaxis]
		# depth_image = np.append(np.append(depth_image, depth_image, axis=2), depth_image, axis=2)
		# print(depth_image.shape, depth_buffer.shape)
		# depth_buffer[i] = depth_image

		# calc and add fps rate
		# cv2.putText(color_image, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (38, 0, 255), 1, cv2.LINE_AA)

		# SHOW COLOR FRAME
		cv2.imshow('frame', color_image)
		cv2.imshow('frame2', depth_image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			print("Keyboard interrupt")
			break

		# update the FPS counter
		fps, framecount, elapsedTime, t1, t2 = FPS_CHECK(2, fps, framecount, elapsedTime, t1, t2)

	print("Writing buffers to files!!!")
	# FILES TO WRITE INIT
	# color_file = cv2.VideoWriter(file1, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS_target, (frame_width_target, frame_height_target))
	# depth_file = cv2.VideoWriter(file2, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS_target, (frame_width_target, frame_height_target))

	# for i in range(color_buffer.shape[0]):
	#	color_file.write(color_buffer[i])
	#	depth_file.write(depth_buffer[i])
	# write depth buffer to file
	# np.savetxt('depth.txt', depth_buffer, fmt="%d", delimiter=',')

	# When everything done, release the video capture and video write objects
	pipeline.stop()

	# Closes all the frames
	cv2.destroyAllWindows()

def MAIN():
	now = datetime.now()
	#dt_string = now.strftime('%d/%m/%Y/%H_%M_%S')
	videofile1 = 'Video_%s.bag' % datetime.now()#'Color_%s.avi' % datetime.now()
	videofile2 = 'Depth_%s.avi' % datetime.now()
	print('Videofile name:', videofile1)

	#VIDEO_WRITE(videofile1, videofile2)

	#time.sleep(5.0)
	print("SLEEEEEEP")

	#VIDEO_READ('Video_2019-05-14 19:59:13.603212.bag')
	VIDEO_READ_RSD('Video_2019-05-14 19:59:13.603212.bag')
	#VIDEO_READ(videofile2)
	#print("END OF FILE")


MAIN()
