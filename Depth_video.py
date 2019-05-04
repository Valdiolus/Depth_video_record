#MIT license by Valdis Gerasymiak
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import math
from datetime import datetime

FPS_target = 15
frame_width_target = 640
frame_height_target = 480

import pyrealsense2 as rs

def VIDEO_WRITE(file):
	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, frame_width_target, frame_height_target, rs.format.z16, FPS_target)
	config.enable_stream(rs.stream.color, frame_width_target, frame_height_target, rs.format.bgr8, FPS_target)

	# Start streaming
	pipeline.start(config)

	# FILES TO WRITE INIT
	color_file = cv2.VideoWriter(file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS_target, (frame_width_target, frame_height_target))

	depth_file = cv2.VideoWriter('depth.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS_target, (frame_width_target, frame_height_target))

	while(True):
		# Wait for a coherent pair of frames: depth and color
		frames = pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		if not depth_frame or not color_frame:
			print("No video data!")
			break

		# Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		#WRITE AND SHOW COLOR FRAME
		color_file.write(color_image)
		cv2.imshow('frame', color_image)

		#WRITE AND SHOW DEPTH FRAME
		depth_file.write(depth_image)
		cv2.imshow('frame', depth_image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	# When everything done, release the video capture and video write objects
	pipeline.stop()

	# Closes all the frames
	cv2.destroyAllWindows()


def VIDEO_READ(file1, file2):
	color_file = cv2.VideoCapture(file1)
	if (color_file.isOpened() == False):
		print("Error opening file")

	depth_file = cv2.VideoCapture(file2)
	if (depth_file.isOpened() == False):
		print("Error opening file")

	time.sleep(0.5)

	while(True):
		#READ FRAME
		ret, frame_color = color_file.read()
		ret, frame_depth = depth_file.read()

		#SHOW FRAME
		cv2.imshow('frame', frame_color)
		cv2.imshow('frame', frame_depth)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the video capture and video write objects
	color_file.release()
	depth_file.release()

	# Closes all the frames
	cv2.destroyAllWindows()

def MAIN():
	now = datetime.now()
	dt_string = now.strftime('%d/%m/%Y/%H:%M:%S')
	videofile = 'Video_proba_' + dt_string + '.avi'
	print('Videofile name:', videofile)

	VIDEO_WRITE(videofile)

	#time.sleep(5.0)
	print("SLEEEEEEP")

	#VIDEO_READ(videofile, videofile)
	#print("END OF FILE")


MAIN()