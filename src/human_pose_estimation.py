#!/usr/bin/env python


import sys
sys.path = ['/home/pi/Downloads/inference_engine_vpu_arm/python/python2.7'] + sys.path
import cv2

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pose_estimation_msgs.msg import *

import human_pose_estimation_functions
import numpy as np
import time
import human_pose_estimation_functions

import argparse

############################################
############ HANDLE ARGS ###################
############################################
parser = argparse.ArgumentParser()
parser.add_argument('--network', type=int, help='Select Network for Human Pose Estimation, 1=fastNN, 2=openPoseSingle, 3=openPoseMulti')



############################################
######### DEFINE POSE POINTS ###############
############################################
#https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
#COCO Output Format
#BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }


#POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


#MPII Output Format
#BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
#               "Background": 15 }

#POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#               ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Chest", "LHip"],
#               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Head"] ]

##for fast eva                  
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13}

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Head"] ]

           
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

#POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
#              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
#              [1,0], [0,14], [14,16], [0,15], [15,17],
#              [2,17], [5,16] ]

nPoints = 18
# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]



############################################
############# INIT NETWORK #################
############################################
#Load pose estimation model
net = cv2.dnn.readNet('/home/pi/catkin_ws/src/pose_estimation/src/networks/human-pose-estimation-0001.xml','/home/pi/catkin_ws/src/pose_estimation/src/networks/human-pose-estimation-0001.bin')
#net = cv2.dnn.readNet('/home/pi/catkin_ws/src/pose_estimation/src/networks/model.xml','/home/pi/catkin_ws/src/pose_estimation/src/networks/model.bin')

#Set device to Neural Compute Stick
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

#init CV Bridge 
bridge = CvBridge()

#Create Ros publisher
pub = rospy.Publisher('/pose_estimation', Persons, queue_size=1)


############################################
############# INIT NETWORK #################
############################################

#Process network output

def processPoseSingle(frame, output):
	#SINGLE PERSON 
	# #source: https://github.com/quanhua92/human-pose-estimation-opencv/blob/master/openpose.py#L57
	frameWidth = frame.shape[1]
	frameHeight = frame.shape[0]
	points = []

	for i in range(len(BODY_PARTS)):
		# Slice heatmap of corresponging body's part.
		heatMap = output[0, i, :, :]

		# Originally, we try to find all the local maximums. To simplify a sample
		# we just find a global one. However only a single pose at the same time
		# could be detected this way.
		_, conf, _, point = cv2.minMaxLoc(heatMap)
		x = (frameWidth * point[0]) / output.shape[3]
		y = (frameHeight * point[1]) / output.shape[2]
		# Add a point if it's confidence is higher than threshold.
		points.append((int(x), int(y)) if conf > 0.15 else None)

	#create msg
	#create BodyPartDetection
	msgPersons = Persons()
	msgPersonDetection = PersonDetection()
	for i in range(len(BODY_PARTS)):
		if points[i] != None:
			msgBodyPartDetection = BodyPartDetection()
			msgBodyPartDetection.part_id = i
			msgBodyPartDetection.x = points[i][0]
			msgBodyPartDetection.y = points[i][1]
			msgBodyPartDetection.confidence = 1
			msgPersonDetection.body_part.append(msgBodyPartDetection)
	msgPersons.persons.append(msgPersonDetection)

	#publish msg
	pub.publish(msgPersons)

	#Show body parts in image
	for pair in POSE_PAIRS:
		partFrom = pair[0]
		partTo = pair[1]
		assert(partFrom in BODY_PARTS)
		assert(partTo in BODY_PARTS)

		idFrom = BODY_PARTS[partFrom]
		idTo = BODY_PARTS[partTo]

		if points[idFrom] and points[idTo]:
			cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
			cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
			cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

	cv2.imshow('image',frame)
	cv2.waitKey(1)

def processPoseMulti(frame, output):
	#MULTI PERSON 
	# #source: https://www.leanopencv.com/multi-person-pose-estimation-in-opencv-using-openpose/
	# https://github.com/spmallick/learnopencv/blob/master/OpenPose-Multi-Person/multi-person-openpose.py
	#Find nose/head/neck (bodypart 0) and find valid pairs by using Part Affinity Maps
	
	frameWidth = frame.shape[1]
	frameHeight = frame.shape[0]

	detected_keypoints = []
	keypoints_list = np.zeros((0,3))
	keypoint_id = 0
	threshold = 0.1

	for part in range(nPoints):
		probMap = output[0,part,:,:]
		probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
		keypoints = human_pose_estimation_functions.getKeypoints(probMap, threshold)
		#("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
		keypoints_with_id = []
		for i in range(len(keypoints)):
			keypoints_with_id.append(keypoints[i] + (keypoint_id,))
			keypoints_list = np.vstack([keypoints_list, keypoints[i]])
			keypoint_id += 1

		detected_keypoints.append(keypoints_with_id)

	frameClone = frame.copy()
	
	#show detected keypoints
	#for i in range(nPoints):
	#	for j in range(len(detected_keypoints[i])):
	#		cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
	#		cv2.imshow("Keypoints",frameClone)

	valid_pairs, invalid_pairs = human_pose_estimation_functions.getValidPairs(frameClone, output, detected_keypoints)
	personwiseKeypoints = human_pose_estimation_functions.getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

	#show multi-person poses
	for i in range(17):
		for n in range(len(personwiseKeypoints)):
			index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
			if -1 in index:
				continue
			B = np.int32(keypoints_list[index.astype(int), 0])
			A = np.int32(keypoints_list[index.astype(int), 1])
			cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

	#create msg
	msgPersons = Persons()
	for i in range(personwiseKeypoints.shape[0]):
		msgPersonDetection = PersonDetection()
		for j in range(len(keypointsMapping)):
			index = int(personwiseKeypoints[i][j])
			if index > -1:
				msgBodyPartDetection = BodyPartDetection()
				msgBodyPartDetection.part_id = j
				msgBodyPartDetection.x = np.uint32(keypoints_list[index, 0])
				msgBodyPartDetection.y = np.uint32(keypoints_list[index, 1])
				msgBodyPartDetection.confidence = np.float32(keypoints_list[index, 2])
				msgPersonDetection.body_part.append(msgBodyPartDetection)
		msgPersons.persons.append(msgPersonDetection)
	pub.publish(msgPersons)

	cv2.imshow("Detected Pose" , frameClone)
	cv2.waitKey(1)
		

def callback(data):
	#print("Message received")
	frame = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

	#cv2.imshow('Feed', frame)
	#cv2.waitKey(1)
	print("Test")
	#feed neural network and process
	#size for openpose network
	#blob = cv2.dnn.blobFromImage(frame, size=(456, 256), ddepth=cv2.CV_8U)
	#size for fast network
	blob = cv2.dnn.blobFromImage(frame, size=(192, 192), ddepth=cv2.CV_8U)
	net.setInput(blob)

	#single pose estimation
	out = net.forward()

	#multi-person pose estimation with openpose network
	#names = ['Mconv7_stage2_L1','Mconv7_stage2_L2']
	#out = net.forward(names)

	
	#output = np.zeros((1,out[0].shape[1]+out[1].shape[1],out[0].shape[2], out[0].shape[3]))

	#for i  in range(out[1].shape[1]):
	#	output[:,i,:,:] = out[1][:,i,:,:]
	
	#for i  in range(out[0].shape[1]):
	#	output[:,i+out[1].shape[1],:,:] = out[0][:,i,:,:]
	
	#extract the pose from the neural network output, single or multi person pose estimation
	processPoseSingle(frame, out)
	#processPoseMulti(frame, output)

def main():
	print('Started listening')
	rospy.init_node('pose_estimation')
	#sub = rospy.Subscriber("/camera/image_raw", Image, callback)
	cap = cv2.VideoCapture('udpsrc port=3000 ! application/x-rtp, encoding-name=JPEG, payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
	#i guess this command does not work :(, buffer is still 5 
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 0);
	if not cap.isOpened():
		print('VideoCapture not opened')
		exit(0)

	print('VideoCapture opened')

	while True:
		startTime = time.time()
		#default buffer size is 5, so empty buffer every time
		for i in xrange(1):
			cap.grab()
			
		ret,frame = cap.read()
		
		if not ret:
			print('empty frame')
			break

		#cv2.imshow('receive', frame)
		#if cv2.waitKey(1)&0xFF == ord('q'):
		#	break
			
		#startTime = time.time()
		
		#blob = cv2.dnn.blobFromImage(frame, size=(192, 192), ddepth=cv2.CV_8U)
		blob = cv2.dnn.blobFromImage(frame, size=(456, 256), ddepth=cv2.CV_8U)
		net.setInput(blob)
		out = net.forward()
		#midTime = time.time()
		
		#processPoseSingle(frame, out)
		endTime = time.time()
		
		#print(midTime - startTime)
		print(endTime - startTime)
		print('\n')
		
		
	cap.release()
	rospy.spin()


if __name__ == '__main__':
    print "Running"
    main()


