#!/usr/bin/env python

import os
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
import argparse

############################################
############ HANDLE ARGS ###################
############################################
parser = argparse.ArgumentParser(description='Ros publisher node for Human pose estimation')
parser.add_argument('-m', '--model', type=str, help='Path to the network models .xml file, .bin has to have same name in the same directory')
parser.add_argument('-o', '--output', type=int, default=0, help='Choose Output Model, 0=COCO(default), 1=MPII, 2=Reduced')
parser.add_argument('-i', '--input', type=str, help='Ros publisher note to get image from, otherwise opencv is used (with gstreamer over updsrc with prot 3000)' )
parser.add_argument('-vis', '--visualize', action='store_true', help='visualize the pose estimation in cv window')
args = parser.parse_args()



############################################
######### DEFINE POSE POINTS ###############
############################################
#https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

#set output 
if args.output == 1:
#MPII Output Format
	BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
				"LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
				"RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
				"Background": 15 }

	POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
				["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
				["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Chest", "LHip"],
				["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Head"] ]
elif args.output == 2:
##for fast eva                  
	BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
				"LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
				"RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13}

	POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
				["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
				["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
				["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Head"] ]
else:
	#COCO Output Format
	BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
				"LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
				"RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
				"LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }


	POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
				["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
				["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
				["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
				["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

   
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
modelXMLPath = ""
modelBinPath = ""
#Load pose estimation model
if args.model == None:
	filepath = os.path.dirname(os.path.realpath(__file__))
	modelXMLPath = filepath + "/networks/human-pose-estimation-0001.xml"
	modelBinPath = filepath + "/networks/human-pose-estimation-0001.bin"
else:
	modelXMLPath = args.model
	if os.path.isfile(modelXMLPath) == False:
		print("Could not find model file (.xml)")
		sys.exit(0)
	modelBinPath = modelXMLPath[:-4] + ".bin"
	if os.path.isfile(modelBinPath) == False:
		print("Could not find model file (.bin)")
		sys.exit(0)
	
try:
	#read network via opencv
	net = cv2.dnn.readNet(modelXMLPath,modelBinPath)

	#Set device to Neural Compute Stick
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
	
	#init CV Bridge 
	bridge = CvBridge()
except:
	print("Could not initialize the network or cvBridge!")
	sys.exit(0)
		
#Create Ros publisher
pub = rospy.Publisher('/pose_estimation', Persons, queue_size=1)


############################################
##### Processing Human Pose Functions ######
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
	if args.visualize:
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
		


############################################
########### Essential Fuinction ############
############################################
def callback(data):
	frame = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

	#feed neural network and process
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


def main():
	print('Started listening')
	rospy.init_node('pose_estimation')
	if args.input != None:
		print("Subscribing to %s", args.input)
		sub = rospy.Subscriber(args.input, Image, callback)
	else:
		print("Using opencv and gstreamer, udpsrc port=3000")
		cap = cv2.VideoCapture('udpsrc port=3000 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
		#i guess this command does not work :(, buffer is still 5 
		#cap.set(cv2.CAP_PROP_BUFFERSIZE, 0);
		if not cap.isOpened():
			print('OpenCV VideoCapture not opened')
			exit(0)
		print('OpenCV VideoCapture opened')
		while True:
			#default buffer size is 5, so empty buffer every time
			for i in xrange(1):
				cap.grab()
			
			ret,frame = cap.read()
		
			if not ret:
				print('empty frame')
				break
		
			blob = cv2.dnn.blobFromImage(frame, size=(192, 192), ddepth=cv2.CV_8U)
			#blob = cv2.dnn.blobFromImage(frame, size=(456, 256), ddepth=cv2.CV_8U)
			net.setInput(blob)
			out = net.forward()
		
			processPoseSingle(frame, out)
		cap.release()
		
	rospy.spin()


if __name__ == '__main__':
    print "Running"
    main()


