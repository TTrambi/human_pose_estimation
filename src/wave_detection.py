#!/usr/bin/env python

import rospy
from pose_estimation_msgs.msg import *


def callback(data):
	RWrist = None
	LWrist = None
	Neck = None
	
	bodyParts = data.persons[0].body_part
	for bodyPart in bodyParts:
		if bodyPart.part_id == 1:
			Neck = bodyPart
		if bodyPart.part_id == 4:
			RWrist = bodyPart
		if bodyPart.part_id == 7:
			LWrist = bodyPart
	
	if Neck != None and RWrist != None and RWrist.y < Neck.y:
		print("Waving")
		print("NeckY: " + str(Neck.y) + " " +  str(Neck.x) + " " + "RWrist: " + str(RWrist.y) + " " + str(RWrist.x))
		#call service
		rospy.wait_for_service('isWaving', 0)
		try: 
			isWaving = rospy.ServiceProxy('isWaving', detectWave)
			resp = isWaving()
		except rospy.ServiceException, e:
			print("Service call failed: %s"%e)
	elif Neck != None and LWrist != None and LWrist.y < Neck.y:
		print("Waving")
		print("NeckY: " + str(Neck.y) + ", " + "LWrist: " + str(LWrist.y))
		#call service
		rospy.wait_for_service('isWaving', 0)
		try: 
			isWaving = rospy.ServiceProxy('isWaving', detectWave)
			resp = isWaving()
		except rospy.ServiceException, e:
			print("Service call failed: %s"%e)
	else:
		print("not waving")


def main():
	rospy.init_node('wave_detection')
	sub = rospy.Subscriber("/pose_estimation", Persons, callback)
	#service = rospy.Service('isWaving', detectWave, handle_isWaving)
	rospy.spin()


if __name__ == '__main__':
    print "Running"
    main()
