// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include "ros/ros.h"

#include <vector>

#include <inference_engine.hpp>

//#include <samples/ocv_common.hpp>

//#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

#include "opencv2/opencv.hpp"

#include "pose_estimation_msgs/Persons.h"
#include "pose_estimation_msgs/PersonDetection.h"
#include "pose_estimation_msgs/BodyPartDetection.h"

using namespace InferenceEngine;
using namespace human_pose_estimation;


int main(int argc, char* argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

	ros::init(argc, argv, "pose_estimation_cpp");
	ros::NodeHandle n;
        ros::Publisher pose_pub = n.advertise<pose_estimation_msgs::Persons>("human_pose_estimation", 1);
	
	HumanPoseEstimator estimator("/home/pi/catkin_ws/src/pose_estimation/src/human-pose-estimation-0001.xml","MYRIAD",0);
	
	std::cout << "Test" << std::endl;
	//std::cout << cv::getBuildInformation() << std::endl;
	cv::VideoCapture cap("updsrc port=3000 ! application/x-rtp, encoding-name=JPEG, payload=26 ! rtpjpegdepay ! jpegdec ! appsink", cv::CAP_GSTREAMER);
	std::cout << "Test1" << std::endl;
	cv::Mat frame;
	cap.read(frame);
	cv::imshow("Test", frame);
	cv::waitKey(1);
	
	while(ros::ok()){
	    //get image from gstreamer via opencv
	    
	    //get pose from Human Pose Estimator
	    
	  
	    //publish human pose 
	}
	
    
  
	//get image from gstreamer
	//cv::Mat image;
	
        //int delay = 33;
        //double inferenceTime = 0.0;
	//std::vector<HumanPose> poses = estimator.estimate(image);
        }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[ INFO ] Execution successful" << std::endl;
    return EXIT_SUCCESS;
}
