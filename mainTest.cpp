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
#include "ocv_common.hpp"

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
	
	//std::cout << cv::getBuildInformation() << std::endl;
	//cv::VideoCapture cap("udpsrc port=3000 ! application/x-rtp, encoding-name=JPEG, payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink", cv::CAP_GSTREAMER);
	cv::VideoCapture cap;
	cap.open("/dev/video0");
	std::cout << "Test1" << std::endl;
	if (!cap.isOpened()) {
            throw std::logic_error("Cannot open input stream");
        }
	
	HumanPoseEstimator estimator("/home/pi/catkin_ws/src/pose_estimation/src/networks/human-pose-estimation-0001.xml","MYRIAD");
	
		
	int delay = 33;
        double inferenceTime = 0.0;
        cv::Mat image;
        if (!cap.read(image)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }
        estimator.estimate(image);
	std::cout << "Test1" << std::endl;
	cap.read(image);
	std::cout << "Test2" << std::endl;
	std::vector<HumanPose> poses = estimator.estimate(image);
	std::cout << "Test2" << std::endl;
	
	do{
	    double t1 = cv::getTickCount();
	    std::cout << "Test4" << std::endl;
            std::vector<HumanPose> poses = estimator.estimate(image);
	    std::cout << "Test5" << std::endl;
            double t2 = cv::getTickCount();
            if (inferenceTime == 0) {
                inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
            } else {
                inferenceTime = inferenceTime * 0.95 + 0.05 * (t2 - t1) / cv::getTickFrequency() * 1000;
            }
            
            renderHumanPose(poses, image);

            cv::Mat fpsPane(35, 155, CV_8UC3);
            fpsPane.setTo(cv::Scalar(153, 119, 76));
            cv::Mat srcRegion = image(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
            cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
            std::stringstream fpsSs;
            fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
            cv::putText(image, fpsSs.str(), cv::Point(16, 32),
                        cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));
            cv::imshow("ICV Human Pose Estimation", image);
	    int key = cv::waitKey(delay) & 255;
            if (key == 'p') {
                delay = (delay == 0) ? 33 : 0;
            } else if (key == 27) {
                break;
            }
	}while(cap.read(image));
	
	/*
	cv::Mat image;
	cv::Mat testImg = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
	while(ros::ok()){
	    //get image from gstreamer via opencv
	    cap.read(image);
	    if (image.empty()){
		std::cout << "Frame empty!" << std::endl;
		break;
	    }
	    cv::imshow("Image", image);
	    cv::waitKey(1);
	    
	    std::vector<HumanPose> poses = estimator.estimate(testImg);
	    
	    //std::cout << "Test1" << std::endl;
	    //std::vector<HumanPose> poses = estimator.estimate(image);
	    //std::cout << "Test2" << std::endl;
	    //get pose from Human Pose Estimator
	    //std::cout << "Test2" << std::endl;
	    //std::vector<HumanPose> poses = estimator.estimate(frame);
	    //std::cout << "Test3" << std::endl;
	    //renderHumanPose(poses, frame);
	    //publish human pose 
	}
	*/
	/*
	cv::Mat image;
	while(ros::ok()){
	    //get image from gstreamer via opencv
	    cap.read(image);
	    if (image.empty()){
		std::cout << "Frame empty!" << std::endl;
		break;
	    }
	    cv::imshow("Image", image);
	    cv::waitKey(1);
	}
	*/
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
