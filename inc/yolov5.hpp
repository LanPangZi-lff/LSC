#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>      // 深度学习模块
#include <opencv2/imgproc.hpp>	// 图像处理模块
#include <opencv2/highgui.hpp>  // 高层GUI图形用户界面

#include<time.h>

// 模型
class YOLOv5
{
public:
    // 初始化
	YOLOv5(const std::string &modelpath,const float &confThreshold=0.3,
        const float &nmsThreshold=0.5,const float &objThreshold=0.3,bool isCuda=false);
	std::vector<cv::Rect>  Detect(cv::Mat frame);  // 检测函数
private:
	float m_confThreshold;
	float m_nmsThreshold;
	float m_objThreshold;
	int m_inpWidth;
	int m_inpHeight;
	int m_numClasses;
    std::string m_classes[80] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
                    "train", "truck", "boat", "traffic light", "fire hydrant",
                    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                    "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
                    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
                    "sink", "refrigerator", "book", "clock", "vase", "scissors",
                    "teddy bear", "hair drier", "toothbrush"};
	const bool m_keepRatio = true;
	cv::dnn::Net m_net;   // dnn里的
	cv::Mat ResizeImage(cv::Mat srcimg, int *newh, int *neww, int *top, int *left);
};
