#include <SphereImage.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov5.hpp"

SphereImage::SphereImage()
{
}

bool SphereImage::DetectCenter(const cv::Mat &img)
{
    if(img.empty())
    {
        std::cout<<"image is empty!!!"<<std::endl;
        return false;
    }
    cv::Mat srcImg=img.clone();
    YOLOv5 model("F:/gaze/LCS/files/last0605.onnx");
    std::vector<cv::Rect> roi = model.Detect(srcImg);
    std::cout << "Detection box: ";
    for (auto val : roi) {
        cv::rectangle(srcImg, val, cv::Scalar(0, 0, 255), 1);
    }
    imshow("w",srcImg);
    cv::waitKey(0);
    return false;
}
