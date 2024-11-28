#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

class SphereImage
{
public:
    SphereImage();
    bool DetectCenter(const cv::Mat &img);
private:
    cv::Point2d m_sphereCenter;
    std::vector<cv::Point2d> m_faculaCenter;
};