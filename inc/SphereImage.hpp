#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

class SphereImage
{
public:
    SphereImage(const int &faculaNumber);
    bool DetectCenter(const cv::Mat &img);
private:
    cv::Point2d m_sphereCenter;
    std::vector<cv::Point2d> m_faculaCenter;
    cv::Rect m_roi; //圆的外接正方形区域
    int m_faculaNumber; //光斑数量
    std::vector<cv::Point2d> SubEdgeDection(cv::Mat input,double sigma ,double th_h,double th_l, int &curveNums,std::vector<int>&curveLimits);
    bool DetectPurkjin(cv::Mat roi_pupil);
};