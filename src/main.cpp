#include <iostream>
#include <opencv2/opencv.hpp>
#include "SphereImage.hpp"
int main()
{
    cv::Mat image=cv::imread("F:/gaze/helmet_light_cali_image/7.5405/left/test/1/91yy.png");
    SphereImage si;
    si.DetectCenter(image);
    return 0;
}