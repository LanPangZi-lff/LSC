#include <SphereImage.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov5.hpp"
#include "edgeTest.hpp"

SphereImage::SphereImage(const int &faculaNumber):m_faculaNumber(faculaNumber)
{
}

bool SphereImage::DetectCenter(const cv::Mat &img)
{
    if(img.empty())
    {
        std::cout<<"image is empty!!!"<<std::endl;
        return false;
    }
    cv::Mat srcImg=img.clone(); //深拷贝一份数据
    YOLOv5 model("F:/gaze/LCS/files/last0605.onnx"); //初始化模型
    std::vector<cv::Rect> roi=model.Detect(srcImg);
    if(roi.size()<1)
    {
        std::cout<<"yolo detect failure"<<std::endl;
        return false;
    }
    m_roi=roi[0];
    m_sphereCenter=cv::Point2d(m_roi.x+m_roi.width/2.0,m_roi.y+m_roi.height/2.0);

    if(DetectPurkjin(img(m_roi).clone()))
    {
        for(auto it:m_faculaCenter)
        {
            cv::circle(srcImg,cv::Point(it),2,cv::Scalar(0,0,255),-1);//在图像上画出普尔钦斑
        }
        //return true;
    }
    cv::rectangle(srcImg, m_roi, cv::Scalar(0, 0, 255), 1);
    circle(srcImg,cv::Point(m_sphereCenter.x,m_sphereCenter.y) ,(m_roi.width+m_roi.height)/4.0,cv::Scalar(255, 0, 120),1);//画圆，空心的
    cv::namedWindow("w",2);
    imshow("w",srcImg);
    cv::waitKey(0);
    return false;
}

bool SphereImage::DetectPurkjin(cv::Mat roi_pupil)
{
    int curveNums; //轮廓的数量
	std::vector<int> curveLimits;
    std::vector<cv::Point2d> points = SubEdgeDection(roi_pupil,1.2,50,50,curveNums,curveLimits);
    //std::cout<<"start"<<std::endl;
	for (int k = 0; k<curveNums; k++) /* write curves */
	{
		std::vector<cv::Point2f> tempLightEdge;
		for (int i = curveLimits[k]; i<curveLimits[k + 1]; i++)
		{
			tempLightEdge.emplace_back(cv::Point2f(points[i].x,points[i].y));
            // tempImage.at<uchar>(int(points[i].y-ROI_param.y),int(points[i].x-ROI_param.x)) = 0;
		}
        if (tempLightEdge.size() > 300)
        {
            continue;
        }
		//声明一个图像的矩
		cv::Moments M;
		//计算要绘制轮廓的矩
		M = moments(tempLightEdge);
		//求取轮廓重心的X坐标
		double cX = double(M.m10 / M.m00);
		//求取轮廓重心的Y坐标
		double cY = double(M.m01 / M.m00);
        if(std::isnan(cX)||std::isinf(cX)||std::isnan(cY)||std::isinf(cY))
        {
            continue;
        }
		m_faculaCenter.emplace_back(cv::Point2d(cX,cY));
        //std::cout<<cv::Point2d(cX,cY)<<std::endl;
	}
    if(m_faculaCenter.size()!=m_faculaNumber)
    {
        return false;
    }
    for(cv::Point2d &pos:m_faculaCenter)
    {
        pos.x+=m_roi.x;
        pos.y+=m_roi.y;
    }
    //排序
    std::sort(m_faculaCenter.begin(),m_faculaCenter.end(),[](cv::Point2d &a, cv::Point2d &b)->bool{return a.x<b.x;});
    std::sort(m_faculaCenter.begin(),m_faculaCenter.begin()+4,[](cv::Point2d &a, cv::Point2d &b)->bool{return a.y<b.y;});
    std::sort(m_faculaCenter.begin()+4,m_faculaCenter.end(),[](cv::Point2d &a, cv::Point2d &b)->bool{return a.y<b.y;});
    return true;
}

std::vector<cv::Point2d> SphereImage::SubEdgeDection(cv::Mat input,double sigma ,double th_h,double th_l, int &curveNums,std::vector<int>&curveLimits)
{
    cv::Mat srcimg,grayimg,dstimg;
	std::vector<cv::Point2d> res;

	//深拷贝图片
	srcimg=input.clone();
	// 判断图片通道数
	if(srcimg.channels()!=1)
	{
		cvtColor(srcimg, srcimg, cv::COLOR_BGR2GRAY);
	}
	
	double * x;          /* x[n] y[n] coordinates of result contour point n */
	double * y;
	int * curve_limits;  /* limits of the curves in the x[] and y[] */
	int N, M;         /* result: N contour points, forming M curves */

	grayimg = srcimg.clone();
	dstimg = grayimg.clone();
	const int iHeight = dstimg.rows;
	const int iWidth = dstimg.cols;
	uchar* pSrc = grayimg.data;//new uchar[iHeight*iWidth];
	uchar* pDst = dstimg.data;

	//亚像素边缘检测
	devernay(&x, &y, &N, &curve_limits, &M, pSrc, pDst, iWidth, iHeight, sigma, th_h, th_l);
	curveNums = M;
	//保存边缘点
	for(int i=0;i<N;++i)
	{
		cv::Point2d pos(x[i],y[i]);
		res.emplace_back(pos);
	}

	curveLimits.clear();
	for(int i=0;i<=M;++i)
	{
		curveLimits.emplace_back(curve_limits[i]);
	}

	//释放内存
    delete x;
    delete y;
    delete curve_limits;
	return res;
}
