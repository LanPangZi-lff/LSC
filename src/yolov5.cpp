#include "yolov5.hpp"
// 初始化
YOLOv5::YOLOv5(const std::string &modelpath,const float &confThreshold,
        const float &nmsThreshold,const float &objThreshold,bool isCuda):
m_confThreshold(confThreshold),m_nmsThreshold(nmsThreshold),m_objThreshold(objThreshold)
{
	m_net = cv::dnn::readNet(modelpath);
	if (isCuda) 
    {
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	//cpu
	else 
    {
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	m_numClasses = sizeof(m_classes)/sizeof(m_classes[0]);  // 类别数量
	m_inpHeight = 640;
	m_inpWidth = 640;
}

cv::Mat YOLOv5::ResizeImage(cv::Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;  // 输入高宽
	*newh = m_inpHeight;    // 指针变量指向输入yolo模型的宽高
	*neww = m_inpWidth;
	cv::Mat dstimg;                 // 定义一个目标源
	if (m_keepRatio && srch != srcw) 
    {  // 高宽不等
		float hw_scale = (float)srch / srcw; // 保存比列
		if (hw_scale > 1) 
        {     // 按照yolov5的预处理进行处理
			*newh = m_inpHeight;
			*neww = int(m_inpWidth / hw_scale); // 
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((m_inpWidth - *neww) * 0.5);
            // 和yolov5的处理对应,没有进行32的取模运算,这个是用114像素填充到(640,640)了,最后输入还是640,640
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, m_inpWidth - *neww - *left, cv::BORDER_CONSTANT, 114);
		}
		else 
        {
			*newh = (int)m_inpHeight * hw_scale;
			*neww = m_inpWidth;
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);	
			*top = (int)(m_inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, m_inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 114);
		}
	}
	else 
    {
		resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}

// 预测
std::vector<cv::Rect> YOLOv5::Detect(cv::Mat frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;  
	cv::Mat dstimg = ResizeImage(frame, &newh, &neww, &padh, &padw);  // 预处理
	cv::Mat blob = cv::dnn::blobFromImage(dstimg, 1 / 255.0, cv::Size(m_inpWidth, m_inpHeight), cv::Scalar(0, 0, 0), true, false); // return:4-dimensional Mat with NCHW dimensions order.
	m_net.setInput(blob);    // 设置输出
	std::vector<cv::Mat> outs;   // 要给空的走一遍
	m_net.forward(outs, m_net.getUnconnectedOutLayersNames());  // [b,num_pre,(5+classes)]

	int num_proposal = outs[0].size[1]; // 25200
	int out_dim2 = outs[0].size[2];  // 
	if (outs[0].dims > 2)
	{
		outs[0] = outs[0].reshape(0, num_proposal);  // 一般都会大于二维的，所以展平二维[b,num_pre*(5+classes)]
	}
	/////generate proposals
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;    //  opencv里保存box的
	std::vector<int> classIds;
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	float* pdata = (float*)outs[0].data;  // 定义浮点型指针，
	for(int i = 0; i < num_proposal; ++i) // 遍历所有的num_pre_boxes
	{
		int index = i * out_dim2;      // prob[b*num_pred_boxes*(classes+5)]  
		float obj_conf = pdata[index + 4];  // 置信度分数
		if (obj_conf > m_objThreshold)  // 大于阈值
		{
			//Mat scores = outs[0].row(row_ind).colRange(5, nout); // 相当于python里的切片操作，每类的预测类别分数
			cv::Mat scores(1, m_numClasses, CV_32FC1, pdata+index + 5);     // 这样操作更好理解
			cv::Point classIdPoint; //定义点
			double max_class_socre; // 定义一个double类型的变量保存预测中类别分数最大值
			// Get the value and location of the maximum score
			cv::minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);  // 求每类类别分数最大的值和索引
			max_class_socre *= obj_conf;   // 最大的类别分数*置信度
			if (max_class_socre > m_confThreshold) // 再次筛选
			{ 
				const int class_idx = classIdPoint.x;  // 类别索引,在yolo里就是表示第几类

				// 经过后处理的只需要直接取就行
				float cx = pdata[index];  //x
				float cy = pdata[index+1];  //y
				float w = pdata[index+2];  //w
				float h = pdata[index+3];  //h

				int left = int((cx - padw - 0.5 * w)*ratiow);  // *ratiow，变回原图尺寸
				int top = int((cy - padh - 0.5 * h)*ratioh);

				confidences.push_back((float)max_class_socre);
				boxes.push_back(cv::Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));  //（x,y,w,h）
				classIds.push_back(class_idx);  // 
			}
		}
	
	}    

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);
    return boxes;
}

