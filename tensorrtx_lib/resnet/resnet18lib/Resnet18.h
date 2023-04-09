/*******************************************************************
** 文件名:	Resnet18.cpp
** 版  权:	(C) littlesheepy 2023 - All Rights Reserved
** 创建人:	littlesheepy
** 日  期:	2023年4月9日
** 版  本:	1.0
** 描  述:
** 应  用:
**************************** 修改记录 ******************************
** 修改人:
** 日  期:
** 描  述:
********************************************************************/
#pragma once
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "resnet18cfg.h"
#include "common.h"
using namespace nvinfer1;

namespace ObjDet {
    class Resnet18
    {
	public:
		Resnet18();
		~Resnet18();
		void Init(std::string& engine_name);
		ICudaEngine* Resnet18::createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
		void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);
		int LoadEngine();
		void doInference();
		void doInference(int batchSize);
		int serialize(std::string& wts_name, std::string& engine_name, bool& is_p6, float& gd, float& gw);
		//void deserialize();
		float* predict(cv::Mat& img);
		std::vector<std::string> predict_str(cv::Mat& img);
	public:
		IRuntime* m_runtime = nullptr;					// 
		ICudaEngine* m_engine = nullptr;					// 
		IExecutionContext*	m_context = nullptr;					// 
		cudaStream_t		m_stream = nullptr;					// 
		std::string			m_engine_name = "";
		void* buffers[2];								//

		float				input[BATCH_SIZE * 3 * INPUT_H * INPUT_W];		// 图片数据
		float				output[BATCH_SIZE * OUTPUT_SIZE];				// 输出数据
		int					inputIndex;
		int					outputIndex;
    };
}
