#pragma once
#include <chrono>
#include <iostream>
#include "NvInferRuntime.h"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include "FasterRCNNcfg.h"
using namespace nvinfer1;
using namespace FasterRCNN;
namespace ObjDet {
	class  Faster_RCNN
	{
	public:
		Faster_RCNN();
		~Faster_RCNN();
		void Init();
		void BuildRcnnModel(unsigned int maxBatchSize, IHostMemory** modelStream, const std::string& wtsfile,
			const std::string& quantizationType);
		void serialize(std::string& wtsFile, std::string& engineFile);
		int LoadEngine(std::string& engineFile);
		void doInference(IExecutionContext& context, cudaStream_t& stream, std::vector<void*>& buffers,
			std::vector<float>& input, std::vector<float*>& output);
		std::vector<FasterRCNN::Detection> predict(cv::Mat& img);

	private:
		IRuntime*			m_runtime = nullptr;						// 
		ICudaEngine*		m_engine = nullptr;							// 
		IExecutionContext*	m_context = nullptr;						// 
		cudaStream_t		m_stream = nullptr;							// 
	};
}