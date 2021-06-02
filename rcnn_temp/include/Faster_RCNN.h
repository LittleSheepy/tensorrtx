#pragma once
#include "NvInferRuntime.h"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>
using namespace nvinfer1;
class  Faster_RCNN
{
public:
	Faster_RCNN();
	~Faster_RCNN();
	void Init(); 
	void doInference(IExecutionContext& context, cudaStream_t& stream, std::vector<void*>& buffers,
		std::vector<float>& input, std::vector<float*>& output);

private:
	IRuntime*			m_runtime	= nullptr;						// 
	ICudaEngine*		m_engine	= nullptr;						// 
	IExecutionContext*	m_context	= nullptr;						// 
	cudaStream_t		m_stream	= nullptr;						// 
};
