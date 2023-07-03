#include "CYoloDet.h"

CYoloDet::CYoloDet(){
}
CYoloDet::~CYoloDet(){
	// Release stream and buffers
	cudaStreamDestroy(m_stream);
	CUDA_CHECK(cudaFree(m_gpu_buffers[0]));
	CUDA_CHECK(cudaFree(m_gpu_buffers[1]));
	delete[] m_cpu_output_buffer;
	cuda_preprocess_destroy();
	// Destroy the engine
	m_context->destroy();
	m_engine->destroy();
	m_runtime->destroy();
}

// 初始化
void CYoloDet::Init() {
	// 1.设置cuda ID
	cudaSetDevice(kGpuId);
	// 2.反序列化engine
	_deserialize_engine();
	// 3.创建stream
	CUDA_CHECK(cudaStreamCreate(&m_stream));
	// 4.Init CUDA preprocessing
	cuda_preprocess_init(kMaxInputImageSize);
	// 5.Prepare cpu and gpu buffers
	_prepare_buffers();

}

std::vector<Detection> CYoloDet::Predict(cv::Mat& img) {
	std::vector<cv::Mat> img_batch;
	img_batch.push_back(img);
	cuda_batch_preprocess(img_batch, m_gpu_buffers[m_inputIndex], kInputW, kInputH, m_stream);
	_do_inference();

	// NMS
	std::vector<std::vector<Detection>> res_batch;
	batch_nms(res_batch, m_cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
	return res_batch[0];
}

// 反序列化
void CYoloDet::_deserialize_engine() {
	std::ifstream file(m_engine_name, std::ios::binary);
	if (!file.good()) {
		std::cerr << "read " << m_engine_name << " error!" << std::endl;
		assert(false);
	}
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	char* serialized_engine = new char[size];
	assert(serialized_engine);
	file.read(serialized_engine, size);
	file.close();

	m_runtime = createInferRuntime(gLogger);
	assert(m_runtime);
	m_engine = (m_runtime)->deserializeCudaEngine(serialized_engine, size);
	assert(m_engine);
	m_context = (m_engine)->createExecutionContext();
	assert(m_context);
	delete[] serialized_engine;
}
void CYoloDet::_prepare_buffers() {
	assert(m_engine->getNbBindings() == 2);
	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	m_inputIndex = m_engine->getBindingIndex(kInputTensorName);
	m_outputIndex = m_engine->getBindingIndex(kOutputTensorName);
	assert(m_inputIndex == 0);
	assert(m_outputIndex == 1);
	// Create GPU buffers on device
	CUDA_CHECK(cudaMalloc((void**)&m_gpu_buffers[m_inputIndex], kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&m_gpu_buffers[m_outputIndex], kBatchSize * kOutputSize * sizeof(float)));

	m_cpu_output_buffer = new float[kBatchSize * kOutputSize];
}
void CYoloDet::_do_inference() {
	m_context->enqueue(kBatchSize, (void**)&m_gpu_buffers, m_stream, nullptr);
	CUDA_CHECK(cudaMemcpyAsync(m_cpu_output_buffer, this->m_gpu_buffers[m_outputIndex], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
	cudaStreamSynchronize(m_stream);
}

IYoloDet* createYoloDetInstance()
{
	return new CYoloDet();
}