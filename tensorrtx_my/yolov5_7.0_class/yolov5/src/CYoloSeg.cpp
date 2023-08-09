#include "CYoloSeg.h"

CYoloSeg::CYoloSeg() {
}
CYoloSeg::~CYoloSeg() {
	// Release stream and buffers
	cudaStreamDestroy(m_stream);
	CUDA_CHECK(cudaFree(m_gpu_buffers[0]));
	CUDA_CHECK(cudaFree(m_gpu_buffers[1]));
	delete[] m_cpu_output_buffer1;
	delete[] m_cpu_output_buffer2;
	cuda_preprocess_destroy(m_img_buffer_host, m_img_buffer_device);
	// Destroy the engine
	m_context->destroy();
	m_engine->destroy();
	m_runtime->destroy();
}

// 初始化
void CYoloSeg::Init() {
	// 1.设置cuda ID
	cudaSetDevice(kGpuId);
	// 2.反序列化engine
	_deserialize_engine();
	// 3.创建stream
	CUDA_CHECK(cudaStreamCreate(&m_stream));
	// 4.Init CUDA preprocessing
	cuda_preprocess_init(kMaxInputImageSize, m_img_buffer_host, m_img_buffer_device);
	// 5.Prepare cpu and gpu buffers
	_prepare_buffers();
}

std::vector<Detection> CYoloSeg::Predict(cv::Mat& img, std::vector<std::vector<Detection>>& res_batch1, std::vector<std::vector<std::vector<cv::Point>>>& contours) {
	std::vector<cv::Mat> img_batch;
	img_batch.push_back(img);
	cuda_batch_preprocess(img_batch, m_gpu_buffers[m_inputIndex], kInputW, kInputH, m_stream, m_img_buffer_host, m_img_buffer_device);
	_do_inference();

	// NMS
	std::vector<std::vector<Detection>> res_batch;
	batch_nms(res_batch, m_cpu_output_buffer1, img_batch.size(), kOutputSize1, kConfThresh, kNmsThresh);
	auto masks = process_mask(&m_cpu_output_buffer2[0], kOutputSize2, res_batch[0]);
	printf(" size() %d", res_batch[0].size());
	for (size_t i = 0; i < res_batch[0].size(); i++) {
		cv::Mat img_mask = scale_mask(masks[i], img);
		cv::Mat binary_mask;
		cv::threshold(img_mask, binary_mask, 0.5, 1, cv::THRESH_BINARY);
		cv::Mat binary_mask_8U;
		binary_mask.convertTo(binary_mask_8U, CV_8UC1);
		std::vector<std::vector<cv::Point>> contours1;
		//cv::imwrite("binary_mask.jpg", binary_mask_8U *255);
		cv::findContours(binary_mask_8U, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		contours.push_back(contours1);
	}
	res_batch1.push_back(res_batch[0]);
	return res_batch[0];
}

// 反序列化
void CYoloSeg::_deserialize_engine() {
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
void CYoloSeg::_prepare_buffers() {
	assert(m_engine->getNbBindings() == 3);
	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	m_inputIndex = m_engine->getBindingIndex(kInputTensorName);
	m_outputIndex1 = m_engine->getBindingIndex(kOutputTensorName);
	m_outputIndex2 = m_engine->getBindingIndex("proto");
	assert(m_inputIndex == 0);
	assert(m_outputIndex1 == 1);
	assert(m_outputIndex2 == 2);
	// Create GPU buffers on device
	CUDA_CHECK(cudaMalloc((void**)&m_gpu_buffers[m_inputIndex], kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&m_gpu_buffers[m_outputIndex1], kBatchSize * kOutputSize1 * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&m_gpu_buffers[m_outputIndex2], kBatchSize * kOutputSize2 * sizeof(float)));

	// Alloc CPU buffers
	m_cpu_output_buffer1 = new float[kBatchSize * kOutputSize1];
	m_cpu_output_buffer2 = new float[kBatchSize * kOutputSize2];
}
void CYoloSeg::_do_inference() {
	m_context->enqueue(kBatchSize, (void**)&m_gpu_buffers, m_stream, nullptr);
	CUDA_CHECK(cudaMemcpyAsync(m_cpu_output_buffer1, this->m_gpu_buffers[m_outputIndex1], kBatchSize * kOutputSize1 * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
	CUDA_CHECK(cudaMemcpyAsync(m_cpu_output_buffer2, this->m_gpu_buffers[m_outputIndex2], kBatchSize * kOutputSize2 * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
	cudaStreamSynchronize(m_stream);
}

void CYoloSeg::SetEngineName(std::string engine_name) {
	m_engine_name = engine_name;
}

void CYoloSeg::SetWeightName(std::string weight_name) {
	m_weight_name = weight_name;
}

IYoloSeg* createYoloSegInstance()
{
	return new CYoloSeg();
}