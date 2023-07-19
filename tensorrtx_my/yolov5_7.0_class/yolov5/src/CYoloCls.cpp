#include "CYoloCls.h"
CYoloCls::CYoloCls() {
}
CYoloCls::~CYoloCls() {
	// Release stream and buffers
	cudaStreamDestroy(m_stream);
	CUDA_CHECK(cudaFree(m_gpu_buffers[0]));
	CUDA_CHECK(cudaFree(m_gpu_buffers[1]));
	delete[] m_cpu_input_buffer;
	delete[] m_cpu_output_buffer;
	// Destroy the engine
	m_context->destroy();
	m_engine->destroy();
	m_runtime->destroy();
}

// 初始化
void CYoloCls::Init() {
	// 1.设置cuda ID
	cudaSetDevice(kGpuId);
	if (!std::experimental::filesystem::exists(m_engine_name)) {
		Serialize();
	}
	// 2.反序列化engine
	_deserialize_engine();
	// 3.创建stream
	CUDA_CHECK(cudaStreamCreate(&m_stream));
	// 4.Prepare cpu and gpu buffers
	_prepare_buffers();

}

int CYoloCls::Predict(cv::Mat& img) {
	std::vector<cv::Mat> img_batch;
	img_batch.push_back(img);
	batch_preprocess(img_batch, m_cpu_input_buffer);
	_do_inference();
	auto res = softmax(m_cpu_output_buffer, kOutputSize);
	auto topk_idx = topk(res, 1);
	return topk_idx[0];
}

void CYoloCls::Serialize() {
	_serialize_engine();
}

void CYoloCls::_serialize_engine() {
	if (m_weight_name.empty()) {
		std::cerr << "have no wts file" << std::endl;
		return;
	}
	// Create builder
	IBuilder* builder = createInferBuilder(gLogger);
	IBuilderConfig* config = builder->createBuilderConfig();

	// Create model to populate the network, then set the outputs and create an engine
	ICudaEngine* engine = nullptr;

	engine = build_cls_engine(kBatchSize, builder, config, DataType::kFLOAT, m_gd, m_gw, m_weight_name);

	assert(engine != nullptr);

	// Serialize the engine
	IHostMemory* serialized_engine = engine->serialize();
	assert(serialized_engine != nullptr);

	// Save engine to file
	std::ofstream p(m_engine_name, std::ios::binary);
	if (!p) {
		std::cerr << "Could not open plan output file" << std::endl;
		assert(false);
	}
	p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

	// Close everything down
	engine->destroy();
	builder->destroy();
	config->destroy();
	serialized_engine->destroy();
}

// 反序列化
void CYoloCls::_deserialize_engine() {
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

void CYoloCls::_prepare_buffers() {
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

	m_cpu_input_buffer = new float[kBatchSize * 3 * kClsInputH * kClsInputW];
	m_cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void CYoloCls::_do_inference() {
	CUDA_CHECK(cudaMemcpyAsync(m_gpu_buffers[0], m_cpu_input_buffer, kBatchSize * 3 * kClsInputH * kClsInputW * sizeof(float), cudaMemcpyHostToDevice, m_stream));
	m_context->enqueue(kBatchSize, (void**)m_gpu_buffers, m_stream, nullptr);
	CUDA_CHECK(cudaMemcpyAsync(m_cpu_output_buffer, m_gpu_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
	cudaStreamSynchronize(m_stream);
}

void CYoloCls::SetEngineName(std::string engine_name) {
	m_engine_name = engine_name;
}

void CYoloCls::SetWeightName(std::string weight_name) {
	m_weight_name = weight_name;
}

IYoloCls* createYoloClsInstance()
{
	return new CYoloCls();
}