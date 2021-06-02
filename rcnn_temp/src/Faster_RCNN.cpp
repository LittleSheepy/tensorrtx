#include "NvInferRunTime.h"

#include "Faster_RCNN.h"
#include "FasterRCNNcfg.h"
#include "cuda_utils.h"
using namespace nvinfer1;
using namespace FasterRCNN;
 Faster_RCNN:: Faster_RCNN()
{
}

 Faster_RCNN::~Faster_RCNN()
 {
 }

 void Faster_RCNN::Init()
 {

 }

 void Faster_RCNN::doInference(IExecutionContext& context, cudaStream_t& stream, std::vector<void*>& buffers,
	 std::vector<float>& input, std::vector<float*>& output) {
	 CUDA_CHECK(cudaMemcpyAsync(buffers[0], input.data(), BATCH_SIZE * INPUT_H * INPUT_W * 3 * sizeof(float),
		 cudaMemcpyHostToDevice, stream));

	 context.enqueue(BATCH_SIZE, buffers.data(), stream, nullptr);

	 CUDA_CHECK(cudaMemcpyAsync(output[0], buffers[1], BATCH_SIZE * DETECTIONS_PER_IMAGE * sizeof(float),
		 cudaMemcpyDeviceToHost, stream));
	 CUDA_CHECK(cudaMemcpyAsync(output[1], buffers[2], BATCH_SIZE * DETECTIONS_PER_IMAGE * 4 * sizeof(float),
		 cudaMemcpyDeviceToHost, stream));
	 CUDA_CHECK(cudaMemcpyAsync(output[2], buffers[3], BATCH_SIZE * DETECTIONS_PER_IMAGE * sizeof(float),
		 cudaMemcpyDeviceToHost, stream));
	 if (MASK_ON)
		 CUDA_CHECK(cudaMemcpyAsync(output[3], buffers[4],
			 BATCH_SIZE * DETECTIONS_PER_IMAGE * POOLER_RESOLUTION * POOLER_RESOLUTION * sizeof(float),
			 cudaMemcpyDeviceToHost, stream));

	 cudaStreamSynchronize(stream);
 }

