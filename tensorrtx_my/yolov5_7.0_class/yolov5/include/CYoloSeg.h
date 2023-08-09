#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>

#include "cuda_utils.h"


#include "logging.h"
//#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
//#include "model.h"
#include "types.h"
#include "config.h"


#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

#include "IYoloSeg.h"
using namespace nvinfer1;
const static int kOutputSize1 = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSize2 = 32 * (kInputH / 4) * (kInputW / 4);
static Logger gLogger;
class CYoloSeg:
    public IYoloSeg
{
public:
    CYoloSeg();
    ~CYoloSeg();
    void Init();
    std::vector<Detection> Predict(cv::Mat& img, 
        std::vector<std::vector<Detection>>& res_batch, std::vector<std::vector<std::vector<cv::Point>>>& contours);
public:
    void SetEngineName(std::string engine_name);
    void SetWeightName(std::string weight_name);
private:
    void _deserialize_engine();
    void _prepare_buffers();
    void _do_inference();

private:
    IRuntime* m_runtime = nullptr;                  // 
    ICudaEngine* m_engine = nullptr;                // 
    IExecutionContext* m_context = nullptr;         // 
    cudaStream_t        m_stream = nullptr;         // 
    float* m_gpu_buffers[3];                        //
    float* m_cpu_output_buffer1 = nullptr;          // 
    float* m_cpu_output_buffer2 = nullptr;          // 
    uint8_t* m_img_buffer_host = nullptr;
    uint8_t* m_img_buffer_device = nullptr;
    std::string         m_weight_name = "yolov5_7.0_seg_LG.wts";
    std::string         m_engine_name = "yolov5_7.0_seg_LG.engine";
    //float               input[BATCH_SIZE * 3 * INPUT_H * INPUT_W];        // 图片数据
    //float               output[BATCH_SIZE * OUTPUT_SIZE];                // 输出数据
    int                 m_inputIndex;
    int                 m_outputIndex1;
    int                 m_outputIndex2;
};

