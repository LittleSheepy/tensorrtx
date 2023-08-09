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

#include "IYoloDet.h"

using namespace nvinfer1;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
static Logger gLogger;
class CYoloDet :
    public IYoloDet
{
public:
    CYoloDet();
    ~CYoloDet();
    void Init();
    std::vector<Detection> Predict(cv::Mat& img);
private:
    void _deserialize_engine();
    void _prepare_buffers();
    void _do_inference();

private:
    IRuntime*           m_runtime   = nullptr;              // 
    ICudaEngine*        m_engine    = nullptr;              // 
    IExecutionContext*  m_context   = nullptr;              // 
    cudaStream_t        m_stream    = nullptr;              // 
    float *             m_gpu_buffers[2];                   //
    float *             m_cpu_output_buffer = nullptr;      // 
    uint8_t* m_img_buffer_host = nullptr;
    uint8_t* m_img_buffer_device = nullptr;
    std::string         m_engine_name = "yolov5_7.0_LG.engine";
    //float               input[BATCH_SIZE * 3 * INPUT_H * INPUT_W];        // 图片数据
    //float               output[BATCH_SIZE * OUTPUT_SIZE];                // 输出数据
    int                 m_inputIndex;
    int                 m_outputIndex;
};

