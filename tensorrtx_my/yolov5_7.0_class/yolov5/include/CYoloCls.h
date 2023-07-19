#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>

#include "cuda_utils.h"


#include "logging.h"
#include "model.h"
#include "preprocess.h"
#include "postprocess.h"
#include "common.h"
#include "types.h"
#include "config.h"


#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>

#include "IYoloCls.h"

using namespace nvinfer1;
const static int kOutputSize = kClsNumClass;
static Logger gLogger;
class CYoloCls :
    public IYoloCls
{
public:
    CYoloCls();
    ~CYoloCls();
    void Init();
    void Serialize();
    int Predict(cv::Mat& img);
public:
    void SetEngineName(std::string engine_name);
    void SetWeightName(std::string weight_name);
private:
    void _serialize_engine();
    void _deserialize_engine();
    void _prepare_buffers();
    void _do_inference();

private:
    IRuntime*           m_runtime = nullptr;                // 
    ICudaEngine*        m_engine = nullptr;                 // 
    IExecutionContext*  m_context = nullptr;                // 
    cudaStream_t        m_stream = nullptr;                 // 
    float*              m_gpu_buffers[2];                   //
    float*              m_cpu_input_buffer = nullptr;       // 
    float*              m_cpu_output_buffer = nullptr;      // 
    std::string         m_weight_name = "yolov5_7.0_cls_LG.wts";
    std::string         m_engine_name = "yolov5_7.0_cls_LG.engine";
    int                 m_inputIndex;
    int                 m_outputIndex;
    float               m_gd=0.33;
    float               m_gw=0.50;
};

