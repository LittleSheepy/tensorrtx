#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

#include "logging.h"
#include <dirent.h>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs

using namespace nvinfer1;

static Logger gLogger;
cv::Mat preprocess_img(cv::Mat & img, int input_w, int input_h);
// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file);

IScaleLayer* addBatchNorm2d(INetworkDefinition * network, std::map<std::string, Weights>&weightMap, ITensor & input, std::string lname, float eps);

IActivationLayer* basicBlock(INetworkDefinition * network, std::map<std::string, Weights>&weightMap, ITensor & input, int inch, int outch, int stride, std::string lname);
int read_files_in_dir(const char* p_dir_name, std::vector<std::string>&file_names);

#endif