#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <dirent.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "./logging.h"
#include "./cuda_utils.h"

using namespace nvinfer1;

void loadWeights(const std::string file, std::map<std::string, Weights>& weightMap);

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);

cv::Mat preprocessImg(cv::Mat& img, int input_w, int input_h);
