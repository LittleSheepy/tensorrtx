#pragma once
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <fstream>
#include <vector>
#include "types.h"

DLL_EXPORT int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names);

void batch_preprocess(std::vector<cv::Mat>& imgs, float* output);
std::vector<float> softmax(float* prob, int n);
std::vector<int> topk(const std::vector<float>& vec, int k);
std::vector<std::string> read_classes(std::string file_name);

