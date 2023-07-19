#include <numeric>
#include "common.h"
int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

void batch_preprocess(std::vector<cv::Mat>& imgs, float* output) {
    for (size_t b = 0; b < imgs.size(); b++) {
        cv::Mat img;
        cv::resize(imgs[b], img, cv::Size(kClsInputW, kClsInputH));
        int i = 0;
        for (int row = 0; row < img.rows; ++row) {
            uchar* uc_pixel = img.data + row * img.step;
            for (int col = 0; col < img.cols; ++col) {
                output[b * 3 * img.rows * img.cols + i] = ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229;  // R - 0.485
                output[b * 3 * img.rows * img.cols + i + img.rows * img.cols] = ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224;
                output[b * 3 * img.rows * img.cols + i + 2 * img.rows * img.cols] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
                uc_pixel += 3;
                ++i;
            }
        }
    }
}

std::vector<float> softmax(float* prob, int n) {
    std::vector<float> res;
    float sum = 0.0f;
    float t;
    for (int i = 0; i < n; i++) {
        t = expf(prob[i]);
        res.push_back(t);
        sum += t;
    }
    for (int i = 0; i < n; i++) {
        res[i] /= sum;
    }
    return res;
}

std::vector<int> topk(const std::vector<float>& vec, int k) {
    std::vector<int> topk_index;
    std::vector<size_t> vec_index(vec.size());
    std::iota(vec_index.begin(), vec_index.end(), 0);

    std::sort(vec_index.begin(), vec_index.end(), [&vec](size_t index_1, size_t index_2) { return vec[index_1] > vec[index_2]; });

    int k_num = std::min<int>(vec.size(), k);

    for (int i = 0; i < k_num; ++i) {
        topk_index.push_back(vec_index[i]);
    }

    return topk_index;
}

std::vector<std::string> read_classes(std::string file_name) {
    std::vector<std::string> classes;
    std::ifstream ifs(file_name, std::ios::in);
    if (!ifs.is_open()) {
        std::cerr << file_name << " is not found, pls refer to README and download it." << std::endl;
        assert(0);
    }
    std::string s;
    while (std::getline(ifs, s)) {
        classes.push_back(s);
    }
    ifs.close();
    return classes;
}