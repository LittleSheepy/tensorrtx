#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include "Common.h"
#include "Resnet18.h"

using namespace std;
using namespace nvinfer1;
using namespace ObjDet;
int main() {
    std::string wts_name = "";
    std::string engine_name = "F:/03weights/08trtx_wts/yolov5/resnet18.engine";
    std::string img_dir = "";
    Resnet18* resnet18 = new Resnet18();
    resnet18->Init(engine_name);
    resnet18->LoadEngine();

    if (!img_dir.empty()) {
        std::vector<std::string> file_names;
        if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
            std::cerr << "read_files_in_dir failed." << std::endl;
            return -1;
        }
        for (int f = 0; f < (int)file_names.size(); f++) {

            cv::Mat img = cv::imread(img_dir + "/" + file_names[f]);
            if (img.empty()) continue;
            auto res = resnet18->predict(img);
            cout << res[0] << endl;
        }
    }
    else {
        cv::VideoCapture capture;
        bool res = capture.open(0);
        while (true) {
            cv::Mat img;
            bool red_res = capture.read(img);
            auto res = resnet18->predict(img);
            cout << res[0] << endl;
            cout << img.cols << "," << img.rows << endl;

            cv::Mat img_lab(img.rows, img.cols + 200, CV_8UC3);
            //auto& res = batch_res[0];
            std::string str = "";
            //std::cout << "size:" << res.size() << std::endl;
            //std::string label_list[] = { "0","1","2", "3","4", "5","6", "7","8", "9",
            //                     "A", "b","C", "d","E", "F","J","h","P","t",
            //                     "L", "U", "u","o", "_" };
            //for (size_t j = 0; j < res.size(); j++) {
            //    cv::Rect r = get_rect(img, res[j].bbox);
            //    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            //    cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            //    //cv::putText(img, std::string(label_list[(int)res[j].class_id]), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            //    //str.append(std::string(label_list[(int)res[j].class_id]));
            //}
            img.copyTo(img_lab(cv::Rect(0, 0, img.cols, img.rows)));

            cv::putText(img_lab, str, cv::Point(650, 200), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0xFF, 0xFF, 0xFF), 4);
            std::cout << "str:" << str << std::endl;
            cv::imshow("摄像头 ", img_lab);
            cv::waitKey(30);
        }
    }
}

/*
int main(int argc, char** argv)
{
    //if (argc != 2) {
    //    std::cerr << "arguments not right!" << std::endl;
    //    std::cerr << "./resnet18 -s   // serialize model to plan file" << std::endl;
    //    std::cerr << "./resnet18 -d   // deserialize plan file and run inference" << std::endl;
    //    return -1;
    //}

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    std::string s_d = "-d";
    if (s_d == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("F:/03weights/08trtx_wts/resnet/resnet18.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (s_d == "-d") {
        std::ifstream file("F:/03weights/08trtx_wts/resnet/resnet18.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }


    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[OUTPUT_SIZE - 10 + i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}
*/
