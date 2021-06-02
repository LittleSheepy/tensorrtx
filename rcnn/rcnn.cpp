#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
//#include "backbone.hpp"
//#include "RpnDecodePlugin.h"
//#include "RpnNmsPlugin.h"
//#include "RoiAlignPlugin.h"
//#include "PredictorDecodePlugin.h"
//#include "BatchedNmsPlugin.h"
//#include "MaskRcnnInferencePlugin.h"
//#include "calibrator.hpp"

#include "FasterRCNNcfg.h"
#include "Faster_RCNN.h"

using namespace ObjDet;
using namespace FasterRCNN;
using namespace std;


bool parse_args(int argc, char** argv, std::string& wtsFile, std::string& engineFile, std::string& imgDir) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s") {
        wtsFile = std::string(argv[2]);
        engineFile = std::string(argv[3]);
    } else if (std::string(argv[1]) == "-d") {
        engineFile = std::string(argv[2]);
        imgDir = std::string(argv[3]);
    } else {
        return false;
    }
    if (argc >= 5 && std::string(argv[4]) == "m") MASK_ON = true;
    return true;
}
int main(int argc, char** argv) {
	std::string wtsFile = "";
	std::string engineFile = "";

	std::string imgDir;
	if (!parse_args(argc, argv, wtsFile, engineFile, imgDir)) {
		std::cerr << "arguments not right!" << std::endl;
		std::cerr << "./rcnn -s [.wts] [.engine] [m] // serialize model to plan file" << std::endl;
		std::cerr << "./rcnn -d [.engine] ../samples [m]  // deserialize plan file and run inference" << std::endl;
		return -1;
	}
	Faster_RCNN * rcnn = new Faster_RCNN();
	rcnn->Init();
	if (!wtsFile.empty()) {
		rcnn->serialize(wtsFile, engineFile);
		return 0;
	}
	//if (argc == 2 && std::string(argv[1]) == "-s") {
	//	IHostMemory* modelStream{ nullptr };
	//	rcnn->APIToModel(BATCH_SIZE, &modelStream);
	//	assert(modelStream != nullptr);
	//	std::ofstream p("../yolov4.engine", std::ios::binary);
	//	if (!p) {
	//		std::cerr << "could not open plan output file" << std::endl;
	//		return -1;
	//	}
	//	p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	//	modelStream->destroy();
	//	return 0;
	//}

	rcnn->LoadEngine(engineFile);

	cv::VideoCapture capture;
	bool res = capture.open(0);
	while (true) {
		cv::Mat img;
		bool red_res = capture.read(img);
		auto res = rcnn->predict(img);
		cout << img.cols << "," << img.rows << endl;

		//std::cout << "size:" << res.size() << std::endl;
		for (size_t j = 0; j < res.size(); j++) {
			cv::Rect r(res[j].bbox[0], res[j].bbox[1], res[j].bbox[2], res[j].bbox[3]);
			cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
			cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
		}
		cv::imshow("ÉãÏñÍ· ", img);
		cv::waitKey(30);
	}


	return 1;
}