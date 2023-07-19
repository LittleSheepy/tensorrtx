#include <iostream>
#include <opencv2/opencv.hpp>
#include "IYoloDet.h"
#include "IYoloCls.h"

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& img_dir) {
	if (argc < 4) return false;
	if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
		wts = std::string(argv[2]);
		engine = std::string(argv[3]);
		auto net = std::string(argv[4]);
		if (net[0] == 'n') {
			gd = 0.33;
			gw = 0.25;
		}
		else if (net[0] == 's') {
			gd = 0.33;
			gw = 0.50;
		}
		else if (net[0] == 'm') {
			gd = 0.67;
			gw = 0.75;
		}
		else if (net[0] == 'l') {
			gd = 1.0;
			gw = 1.0;
		}
		else if (net[0] == 'x') {
			gd = 1.33;
			gw = 1.25;
		}
		else if (net[0] == 'c' && argc == 7) {
			gd = atof(argv[5]);
			gw = atof(argv[6]);
		}
		else {
			return false;
		}
	}
	else if (std::string(argv[1]) == "-d" && argc == 5) {
		wts = std::string(argv[2]);
		engine = std::string(argv[3]);
		img_dir = std::string(argv[4]);
	}
	else {
		return false;
	}
	return true;
}
DLL_EXPORT void get_rect_ex(cv::Mat& img, float bbox[4], cv::Rect& rect);
int yolov5_main(int argc, char** argv) {
	std::string img_dir = "./images/";
	IYoloDet* yolo_det = createYoloDetInstance();
	yolo_det->Init();
	if (!img_dir.empty()) {
		std::vector<std::string> file_names;
		if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
			std::cerr << "read_files_in_dir failed." << std::endl;
			return -1;
		}
		for (int f = 0; f < (int)file_names.size(); f++) {

			cv::Mat img = cv::imread(img_dir + "/" + file_names[f]);
			if (img.empty()) continue;
			auto res = yolo_det->Predict(img);
			for (size_t j = 0; j < res.size(); j++) {
				cv::Rect r;
				get_rect_ex(img, res[j].bbox, r);
				cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
				cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			}
			cv::imwrite("images_result/" + file_names[f], img);
		}
	}
	else {
		cv::VideoCapture capture;
		bool res = capture.open(0);
		while (true) {
			cv::Mat img;
			bool red_res = capture.read(img);
			auto res = yolo_det->Predict(img);
			std::cout << img.cols << "," << img.rows << std::endl;

			cv::Mat img_lab(img.rows, img.cols + 200, CV_8UC3);
			//auto& res = batch_res[0];
			std::cout << "size:" << res.size() << std::endl;
			std::string str = "";
			std::string label_list[] = { "0","1","2", "3","4", "5","6", "7","8", "9",
								 "A", "b","C", "d","E", "F","J","h","P","t",
								 "L", "U", "u","o", "_" };
			for (size_t j = 0; j < res.size(); j++) {
				cv::Rect r;
				get_rect_ex(img, res[j].bbox, r);
				cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
				cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
				//cv::putText(img, std::string(label_list[(int)res[j].class_id]), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
				//str.append(std::string(label_list[(int)res[j].class_id]));
			}
			img.copyTo(img_lab(cv::Rect(0, 0, img.cols, img.rows)));

			cv::putText(img_lab, str, cv::Point(650, 200), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0xFF, 0xFF, 0xFF), 4);
			std::cout << "str:" << str << std::endl;
			cv::imshow("摄像头 ", img_lab);
			cv::waitKey(30);
		}
	}

}
/*
-d yolov5s.engine images
*/
int yolov5_cls_main(int argc, char** argv) {
	IYoloCls* yolo_cls = createYoloClsInstance();
	std::string wts_name = "";
	std::string engine_name = "";
	float gd = 0.0f, gw = 0.0f;
	std::string img_dir;

	if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir)) {
		std::cerr << "arguments not right!" << std::endl;
		std::cerr << "./yolov5_cls -s [.wts] [.engine] [n/s/m/l/x or c gd gw]  // serialize model to plan file" << std::endl;
		std::cerr << "./yolov5_cls -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
		return -1;
	}
	yolo_cls->SetWeightName(wts_name);
	yolo_cls->SetEngineName(engine_name);
	if (argv[1] == "-s") {
		yolo_cls->Init();
		return 0;
	}
	yolo_cls->Init();
	std::vector<std::string> file_names;
	if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
		std::cerr << "read_files_in_dir failed." << std::endl;
		return -1;
	}

	//auto classes = read_classes("imagenet_classes.txt");
	// batch predict
	for (size_t i = 0; i < file_names.size(); i++) {
		cv::Mat img = cv::imread(img_dir + "/" + file_names[i]);
		int idx = yolo_cls->Predict(img);
		std::cout << "  " << idx << std::endl;
	}
}

int main(int argc, char** argv) {
	yolov5_cls_main(argc, argv);
}