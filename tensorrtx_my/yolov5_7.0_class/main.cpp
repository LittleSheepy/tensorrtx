#include <iostream>
#include <opencv2/opencv.hpp>
#include "IYoloDet.h"


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
			cv::imshow("ÉãÏñÍ· ", img_lab);
			cv::waitKey(30);
		}
	}

}

int main(int argc, char** argv) {
	yolov5_main(argc, argv);
}