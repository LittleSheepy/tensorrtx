#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include <opencv2/opencv.hpp>
//#include "logging.h"
#include "common.hpp"
#include "yolov5.h"
#include "cfg.h"
//#include "cJSON.h"
using namespace DeepBlueDet;
using namespace nvinfer1;
using namespace std;
cv::Rect get_rect(cv::Mat& img, float bbox[4]);
//std::string IMG_NAME;
int main1(int argc, char** argv) {
	/**/
	// 配置
	Config cfg("cfg.txt");
	cout << cfg.getLines() << endl;
	cout << "X1=" << cfg.getCFG("X1") << endl;
	//cout << cfg.getCFG("cxx") << endl;
	int X1 = atoi(cfg.getCFG("X1").c_str());
	int Y1 = atoi(cfg.getCFG("Y1").c_str());
	int X2 = atoi(cfg.getCFG("X2").c_str());
	int Y2 = atoi(cfg.getCFG("Y2").c_str());


	/*
	// cJson
	FILE *f;//输入文件
	long len;//文件长度
	char *content;//文件内容
	cJSON *json;//封装后的json对象
	f = fopen("./cfg.json", "rb");
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);
	content = (char*)malloc(len + 1);
	fread(content, 1, len, f);
	fclose(f);

	json = cJSON_Parse(content);
	if (!json) {
		printf("Error before: [%s]\n", cJSON_GetErrorPtr());
	}
	cJSON *name = cJSON_GetObjectItem(json, "name");
	cout << "cjson name=" << name->valuestring << endl;

	cJSON *Roi = cJSON_GetObjectItem(json, "aaa");
	cout << "cjson name=" << Roi->valuestring << endl;
	*/

	YOLOV5* yolov5 = new YOLOV5();
	yolov5->Init();
	cv::VideoCapture capture;
	bool res1 = capture.isOpened();
	bool res = capture.open(0);
	while (true) {
		cv::Mat img_org;
		bool red_res = capture.read(img_org);
		//capture >> img;
		cv::Mat img = img_org(cv::Rect(X1, Y1, X2 - X1, Y2 - Y1));
		auto res = yolov5->predict(img);
		cout << img.cols << "," << img.rows << endl;

		cv::Mat img_lab(img.rows, img.cols + 200, CV_8UC3);
		//auto& res = batch_res[0];
        std::cout << "size:" <<res.size() << std::endl;
        std::string str = "";
		std::string label_list[] = { "0","1","2", "3","4", "5","6", "7","8", "9",
							 "A", "b","C", "d","E", "F","J","h","P","t",
							 "L", "U", "u","o", "_" };
		for (size_t j = 0; j < res.size(); j++) {
			cv::Rect r = get_rect(img, res[j].bbox);
			cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::string(label_list[(int)res[j].class_id]), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            str.append(std::string(label_list[(int)res[j].class_id]));
        }
		img.copyTo(img_lab(cv::Rect(0, 0, img.cols, img.rows)));

		cv::putText(img_lab, str, cv::Point(650, 200), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0xFF, 0xFF, 0xFF), 4);
        std::cout << "str:"<<str<<std::endl;
		cv::imshow("摄像头 ", img_lab);
		cv::waitKey(30); 
	}
	return 0;
}

int main2(int argc, char** argv) {
	/**/
	// 配置
	Config cfg("cfg.txt");
	cout << cfg.getLines() << endl;
	cout << "X1=" << cfg.getCFG("X1") << endl;
	//cout << cfg.getCFG("cxx") << endl;
	int X1 = atoi(cfg.getCFG("X1").c_str());
	int Y1 = atoi(cfg.getCFG("Y1").c_str());
	int X2 = atoi(cfg.getCFG("X2").c_str());
	int Y2 = atoi(cfg.getCFG("Y2").c_str());

	YOLOV5* yolov5 = new YOLOV5();
	yolov5->Init();
	std::vector<std::string> file_names;
	if (read_files_in_dir("../images/", file_names) < 0) {
		std::cout << "read_files_in_dir failed." << std::endl;
		return -1;
	}
	int fcount = 0;

	std::vector<std::string> rec_label_list; 
	std::string path = "../predict.txt";
	std::ofstream fout(path);
	for (int f = 0; f < (int)file_names.size(); f++) {
		rec_label_list.clear();
		std::string strFileName = file_names[fcount++];
		//IMG_NAME = strFileName;
		cv::Mat img_org = cv::imread("../images/" + strFileName);
		if (img_org.empty()) continue;
		cv::Mat img = img_org(cv::Rect(X1, Y1, X2 - X1, Y2 - Y1));
		cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
		rec_label_list = yolov5->predict_str(pr_img);
		std::cout << "rec_label_list:" << rec_label_list[0] << std::endl;
		if (rec_label_list.size() > 0) {
			std::cout << strFileName << "result : " << rec_label_list[0] << std::endl;
			fout << strFileName << ";" << rec_label_list[0] << std::endl;
		}
		else {
			std::cout << file_names[fcount++] << "result : " << "null" << std::endl;
			fout << strFileName << ";" << "null" << std::endl;
		}
		//cv::imwrite(file_names[fcount], img)
	}
	fout.close();
	while (true) {}
}

int main(int argc, char** argv) {
	/**/
	// 配置
	Config cfg("cfg.txt");
	cout << cfg.getLines() << endl;
	cout << "X1=" << cfg.getCFG("X1") << endl;
	//cout << cfg.getCFG("cxx") << endl;
	int X1 = atoi(cfg.getCFG("X1").c_str());
	int Y1 = atoi(cfg.getCFG("Y1").c_str());
	int X2 = atoi(cfg.getCFG("X2").c_str());
	int Y2 = atoi(cfg.getCFG("Y2").c_str());

	YOLOV5* yolov5 = new YOLOV5();
	yolov5->Init();
	std::vector<std::string> file_names;
	if (read_files_in_dir("../images/", file_names) < 0) {
		std::cout << "read_files_in_dir failed." << std::endl;
		return -1;
	}
	int fcount = 0;

	std::vector<std::string> rec_label_list;
	std::string path = "../predict.txt";
	std::ofstream fout(path);
	for (int f = 0; f < (int)file_names.size(); f++) {
		rec_label_list.clear();
		std::string strFileName = file_names[fcount++];
		//IMG_NAME = strFileName;
		cv::Mat img_org = cv::imread("../images/" + strFileName);
		if (img_org.empty()) continue;
		cv::Mat img = img_org(cv::Rect(X1, Y1, X2 - X1, Y2 - Y1));
		//cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
		auto res = yolov5->predict(img);
		cout << img.cols << "," << img.rows << endl;

		cv::Mat img_lab(img.rows, img.cols + 200, CV_8UC3);
		//auto& res = batch_res[0];
		std::cout << "size:" << res.size() << std::endl;
		std::string str = "";
		std::string label_list[] = { "0","1","2", "3","4", "5","6", "7","8", "9",
							 "A", "b","C", "d","E", "F","J","h","P","t",
							 "L", "U", "u","o", "_" };
		for (size_t j = 0; j < res.size(); j++) {
			cv::Rect r = get_rect(img, res[j].bbox);
			cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
			cv::putText(img, std::string(label_list[(int)res[j].class_id]), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			str.append(std::string(label_list[(int)res[j].class_id]));
		}
		img.copyTo(img_lab(cv::Rect(0, 0, img.cols, img.rows)));

		cv::putText(img_lab, str, cv::Point(650, 200), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0xFF, 0xFF, 0xFF), 4);
		std::cout << "str:" << str << std::endl;
		cv::imwrite("../result/result.jpg", img_lab);
		//cv::imshow("摄像头 ", img_lab);
		//cv::waitKey(30);
	}
	fout.close();
	while (true) {}
}