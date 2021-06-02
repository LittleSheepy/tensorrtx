#pragma once
#include <opencv2/opencv.hpp>
#include "./logging.h"
namespace FasterRCNN {
	#define DEVICE 0
	#define BATCH_SIZE 1
	#define BACKBONE_RESNETTYPE R50
	// data
	static const std::vector<float> PIXEL_MEAN = { 103.53, 116.28, 123.675 };
	static const std::vector<float> PIXEL_STD = { 1.0, 1.0, 1.0 };
	static constexpr float MIN_SIZE = 800.0;
	static constexpr float MAX_SIZE = 1333.0;
	static constexpr int NUM_CLASSES = 80;
	static constexpr int INPUT_H = 480;
	static constexpr int INPUT_W = 640;
	static int IMAGE_HEIGHT = 800;
	static int IMAGE_WIDTH = 1333;
	static constexpr int LOCATIONS = 4;
	struct alignas(float) Detection {
		//x y w h
		float bbox[LOCATIONS];
		float score;
		float class_id;
	};
	// backbone
	enum RESNETTYPE {
		R18 = 0,
		R34,
		R50,
		R101,
		R152
	};



	static const int RES2_OUT_CHANNELS = (BACKBONE_RESNETTYPE == R18 ||
		BACKBONE_RESNETTYPE == R34) ? 64 : 256;
	// rpn
	static const std::vector<float> ANCHOR_SIZES = { 32, 64, 128, 256, 512 };
	static const std::vector<float> ASPECT_RATIOS = { 0.5, 1.0, 2.0 };
	static constexpr int PRE_NMS_TOP_K_TEST = 6000;
	static constexpr float RPN_NMS_THRESH = 0.7;
	static constexpr int POST_NMS_TOPK = 1000;
	// roialign
	static constexpr int STRIDES = 16;
	static constexpr int SAMPLING_RATIO = 0;
	static constexpr int POOLER_RESOLUTION = 14;
	// roihead
	static constexpr float NMS_THRESH_TEST = 0.5;
	static constexpr int DETECTIONS_PER_IMAGE = 100;
	static constexpr float SCORE_THRESH = 0.6;
	static const std::vector<float> BBOX_REG_WEIGHTS = { 10.0, 10.0, 5.0, 5.0 };
	static bool MASK_ON = false;

	static const char* INPUT_NODE_NAME = "images";
	static const std::vector<std::string> OUTPUT_NAMES = { "scores", "boxes",
	"labels", "masks" };

	static Logger gLogger;
}





