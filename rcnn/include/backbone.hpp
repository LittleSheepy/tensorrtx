#pragma once
#include <vector>
#include <map>
#include <string>
#include "common.hpp"
#include "FasterRCNNcfg.h"
/* when stride>1, whether to put stride in the first 1x1 convolution or the bottleneck 3x3 convolution.
set false when use backbone from torchvision*/
#define STRIDE_IN_1X1 true
using namespace FasterRCNN;

static const std::map<RESNETTYPE, std::vector<int>> num_blocks_per_stage = {
    {R18, {2, 2, 2, 2}},
    {R34, {3, 4, 6, 3}},
    {R50, {3, 4, 6, 3}},
    {R101, {3, 4, 23, 3}},
    {R152, {3, 8, 36, 3}}
};

ILayer* BasicStem(INetworkDefinition *network,
	std::map<std::string, Weights>& weightMap,
	const std::string& lname, ITensor& input,
	int out_channels,
	int group_num = 1);

ITensor* BasicBlock(INetworkDefinition *network,
	std::map<std::string, Weights>& weightMap,
	const std::string& lname,
	ITensor& input,
	int in_channels,
	int out_channels,
	int stride = 1);

ITensor* BottleneckBlock(INetworkDefinition *network,
	std::map<std::string, Weights>& weightMap,
	const std::string& lname,
	ITensor& input,
	int in_channels,
	int bottleneck_channels,
	int out_channels,
	int stride = 1,
	int dilation = 1,
	int group_num = 1);
ITensor* MakeStage(INetworkDefinition *network,
	std::map<std::string, Weights>& weightMap,
	const std::string& lname,
	ITensor& input,
	int stage,
	RESNETTYPE resnet_type,
	int in_channels,
	int bottleneck_channels,
	int out_channels,
	int first_stride = 1,
	int dilation = 1);

ITensor* BuildResNet(INetworkDefinition *network,
	std::map<std::string, Weights>& weightMap,
	ITensor& input,
	RESNETTYPE resnet_type,
	int stem_out_channels,
	int bottleneck_channels,
	int res2_out_channels,
	int res5_dilation = 1);