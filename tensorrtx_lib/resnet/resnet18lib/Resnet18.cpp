/*******************************************************************
** 文件名:	Resnet18.cpp
** 版  权:	(C) littlesheepy 2023 - All Rights Reserved
** 创建人:	littlesheepy
** 日  期:	2023年4月9日
** 版  本:	1.0
** 描  述:
** 应  用:
**************************** 修改记录 ******************************
** 修改人:
** 日  期:
** 描  述:
********************************************************************/
#include "Resnet18.h"
#include "common.h"
#include "resnet18cfg.h"

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
namespace ObjDet {
	Resnet18::Resnet18()
	{
	}
	Resnet18::~Resnet18()
	{
		// Release stream and buffers
		//cudaStreamDestroy(this->m_stream);
		//CHECK(cudaFree(this->buffers[inputIndex]));
		//CHECK(cudaFree(this->buffers[outputIndex]));
		//// Destroy the engine
		//m_context->destroy();
		//m_engine->destroy();
		//m_runtime->destroy();
	}

	void Resnet18::Init(std::string& engine_name)
	{
		cudaSetDevice(DEVICE);
		m_engine_name = engine_name;
		// cfg
		//Config cfg("cfg.txt");
	}

    // Creat the engine using only the API and not any parser.
    ICudaEngine* Resnet18::createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
    {
        INetworkDefinition* network = builder->createNetworkV2(0U);

        // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
        ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
        assert(data);

        std::map<std::string, Weights> weightMap = loadWeights("F:/03weights/08trtx_wts/resnet/resnet18.wts");
        Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

        IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{ 7, 7 }, weightMap["conv1.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{ 2, 2 });
        conv1->setPaddingNd(DimsHW{ 3, 3 });

        IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        assert(relu1);

        IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
        assert(pool1);
        pool1->setStrideNd(DimsHW{ 2, 2 });
        pool1->setPaddingNd(DimsHW{ 1, 1 });

        IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
        IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");

        IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "layer2.0.");
        IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "layer2.1.");

        IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "layer3.0.");
        IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "layer3.1.");

        IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "layer4.0.");
        IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "layer4.1.");

        IPoolingLayer* pool2 = network->addPoolingNd(*relu9->getOutput(0), PoolingType::kAVERAGE, DimsHW{ 7, 7 });
        assert(pool2);
        pool2->setStrideNd(DimsHW{ 1, 1 });

        IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
        assert(fc1);

        fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
        std::cout << "set name out" << std::endl;
        network->markOutput(*fc1->getOutput(0));

        // Build engine
        builder->setMaxBatchSize(maxBatchSize);
        config->setMaxWorkspaceSize(1 << 20);
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        std::cout << "build out" << std::endl;

        // Don't need the network any more
        network->destroy();

        // Release host memory
        for (auto& mem : weightMap)
        {
            free((void*)(mem.second.values));
        }

        return engine;
    }

    void Resnet18::APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
    {
        // Create builder
        IBuilder* builder = createInferBuilder(gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();

        // Create model to populate the network, then set the outputs and create an engine
        ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
        assert(engine != nullptr);

        // Serialize the engine
        (*modelStream) = engine->serialize();

        // Close everything down
        engine->destroy();
        builder->destroy();
        config->destroy();
    }

    int Resnet18::serialize(std::string& wts_name, std::string& engine_name, bool& is_p6, float& gd, float& gw) {
        if (wts_name.empty()) { return -1; }
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    int Resnet18::LoadEngine() {
        // create a model using the API directly and serialize it to a stream
        char* trtModelStream{ nullptr };
        size_t size{ 0 };
        // std::string engine_name = STR2(NET);
        std::string engine_name = "F:/03weights/08trtx_wts/yolov5/resnet18.engine";
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        m_runtime = createInferRuntime(gLogger);
        assert(m_runtime != nullptr);
        m_engine = m_runtime->deserializeCudaEngine(trtModelStream, size);
        assert(m_engine != nullptr);
        this->m_context = m_engine->createExecutionContext();
        assert(this->m_context != nullptr);
        delete[] trtModelStream;

        assert(m_engine->getNbBindings() == 2);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        inputIndex = m_engine->getBindingIndex(INPUT_BLOB_NAME);
        outputIndex = m_engine->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        // Create stream
        CHECK(cudaStreamCreate(&m_stream));
    }
    void Resnet18::doInference(int batchSize)
    {
        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], this->input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        this->m_context->enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(this->output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    float* Resnet18::predict(cv::Mat& img) {
        float* data = this->input;
        cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
        int i = 0;
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < INPUT_W; ++col) {
                data[i] = (float)uc_pixel[2] / 255.0;
                data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        return this->output;
    }
}
