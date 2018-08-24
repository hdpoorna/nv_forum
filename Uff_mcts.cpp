#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"

#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:

    Logger(): Logger(Severity::kWARNING) {}

    Logger(Severity severity): reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};
};

static Logger gLogger;

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_mnist: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

inline int64_t volume(const Dims& d)
{
	int64_t v = 1;
	for (int64_t i = 0; i < d.nbDims; i++)
		v *= d.d[i];
	return v;
}

inline unsigned int elementSize(DataType t)
{
	switch (t)
	{
	case DataType::kINT32:
		// Fallthrough, same as kFLOAT
	case DataType::kFLOAT: return 4;
	case DataType::kHALF: return 2;
	case DataType::kINT8: return 1;
	}
	assert(0);
	return 0;
}

static const int INPUT_H = 9;
static const int INPUT_W = 9;

static const int BATCH_SIZE = 1;
static const int NUM_INPUT_CHANNELS = 10;

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

void* createMnistCudaBuffer(int64_t eltCount, DataType dtype)
{
    /* in that specific case, eltCount == INPUT_H * INPUT_W */
    assert(eltCount == BATCH_SIZE * INPUT_H * INPUT_W * NUM_INPUT_CHANNELS);
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    float* inputs = new float[eltCount];

    for (int k = 0; k < NUM_INPUT_CHANNELS; k++)
    {
        for (int i = 0; i < INPUT_W; i++)
        {
            for (int j = 0; j < INPUT_H; j++)
            {
                if (i == 3 && j == 0 && k == 3) inputs[(k*INPUT_H*INPUT_W) + (i * INPUT_H) + j] = 1.0f;
                else inputs[(k*INPUT_H*INPUT_W) + (j * INPUT_H) + i] = 0.0f;
            }    
        }
    }    

    void* deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

    delete[] inputs;
    return deviceMem;
}

void printOutput(int64_t eltCount, DataType dtype, void* buffer)
{
    std::cout << eltCount << " eltCount" << std::endl;
    assert(elementSize(dtype) == sizeof(float));
    std::cout << "--- OUTPUT ---" << std::endl;

    size_t memSize = eltCount * elementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = 0;
    for (int i = 0; i < eltCount; ++i)
        if (outputs[i] > outputs[maxIdx])
            maxIdx = i;

    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        std::cout << eltIdx << " => " << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
            std::cout << "***";
        std::cout << "\n";
    }

    std::cout << std::endl;
    delete[] outputs;
}

ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

#if 1
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
#else
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    builder->setFp16Mode(true);
#endif

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}

void execute(ICudaEngine& engine)
{  
    IExecutionContext* context = engine.createExecutionContext();    

    int nbBindings = engine.getNbBindings();
    assert(nbBindings == 2);

    std::vector<void*> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, BATCH_SIZE);
    
    buffers.clear();
    buffers.reserve(nbBindings);

    int bindingIdxInput = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (engine.bindingIsInput(i))
            bindingIdxInput = i;
        else
        {
            auto bufferSizesOutput = buffersSizes[i];            
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first * elementSize(bufferSizesOutput.second));
        }
    }

    auto bufferSizesInput = buffersSizes[bindingIdxInput];
    
    // read input
    buffers[bindingIdxInput] = createMnistCudaBuffer(bufferSizesInput.first, bufferSizesInput.second);
   
    auto t_start = std::chrono::high_resolution_clock::now();
    context->execute(BATCH_SIZE, &buffers[0]);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    
    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
    {
        if (engine.bindingIsInput(bindingIdx))
            continue;

        auto bufferSizesOutput = buffersSizes[bindingIdx];
        printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                        buffers[bindingIdx]);                
    }
    
    CHECK(cudaFree(buffers[bindingIdxInput]));
          
    std::cout << "Average is " << total << " ms." << std::endl;
    
    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
    context->destroy();
}

int main(int argc, char** argv)
{
    std::string fileName = "models/test4.uff";
    std::cout << fileName << std::endl;

    auto parser = createUffParser();

    /* Register tensorflow input */
    parser->registerInput("input_node", Dims3(10, 9, 9), UffInputOrder::kNCHW);
    //parser->registerOutput("p_output");   
    parser->registerOutput("v_output"); 

    ICudaEngine* engine = loadModelAndCreateEngine(fileName.c_str(), BATCH_SIZE, parser);

    if (!engine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");

    /* we need to keep the memory created by the parser */
    parser->destroy();

    int nbBindings = engine->getNbBindings();
    assert(nbBindings == 2);
        
    for (int kk = 0; kk < 10; kk++)
    {
        execute(*engine);    
    }    
    
    engine->destroy();
    shutdownProtobufLibrary();
    return EXIT_SUCCESS;
}
