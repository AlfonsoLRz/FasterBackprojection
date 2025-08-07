#include "stdafx.h"
#include "CudaContext.h"

#include "CudaHelper.h"

CudaContext::CudaContext()
{
	// Initialize CUDA
	CUresult cuErr = cuInit(0);
	if (cuErr != CUDA_SUCCESS)
		throw std::runtime_error("CUDA Driver initialization failed");

	CUdevice device = setDevice();

	//
	cuDeviceGet(&device, 0);
	cuCtxCreate(&_cudaContext, nullptr, 0, device);
	if (_cudaContext == nullptr)
		throw std::runtime_error("Failed to create CUDA context");
}

CUdevice CudaContext::setDevice(uint8_t deviceIndex)
{
    // Pick device
    int numDevices;
    CUdevice selectedDevice = 0;
    CudaHelper::checkError(cudaGetDeviceCount(&numDevices));

    if (deviceIndex == UINT8_MAX)
    {
        size_t bestScore = 0;
        for (int deviceIdx = 0; deviceIdx < numDevices; deviceIdx++)
        {
            int clockRate;
            int numProcessors;
            CudaHelper::checkError(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, deviceIdx));
            CudaHelper::checkError(cudaDeviceGetAttribute(&numProcessors, cudaDevAttrMultiProcessorCount, deviceIdx));

            size_t score = clockRate * numProcessors;
            if (score > bestScore)
            {
	            selectedDevice = deviceIdx;
                bestScore = score;
            }
        }

        if (bestScore == 0)
            throw std::runtime_error("CudaModule: No appropriate CUDA device found!");
    }
    else
    {
        selectedDevice = glm::clamp(deviceIndex, static_cast<uint8_t>(0), static_cast<uint8_t>(numDevices));
    }

    CudaHelper::checkError(cudaSetDevice(selectedDevice));

	return selectedDevice;
}

CudaContext::~CudaContext()
{
	if (_cudaContext)
	{
		cuCtxDestroy(_cudaContext);
		_cudaContext = nullptr;
	}
}
