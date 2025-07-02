#include "stdafx.h"
#include "CudaHelper.h"

//

CUdevice CudaHelper::_selectedDevice = 0;

// Public methods

CudaHelper::CudaHelper()
{
}

CudaHelper::~CudaHelper()
= default;

glm::uint CudaHelper::getMaxThreadsBlock()
{
    cudaDeviceProp prop;
    checkError(cudaGetDeviceProperties(&prop, _selectedDevice));

    return prop.maxThreadsPerBlock;
}

glm::ivec3 CudaHelper::getMaxBlockDimensions()
{
    cudaDeviceProp prop;
    checkError(cudaGetDeviceProperties(&prop, _selectedDevice));
	return { prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] };
}

glm::ivec3 CudaHelper::getMaxGridSize()
{
	cudaDeviceProp prop;
	checkError(cudaGetDeviceProperties(&prop, _selectedDevice));
	return { prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] };
}

void CudaHelper::synchronize(const std::string& kernelName)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        std::cerr << "CUDA error in " << kernelName << ": " << cudaGetErrorString(error) << '\n';
    CudaHelper::checkError(cudaDeviceSynchronize());
}

void CudaHelper::startTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{
    checkError(cudaEventCreate(&startEvent));
    checkError(cudaEventCreate(&stopEvent));
    checkError(cudaEventRecord(startEvent, nullptr));
}

float CudaHelper::stopTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{
    float ms;
    checkError(cudaEventRecord(stopEvent, 0));
    checkError(cudaEventSynchronize(stopEvent));
    checkError(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    return ms;
}

// Protected methods

void CudaHelper::checkError(cudaError_t result)
{
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}