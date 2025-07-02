#pragma once

class CudaHelper
{
protected:
	static CUdevice _selectedDevice;

public:
	CudaHelper();
	virtual ~CudaHelper();
	
	static void checkError(cudaError_t result);

	template<typename T>
	static void uploadDataToGPU(T*& bufferPointer, T* buffer, size_t size);

	template<typename T>
	static void downloadBufferGPU(T*& bufferPointer, T* buffer, size_t size, size_t offset = 0);

	template<typename T>
	static void free(T*& bufferPointer) { cudaFree(bufferPointer); }

	template<typename T>
	static void freeHost(T*& bufferPointer) { cudaFreeHost(bufferPointer); }

	static glm::uint getMaxThreadsBlock();
	static glm::ivec3 getMaxBlockDimensions();
	static glm::ivec3 getMaxGridSize();
	static glm::uint getNumBlocks(glm::uint size, glm::uint blockThreads) { return (size + blockThreads) / blockThreads; }

	template<typename T>
	static void initializeBufferGPU(T*& bufferPointer, size_t size, T* buffer = nullptr);

	template<typename T>
	static void initializeZeroBufferGPU(T*& bufferPointer, size_t size);

	static void synchronize(const std::string& kernelName = "");

	static void startTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent);

	static float stopTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent);
};

template <typename T>
void CudaHelper::uploadDataToGPU(T*& bufferPointer, T* buffer, size_t size)
{
	CudaHelper::checkError(cudaMemcpy(bufferPointer, buffer, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void CudaHelper::downloadBufferGPU(T*& bufferPointer, T* buffer, size_t size, size_t offset)
{
	CudaHelper::checkError(cudaMemcpy(buffer, bufferPointer + offset, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template<typename T>
void CudaHelper::initializeBufferGPU(T*& bufferPointer, size_t size, T* buffer)
{
	CudaHelper::checkError(cudaMalloc((void**)&bufferPointer, size * sizeof(T)));
	if (buffer)
		CudaHelper::checkError(cudaMemcpy(bufferPointer, buffer, size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void CudaHelper::initializeZeroBufferGPU(T*& bufferPointer, size_t size)
{
	CudaHelper::checkError(cudaMalloc((void**)&bufferPointer, size * sizeof(T)));
	CudaHelper::checkError(cudaMemset(bufferPointer, 0, size * sizeof(T)));
}
