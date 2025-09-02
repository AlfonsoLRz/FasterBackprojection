#pragma once

class CudaHelper
{
protected:
	static CUdevice								_selectedDevice;
	static size_t								_allocatedMemory;
	static std::unordered_map<void*, size_t>	_allocatedPointers;

public:
	CudaHelper();
	virtual ~CudaHelper();
	
	static void checkError(cudaError_t result);

	template<typename T>
	static void uploadDataTo(T*& bufferPointer, T* buffer, size_t size);

	template<typename T>
	static void downloadBuffer(T*& bufferPointer, T* buffer, size_t size, size_t offset = 0);

	template<typename T>
	std::unique_ptr<std::vector<T>> downloadBuffer(T*& bufferPointer, size_t size, size_t offset = 0);

	template<typename T>
	static void free(T* bufferPointer);
	template<typename T>
	static void freeAsync(T* bufferPointer, cudaStream_t stream);

	template<typename T>
	static void reset(T*& bufferPointer);
	template<typename T>
	static void freeHost(T*& bufferPointer);

	static glm::uint	getMaxThreadsBlock();
	static glm::ivec3	getMaxBlockDimensions();
	static glm::ivec3	getMaxGridSize();
	static glm::uint	getNumBlocks(glm::uint size, glm::uint blockThreads) { return (size + blockThreads) / blockThreads; }
	static size_t		getAllocatedMemory() { return _allocatedMemory; }
	static size_t		getMaxAllocatableMemory();

	template<typename T>
	static void initializeBuffer(T*& bufferPointer, size_t size, T* buffer = nullptr, bool trackMemory = true);
	template<typename T>
	static void initializeBufferAsync(T*& bufferPointer, size_t size, T* buffer = nullptr, cudaStream_t stream = nullptr, bool trackMemory = true);
	static void initializeBuffer(void*& bufferPointer, size_t size, bool trackMemory = true);
	template<typename T>
	static void initializeHostBuffer(T*& bufferPointer, size_t size, T* buffer = nullptr, bool trackMemory = true);
	template<typename T>
	static void initializeZeroBuffer(T*& bufferPointer, size_t size, bool trackMemory = true);
	template<typename T>
	static void initializeZeroBufferAsync(T*& bufferPointer, size_t size, cudaStream_t stream = nullptr, bool trackMemory = true);

	static void synchronize(const std::string& kernelName = "");

	static void  startTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent);
	static float stopTimer(const cudaEvent_t& startEvent, const cudaEvent_t& stopEvent);

	static void createStreams(std::initializer_list<cudaStream_t*> streams);
	static void waitFor(std::initializer_list<cudaStream_t*> streams);
	static void destroyStreams(std::initializer_list<cudaStream_t*> streams);
};

template <typename T>
void CudaHelper::uploadDataTo(T*& bufferPointer, T* buffer, size_t size)
{
	CudaHelper::checkError(cudaMemcpy(bufferPointer, buffer, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void CudaHelper::downloadBuffer(T*& bufferPointer, T* buffer, size_t size, size_t offset)
{
	CudaHelper::checkError(cudaMemcpy(buffer, bufferPointer + offset, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template <typename T>
std::unique_ptr<std::vector<T>> CudaHelper::downloadBuffer(T*& bufferPointer, size_t size, size_t offset)
{
	std::unique_ptr<std::vector<float>> buffer;
	buffer.reset(new std::vector<float>(size - offset));

	CudaHelper::checkError(cudaMemcpy(buffer->data(), bufferPointer + offset, sizeof(T) * size, cudaMemcpyDeviceToHost));

	return buffer;
}

template <typename T>
void CudaHelper::free(T* bufferPointer)
{
	if (!bufferPointer) return;
	cudaFree(bufferPointer);

	const auto it = _allocatedPointers.find(bufferPointer);
	if (it != _allocatedPointers.end())
	{
		_allocatedMemory -= it->second;
		_allocatedPointers.erase(it);
	}
}

template <typename T>
void CudaHelper::freeAsync(T* bufferPointer, cudaStream_t stream)
{
	if (!bufferPointer) return;
	cudaFreeAsync(bufferPointer, stream);

	const auto it = _allocatedPointers.find(bufferPointer);
	if (it != _allocatedPointers.end())
	{
		_allocatedMemory -= it->second;
		_allocatedPointers.erase(it);
	}
}

template <typename T>
void CudaHelper::reset(T*& bufferPointer)
{
	if (!bufferPointer) return;

	cudaFree(bufferPointer);
	const auto it = _allocatedPointers.find(bufferPointer);
	if (it != _allocatedPointers.end())
	{
		_allocatedMemory -= it->second;
		_allocatedPointers.erase(it);
	}

	bufferPointer = nullptr;
}

template <typename T>
void CudaHelper::freeHost(T*& bufferPointer)
{
	cudaFreeHost(bufferPointer);

	const auto it = _allocatedPointers.find(bufferPointer);
	if (it != _allocatedPointers.end())
	{
		_allocatedMemory -= it->second;
		_allocatedPointers.erase(it);
	}
}

template<typename T>
void CudaHelper::initializeBuffer(T*& bufferPointer, size_t size, T* buffer, bool trackMemory)
{
	CudaHelper::checkError(cudaMalloc((void**)(&bufferPointer), size * sizeof(T)));
	if (buffer)
		CudaHelper::checkError(cudaMemcpy(bufferPointer, buffer, size * sizeof(T), cudaMemcpyHostToDevice));

	if (trackMemory)
	{
		_allocatedMemory += size * sizeof(T);
		_allocatedPointers[bufferPointer] = size * sizeof(T);
	}
}

template <typename T>
void CudaHelper::initializeBufferAsync(T*& bufferPointer, size_t size, T* buffer, cudaStream_t stream, bool trackMemory)
{
	CudaHelper::checkError(cudaMallocAsync((void**)(&bufferPointer), size * sizeof(T), stream));
	if (buffer)
		CudaHelper::checkError(cudaMemcpyAsync(bufferPointer, buffer, size * sizeof(T), cudaMemcpyHostToDevice, stream));

	if (trackMemory)
	{
		_allocatedMemory += size * sizeof(T);
		_allocatedPointers[bufferPointer] = size * sizeof(T);
	}
}

template <typename T>
void CudaHelper::initializeHostBuffer(T*& bufferPointer, size_t size, T* buffer, bool trackMemory)
{
	CudaHelper::checkError(cudaHostAlloc((void**)(&bufferPointer), size * sizeof(T), cudaHostAllocDefault));
	if (buffer)
		CudaHelper::checkError(cudaMemcpy(bufferPointer, buffer, size * sizeof(T), cudaMemcpyHostToDevice));

	if (trackMemory)
	{
		_allocatedMemory += size * sizeof(T);
		_allocatedPointers[bufferPointer] = size * sizeof(T);
	}
}

template <typename T>
void CudaHelper::initializeZeroBuffer(T*& bufferPointer, size_t size, bool trackMemory)
{
	CudaHelper::checkError(cudaMalloc((void**)(&bufferPointer), size * sizeof(T)));
	CudaHelper::checkError(cudaMemset(bufferPointer, 0, size * sizeof(T)));

	if (trackMemory)
	{
		_allocatedMemory += size * sizeof(T);
		_allocatedPointers[bufferPointer] = size * sizeof(T);
	}
}

template <typename T>
void CudaHelper::initializeZeroBufferAsync(T*& bufferPointer, size_t size, cudaStream_t stream, bool trackMemory)
{
	CudaHelper::checkError(cudaMallocAsync((void**)(&bufferPointer), size * sizeof(T), stream));
	CudaHelper::checkError(cudaMemsetAsync(bufferPointer, 0, size * sizeof(T), stream));

	if (trackMemory)
	{
		_allocatedMemory += size * sizeof(T);
		_allocatedPointers[bufferPointer] = size * sizeof(T);
	}
}
