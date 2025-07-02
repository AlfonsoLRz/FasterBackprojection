#pragma once

#include "Singleton.h"

class CudaContext : public Singleton<CudaContext>
{
	friend class Singleton<CudaContext>;

private:
	CUcontext _cudaContext = nullptr;

private:
	CudaContext();
	CUdevice setDevice(uint8_t deviceIndex = UINT8_MAX);

public:
	~CudaContext();
	CUcontext getContext() const { return _cudaContext; }
};

