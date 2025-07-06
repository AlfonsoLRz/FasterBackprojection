#pragma once

#include "stdafx.h"

#include "GpuStructs.cuh"
#include "math.cuh"
#include "transient_utils.cuh"

inline __global__ void normalizeReconstruction(float* __restrict__ v, glm::uint size, const float* maxValue, const float* minValue)
{
	const glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	v[idx] = (v[idx] - *minValue) * safeRCP(*maxValue - *minValue);
}

// Laplacian filter for 3D volume

inline __global__ void laplace(
		const float* __restrict__ volume, float* __restrict__ processed, glm::uvec3 resolution, float filterSize)
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	const int tz = blockIdx.z * blockDim.z + threadIdx.z;

	if (tx >= resolution.x || ty >= resolution.y || tz >= resolution.z)
		return;

	int halfFilterSize = static_cast<int>(floor(filterSize / 2));
	float laplacian = .0f;

	for (int x = -halfFilterSize; x <= halfFilterSize; ++x)
	{
		for (int y = -halfFilterSize; y <= halfFilterSize; ++y)
		{
			for (int z = -halfFilterSize; z <= halfFilterSize; ++z)
			{
				int nx = tx + x;
				int ny = ty + y;
				int nz = tz + z;
				if (nx < 0 || nx >= resolution.x || ny < 0 || ny >= resolution.y || nz < 0 || nz >= resolution.z)
					continue;

				laplacian += volume[nz * resolution.y * resolution.x + ny * resolution.x + nx];
			}
		}
	}

	processed[tz * resolution.y * resolution.x + ty * resolution.x + tx] = 
		filterSize * filterSize * filterSize * volume[tz * resolution.y * resolution.x + ty * resolution.x + tx] - laplacian;
}

// LoG

inline __global__ void initializeFourierVolume(float* __restrict__ input, cufftComplex* __restrict__ output, glm::uint size)
{
	glm::uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		output[tid].x = input[tid];
		output[tid].y = 0.0f;
	}
}

inline __global__ void multiplyKernel(cufftComplex* __restrict__ volume, const cufftComplex* __restrict__ kernel, glm::uint size)
{
	glm::uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size)
	{
		cufftComplex a = volume[tid];
		cufftComplex b = kernel[tid];
		volume[tid].x = a.x * b.x - a.y * b.y;
		volume[tid].y = a.x * b.y + a.y * b.x;
	}
}

inline __global__ void normalizeIFFT(float* __restrict__ data, glm::uint size)
{
	glm::uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) 
		data[tid] /= static_cast<float>(size);
}

inline __global__ void buildLoGKernel3D(cufftComplex* kernel, int nx, int ny, int nz, float sigma)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	// Note: For R2C output, x dimension is only 0...nx/2
	if (x > nx / 2 || y >= ny || z >= nz) return;

	// Index for packed format (nx/2+1)*ny*nz
	int idx = z * ny * (nx / 2 + 1) + y * (nx / 2 + 1) + x;

	// Normalized frequency coordinates
	float fx = (x <= nx / 2) ? static_cast<float>(x) : static_cast<float>(x - nx);
	float fy = (y <= ny / 2) ? static_cast<float>(y) : static_cast<float>(y - ny);
	float fz = (z <= nz / 2) ? static_cast<float>(z) : static_cast<float>(z - nz);

	// Scale to [-0.5, 0.5) 
	float fxn = fx / nx;
	float fyn = fy / ny;
	float fzn = fz / nz;

	// Frequency magnitude squared
	float k2 = fxn * fxn + fyn * fyn + fzn * fzn;

	// Laplacian of Gaussian in frequency space
	float lap = -4.0f * PI * PI * k2;
	float gauss = expf(-2.0f * PI * PI * sigma * sigma * k2);
	float val = lap * gauss;

	kernel[idx].x = val;
	kernel[idx].y = 0.0f;  // No imaginary part for real, symmetric kernel
}