#pragma once

#include "stdafx.h"

#include <cufft.h>

// Error checking macro 
#define CUFFT_CHECK(err) \
    do { \
        cufftResult err_ = (err); \
        if (err_ != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT error " << err_ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

enum class PadMode: uint8_t
{
	Zero = 0,
    Edge = 1
};

inline __global__ void padBuffer(
    glm::uint sliceSize, glm::uint timeDim, glm::uint padding, glm::uint paddedTimeDim,
    const float* __restrict__ H_orig, cufftComplex* __restrict__ H, PadMode padMode)
{
	const glm::uint xy = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint t = blockIdx.y * blockDim.y + threadIdx.y;
    if (xy >= sliceSize || t >= paddedTimeDim)
		return;

    if (t < padding)
    {
        if (padMode == PadMode::Edge)
        {
            H[xy * paddedTimeDim + t] = cufftComplex{ H_orig[xy * timeDim + 0], .0f };
		}
    }
    else if (t > timeDim + padding)
    {
        if (padMode == PadMode::Edge)
        {
            H[xy * paddedTimeDim + t] = cufftComplex{ H_orig[xy * timeDim + timeDim - 1], .0f };
        }
    }
}

inline __global__ void readBackFromIFFT(
    const cufftComplex* __restrict__ H, float* __restrict__ H_orig, 
	glm::uint sliceSize, glm::uint timeBins, glm::uint paddedTimeBins, glm::uint padding)
{
	const glm::uint xy = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint t = blockIdx.y * blockDim.y + threadIdx.y;

    if (xy >= sliceSize || t >= timeBins)
		return;

	H_orig[xy * timeBins + t] = H[xy * paddedTimeBins + t + padding].x;
}

inline __global__ void multiplyHK(cufftComplex *d_H, const cufftComplex *d_K, size_t batch, size_t numTimeBins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * numTimeBins) 
        return;

    int t = idx % numTimeBins;
    cufftComplex h = d_H[idx];
    cufftComplex k = d_K[t];

    float real = (h.x * k.x - h.y * k.y);
    float imag = (h.x * k.y + h.y * k.x);

    d_H[idx] = { real, imag };
}

inline __global__ void normalizeH(cufftComplex* d_H, size_t batch, size_t timeSize)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch * timeSize)
    {
        d_H[idx].x /= static_cast<float>(timeSize);
        d_H[idx].y /= static_cast<float>(timeSize);
    }
}