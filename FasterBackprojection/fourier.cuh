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
    else
    {
		H[xy * paddedTimeDim + t] = cufftComplex{ H_orig[xy * timeDim + t - padding], .0f };
    }
}

inline __global__ void readBackFromIFFT(
    const cufftComplex* __restrict__ H, float* __restrict__ H_orig, 
	glm::uint sliceSize, glm::uint timeBins, glm::uint paddedTimeBins, glm::uint padding, glm::uint size)
{
    const glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    const glm::uint xy = idx / timeBins, t = idx % timeBins;
	H_orig[xy * timeBins + t] = H[xy * paddedTimeBins + t + padding].x;
}

inline __global__ void multiplyHK(cufftComplex *d_H, const cufftComplex *d_K, glm::uint batch, glm::uint numTimeBins, glm::uint size)
{
    const glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    const glm::uint t = idx % numTimeBins;
    cufftComplex h = d_H[idx];
    cufftComplex k = d_K[t];

    d_H[idx] = { h.x * k.x - h.y * k.y, h.x * k.y + h.y * k.x };
}

inline __global__ void normalizeH(cufftComplex* d_H, size_t batch, size_t timeSize)
{
    const glm::uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch * timeSize)
    {
        d_H[idx].x /= static_cast<float>(timeSize);
        d_H[idx].y /= static_cast<float>(timeSize);
    }
}

inline __global__ void shiftFFT(
    const cufftComplex* __restrict__ H, cufftComplex* __restrict__ H_aux, const glm::uvec3 resolution, const glm::uvec3 shift)
{
    const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const glm::uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution.x || y >= resolution.y || z >= resolution.z)
        return;

    const glm::uint shifted_x = (x + shift.x) % resolution.x;
    const glm::uint shifted_y = (y + shift.y) % resolution.y;
    const glm::uint shifted_z = (z + shift.z) % resolution.z;

    H_aux[shifted_x + resolution.x * (shifted_y + resolution.y * shifted_z)] = H[x + resolution.x * (y + resolution.y * z)];
}

inline __global__ void unshiftFFT(
    cufftComplex* __restrict__ H, const cufftComplex* __restrict__ H_aux, const glm::uvec3 resolution, const glm::uvec3 shift)
{
    const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const glm::uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution.x || y >= resolution.y || z >= resolution.z)
        return;

    const glm::uint unshifted_x = (x + shift.x) % resolution.x;
    const glm::uint unshifted_y = (y + shift.y) % resolution.y;
    const glm::uint unshifted_z = (z + shift.z) % resolution.z;

    H[unshifted_x + resolution.x * (unshifted_y + resolution.y * unshifted_z)] = H_aux[x + resolution.x * (y + resolution.y * z)];
}