#pragma once

#include "stdafx.h"

#include <cufft.h>

// 3D texture for interpolation (x, y, z order)
//texture<float, 3, cudaReadModeElementType> texInterpData;

inline __device__ glm::uint getKernelIdx(glm::uint x, glm::uint y, glm::uint t, const glm::uvec3& dataResolution)
{
    return t + dataResolution.z * (y + dataResolution.y * x);
}

inline __global__ void padIntensityFFT(const float* H, cufftComplex* H_pad, glm::uvec3 currentResolution, glm::uvec3 newResolution)
{
    const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
        return;

    H_pad[getKernelIdx(x, y, t, newResolution)].x = H[getKernelIdx(x, y, t, currentResolution)];
}

inline __global__ void unpadIntensityFFT(float* H, const cufftComplex* H_pad, glm::uvec3 currentResolution, glm::uvec3 newResolution)
{
    const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
        return;

    H[getKernelIdx(x, y, t, currentResolution)] = H_pad[getKernelIdx(x, y, t, newResolution)].x;
}

__global__ void stoltKernel(cufftComplex* d_H, glm::vec3 originalResolution, glm::uvec3 newResolution, float rangeWidth)
{
	const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
		return;

    const glm::uint idx = x * newResolution.y * newResolution.z + y * newResolution.z + t;
    glm::vec3 normCoordinates = (glm::vec3(x, y, t) - originalResolution) / originalResolution;

    // Stolt trick calculation
    const float sqrt_term = sqrtf(rangeWidth * (normCoordinates.x * normCoordinates.x + normCoordinates.y * normCoordinates.y) + normCoordinates.z * normCoordinates.z);

	//if (normCoordinates.z > 0.0f) 
 //   {
 //       normCoordinates = (normCoordinates + 1.0f) / 2.0f;
 //       float interpolatedVal = tex3D(interpolateTexture, normCoordinates.x, normCoordinates.y, sqrt_term);
 //       d_H[idx] = interpolatedVal * fabsf(normCoordinates.z) / sqrt_term;
 //   }
 //   else 
 //       d_H[idx] = 0.0f;
}