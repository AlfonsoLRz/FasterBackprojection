#pragma once

#include "stdafx.h"

#include <cufft.h>

#include "math.cuh"

// 3D texture for interpolation (x, y, z order)
//texture<float, 3, cudaReadModeElementType> texInterpData;

inline __forceinline__ __device__ glm::uint getKernelIdx(glm::uint x, glm::uint y, glm::uint t, const glm::uvec3& dataResolution)
{
    return t + dataResolution.z * (y + dataResolution.y * x);
}

inline __global__ void padIntensityFFT_FK(
    const float* __restrict__ H, cufftComplex* __restrict__ H_pad, glm::uvec3 currentResolution, glm::uvec3 newResolution, float divisor)
{
    const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= currentResolution.x || y >= currentResolution.y || t >= currentResolution.z / 1)
        return;

	H_pad[getKernelIdx(x, y, t, newResolution)] = make_cuComplex(H[getKernelIdx(x, y, t, currentResolution)] * static_cast<float>(t) * divisor, .0f);
}

inline __global__ void unpadIntensityFFT_FK(float* H, const cufftComplex* H_pad, glm::uvec3 currentResolution, glm::uvec3 newResolution)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
        return;

    H[getKernelIdx(x, y, t, currentResolution)] = H_pad[getKernelIdx(x, y, t, newResolution)].x;
}

inline __device__ glm::uvec3 getCyclicShiftedIndices(glm::uint& x, glm::uint& y, glm::uint& z, glm::uvec3 shift, glm::uvec3 resolution)
{
    x = (x + shift.x) % resolution.x;
    y = (y + shift.y) % resolution.y;
    z = (z + shift.z) % resolution.z;
}

__global__ void stoltKernel(
    const cufftComplex* __restrict__ H,
    cufftComplex* __restrict__ result,
    glm::uvec3 originalResolution,
    glm::uvec3 fftResolution,
    glm::uvec3 shift,
    float stoltConst,
    float maxSqrtTerm
)
{
    glm::uint z = blockIdx.x * blockDim.x + threadIdx.x; 
    glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;  
    glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= fftResolution.x || y >= fftResolution.y || z >= fftResolution.z)
		return;

    if (z <= originalResolution.z)
    {
        getCyclicShiftedIndices(x, y, z, shift, fftResolution);
        result[getKernelIdx(x, y, z, fftResolution)] = make_cuComplex(0.0f, 0.0f);
        return;
	}

    // Normalize to [-1, 1]
    float fx = 2.0f * (static_cast<float>(x) / static_cast<float>(fftResolution.x)) - 1.0f;
    float fy = 2.0f * (static_cast<float>(y) / static_cast<float>(fftResolution.y)) - 1.0f;
    float fz = 2.0f * (static_cast<float>(z) / static_cast<float>(fftResolution.z)) - 1.0f;

    // Compute Stolt sqrt term
    float sqrt_term = sqrtf(stoltConst * (fx * fx + fy * fy) + fz * fz);

    // Map (sqrt_term, fy, fx) from [-1, 1] → [0, dims)
    float ix = (fx + 1.0f) * 0.5f * static_cast<float>(fftResolution.x);
    float iy = (fy + 1.0f) * 0.5f * static_cast<float>(fftResolution.y);
    float iz = (sqrt_term + 1.0f) * 0.5f * static_cast<float>(fftResolution.z);

	ix = fmod(ix + static_cast<float>(shift.x), static_cast<float>(fftResolution.x));
	iy = fmod(iy + static_cast<float>(shift.y), static_cast<float>(fftResolution.y));
	iz = fmod(iz + static_cast<float>(shift.z), static_cast<float>(fftResolution.z));

    int x0 = static_cast<int>(floorf(ix)), y0 = static_cast<int>(floorf(iy)), z0 = static_cast<int>(floorf(iz));
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

    float dx = ix - static_cast<float>(x0);
    float dy = iy - static_cast<float>(y0);
    float dz = iz - static_cast<float>(z0);

    // Clamp all indices
    x0 = glm::max(0, glm::min(x0, static_cast<int>(fftResolution.x) - 1));
    x1 = glm::max(0, glm::min(x1, static_cast<int>(fftResolution.x) - 1));
    y0 = glm::max(0, glm::min(y0, static_cast<int>(fftResolution.y) - 1));
    y1 = glm::max(0, glm::min(y1, static_cast<int>(fftResolution.y) - 1));
    z0 = glm::max(0, glm::min(z0, static_cast<int>(fftResolution.z) - 1));
    z1 = glm::max(0, glm::min(z1, static_cast<int>(fftResolution.z) - 1));

    // Fetch values
    cufftComplex c000 = H[getKernelIdx(x0, y0, z0, fftResolution)];
    cufftComplex c001 = H[getKernelIdx(x1, y0, z0, fftResolution)];
    cufftComplex c010 = H[getKernelIdx(x0, y1, z0, fftResolution)];
    cufftComplex c011 = H[getKernelIdx(x1, y1, z0, fftResolution)];
    cufftComplex c100 = H[getKernelIdx(x0, y0, z1, fftResolution)];
    cufftComplex c101 = H[getKernelIdx(x1, y0, z1, fftResolution)];
    cufftComplex c110 = H[getKernelIdx(x0, y1, z1, fftResolution)];
    cufftComplex c111 = H[getKernelIdx(x1, y1, z1, fftResolution)];

    // Trilinear interpolation
    cufftComplex c00 = complexLerp(c000, c001, dx);
    cufftComplex c01 = complexLerp(c010, c011, dx);
    cufftComplex c10 = complexLerp(c100, c101, dx);
    cufftComplex c11 = complexLerp(c110, c111, dx);

    cufftComplex c0 = complexLerp(c00, c01, dy);
    cufftComplex c1 = complexLerp(c10, c11, dy);

    cufftComplex res = complexLerp(c0, c1, dz);

    getCyclicShiftedIndices(x, y, z, shift, fftResolution);
    result[getKernelIdx(x, y, z, fftResolution)] = complexMulScalar(res, glm::abs(fz) * maxSqrtTerm);
}