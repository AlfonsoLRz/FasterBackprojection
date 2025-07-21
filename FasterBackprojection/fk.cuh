#pragma once

#include "stdafx.h"

#include <cufft.h>

#include "math.cuh"

// 3D texture for interpolation (x, y, z order)
//texture<float, 3, cudaReadModeElementType> texInterpData;

inline __device__ glm::uint getKernelIdx(glm::uint x, glm::uint y, glm::uint t, const glm::uvec3& dataResolution)
{
    return t + dataResolution.z * (y + dataResolution.y * x);
}

inline __global__ void padIntensityFFT_FK(const float* H, cufftComplex* H_pad, glm::uvec3 currentResolution, glm::uvec3 newResolution)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= currentResolution.x || y >= currentResolution.y || t >= currentResolution.z)
        return;

    H_pad[getKernelIdx(x, y, t, newResolution)].x = H[getKernelIdx(x, y, t, currentResolution)] * static_cast<float>(t) / static_cast<float>(currentResolution.z);
}

inline __global__ void unpadIntensityFFT_FK(float* H, const cufftComplex* H_pad, glm::uvec3 currentResolution, glm::uvec3 newResolution)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
        return;

    H[getKernelIdx(x, y, t, currentResolution)] = H_pad[getKernelIdx(x, y, t, newResolution)].x;
}

//__global__ void stoltKernel(
//    const cufftComplex* __restrict__ H, cufftComplex* __restrict__ result, glm::uvec3 originalResolution, glm::uvec3 newResolution, float rangeWidth)
//{
//	const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
//    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
//		return;
//
//    glm::vec3 normCoordinates = (glm::vec3(x, y, t) - glm::vec3(originalResolution)) / glm::vec3(originalResolution);
//
//    // Stolt trick calculation
//    const float sqrt_term = sqrtf(rangeWidth * (normCoordinates.x * normCoordinates.x + normCoordinates.y * normCoordinates.y) + normCoordinates.z * normCoordinates.z);
//
//    // Map to [0, originalResolution-1] for interpolation
//    float fx = (normCoordinates.x + 1.0f) * 0.5f * static_cast<float>(originalResolution.x - 1);
//    float fy = (normCoordinates.y + 1.0f) * 0.5f * static_cast<float>(originalResolution.y - 1);
//    float fz = sqrt_term * static_cast<float>(originalResolution.z - 1); // sqrt_term ∈ [0, max_z], mapped to [0, M-1]
//
//    // Clamp coordinates to avoid out-of-bounds
//    fx = fmaxf(0.0f, fminf(fx, static_cast<float>(originalResolution.x - 1)));
//    fy = fmaxf(0.0f, fminf(fy, static_cast<float>(originalResolution.y - 1)));
//    fz = fmaxf(0.0f, fminf(fz, static_cast<float>(originalResolution.z - 1)));
//
//    // Trilinear interpolation (from originalResolution grid)
//    glm::uint x0 = static_cast<glm::uint>(floorf(fx)), x1 = x0 + 1;
//    glm::uint y0 = static_cast<glm::uint>(floorf(fy)), y1 = y0 + 1;
//    glm::uint z0 = static_cast<glm::uint>(floorf(fz)), z1 = z0 + 1;
//
//    // Clamp again (x1/y1/z1 might exceed bounds)
//    x1 = glm::min(x1, originalResolution.x - 1);
//    y1 = glm::min(y1, originalResolution.y - 1);
//    z1 = glm::min(z1, originalResolution.z - 1);
//
//    float dx = fx - static_cast<float>(x0);
//    float dy = fy - static_cast<float>(y0);
//    float dz = fz - static_cast<float>(z0);
//
//    cufftComplex c000 = H[getKernelIdx(x0, y0, z0, newResolution)];
//    cufftComplex c001 = H[getKernelIdx(x1, y0, z0, newResolution)];
//    cufftComplex c010 = H[getKernelIdx(x0, y1, z0, newResolution)];
//    cufftComplex c011 = H[getKernelIdx(x1, y1, z0, newResolution)];
//    cufftComplex c100 = H[getKernelIdx(x0, y0, z1, newResolution)];
//    cufftComplex c101 = H[getKernelIdx(x1, y0, z1, newResolution)];
//    cufftComplex c110 = H[getKernelIdx(x0, y1, z1, newResolution)];
//    cufftComplex c111 = H[getKernelIdx(x1, y1, z1, newResolution)];
//
//    cufftComplex c00 = complexLerp(c000, c001, dx);
//    cufftComplex c01 = complexLerp(c010, c011, dx);
//    cufftComplex c10 = complexLerp(c100, c101, dx);
//    cufftComplex c11 = complexLerp(c110, c111, dx);
//
//    cufftComplex c0 = complexLerp(c00, c01, dy);
//    cufftComplex c1 = complexLerp(c10, c11, dy);
//
//    // Write result
//    result[getKernelIdx(x, y, t, newResolution)] = complexLerp(c0, c1, dz);
//}

__global__ void stoltKernel(
    const cufftComplex* __restrict__ H,
    cufftComplex* __restrict__ result,
    glm::uvec3 originalResolution,
    glm::uvec3 fftResolution,
    float stoltConst,
    float maxSqrtTerm
)
{
    const glm::uint z = blockIdx.x * blockDim.x + threadIdx.x; 
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;  
    const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= fftResolution.x || y >= fftResolution.y || z >= fftResolution.z)
		return;

    if (z <= originalResolution.z)
    {
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

    int x0 = floorf(ix), y0 = floorf(iy), z0 = floorf(iz);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

    float dx = ix - x0;
    float dy = iy - y0;
    float dz = iz - z0;

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

    // Store result
    result[getKernelIdx(x, y, z, fftResolution)] = complexMulScalar(res, glm::abs(fz) / maxSqrtTerm);
}