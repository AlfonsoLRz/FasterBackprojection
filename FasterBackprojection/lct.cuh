#pragma once

#include "stdafx.h"
#include "math.cuh"

inline __forceinline__ __device__ glm::uint getKernelIdx(glm::uint x, glm::uint y, glm::uint t, const glm::uvec3& dataResolution)
{
	return x * dataResolution.y * dataResolution.z + y * dataResolution.z + t;
}

inline __global__ void computePSFKernel(float* __restrict__ psf, glm::uvec3 dataResolution, float slope)
{
    glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

    glm::vec3 grid = glm::vec3(
        -1.0f + (2.0f * static_cast<float>(x)) / (static_cast<float>(dataResolution.x) - 1),
        -1.0f + (2.0f * static_cast<float>(y)) / (static_cast<float>(dataResolution.y) - 1),
		(2.0f * static_cast<float>(t)) / (static_cast<float>(dataResolution.z) - 1)
	);

    psf[getKernelIdx(x, y, t, dataResolution)] = fabsf(((4 * slope) * (4 * slope)) * (grid.x * grid.x + grid.y * grid.y) - grid.z);
}

inline __global__ void findMinimumBinarize(float* __restrict__ psf, glm::uvec3 dataResolution)
{
	const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dataResolution.x || y >= dataResolution.y) return;

    // Find minimum along t-axis
    float minZ = FLT_MAX;
    for (glm::uint t = 0; t < dataResolution.z; t++) 
        minZ = glm::min(psf[getKernelIdx(x, y, t, dataResolution)], minZ);

    // Binarize
    for (glm::uint t = 0; t < dataResolution.z; t++)
        psf[getKernelIdx(x, y, t, dataResolution)] = static_cast<float>(glm::epsilonEqual(psf[getKernelIdx(x, y, t, dataResolution)], minZ, glm::epsilon<float>()));
}

inline __global__ void normalizePSF(float* __restrict__ psf, const float* __restrict__ centerSum, glm::uvec3 dataResolution)
{
    glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

	const glm::uint kernelIdx = getKernelIdx(x, y, t, dataResolution);
    psf[kernelIdx] *= safeRCP(*centerSum);
	//psf[kernelIdx] *= psf[kernelIdx];       // Prepare for L2 normalization
}

inline __global__ void l2NormPSF(float* __restrict__ psf, const float* __restrict__ sqrtNorm, glm::uvec3 dataResolution)
{
    glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

    const glm::uint kernelIdx = getKernelIdx(x, y, t, dataResolution);
    //psf[kernelIdx] *= safeRCP(psf[kernelIdx]);
    psf[kernelIdx] *= safeRCP(sqrtf(*sqrtNorm));
}

inline __global__ void rollPSF(const float* __restrict__ psf, cufftComplex* __restrict__ rolledPsf, glm::uvec3 originalDataResolution, glm::uvec3 dataResolution)
{
    glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

    // Roll operation (only in x and y dimensions)
    glm::uint new_x = (x + originalDataResolution.x) % dataResolution.x;
    glm::uint new_y = (y + originalDataResolution.y) % dataResolution.y;

    rolledPsf[getKernelIdx(new_x, new_y, t, dataResolution)].x = psf[getKernelIdx(x, y, t, dataResolution)];
}

inline __global__ void multiplyPSF(cufftComplex* __restrict__ d_H, const cufftComplex* __restrict__ d_K, glm::uint size)
{
    const glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    cufftComplex h = d_H[idx], k = d_K[idx];
    d_H[idx] = { h.x * k.x - h.y * k.y, h.x * k.y + h.y * k.x };
}

inline __global__ void padIntensityFFT(const float* __restrict__ H, cufftComplex* __restrict__ H_pad, glm::uvec3 currentResolution, glm::uvec3 newResolution)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x;
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
        return;

    H_pad[x * newResolution.y * newResolution.z + y * newResolution.z + t].x = H[x * currentResolution.y * currentResolution.z + y * currentResolution.z + t];
}

inline __global__ void padIntensityFFT_unrolled(
    const float* __restrict__ H, cufftComplex* __restrict__ H_pad,
    glm::uvec3 currentResolution, glm::uvec3 newResolution, glm::uint stride)
{
    const glm::uint t = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
        return;

	#pragma unroll
    for (glm::uint i = t; i < t + stride && i < newResolution.z; ++i)
    	H_pad[getKernelIdx(x, y, i, newResolution)].x = H[getKernelIdx(x, y, i, currentResolution)];
}

inline __global__ void unpadIntensityFFT(float* __restrict__ H, const cufftComplex* __restrict__ H_pad, glm::uvec3 currentResolution, glm::uvec3 newResolution)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x;
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
        return;

    H[x * currentResolution.y * currentResolution.z + y * currentResolution.z + t] = H_pad[x * newResolution.y * newResolution.z + y * newResolution.z + t].x;
}

inline __global__ void unpadIntensityFFT_unrolled(
    float* __restrict__ H, const cufftComplex* __restrict__ H_pad,
    glm::uvec3 currentResolution, glm::uvec3 newResolution, glm::uint stride)
{
    const glm::uint t = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
        return;

	#pragma unroll
    for (glm::uint i = t; i < t + stride && i < currentResolution.z; ++i)
        H[getKernelIdx(x, y, i, currentResolution)] = H_pad[getKernelIdx(x, y, i, newResolution)].x;
}

inline __global__ void wienerFilterPsf(cufftComplex* __restrict__ psf, glm::uvec3 dataResolution, float snr)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) 
        return;

	const glm::uint kernelIdx = getKernelIdx(x, y, t, dataResolution);
	cufftComplex value = psf[kernelIdx], conjugate = { value.x, -value.y };
    float wienerFactor = safeRCP(value.x * value.x + value.y * value.y + 1.0f / snr);
    psf[kernelIdx].x = wienerFactor * conjugate.x;
    psf[kernelIdx].y = wienerFactor * conjugate.y;
}

template <bool diffuse>
inline __global__ void scaleIntensity(float* __restrict__ intensity, glm::uvec3 dataResolution, glm::uint numElements, float divisor)
{
	const glm::uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numElements)
		return;

	glm::uint t = tid % dataResolution.z;
    float weight = static_cast<float>(t) * divisor;
    weight = diffuse ? weight * weight * weight * weight : weight * weight;

    intensity[tid] *= static_cast<float>(t) * weight;
}

inline __global__ void multiplyTransformTranspose(
    const float* __restrict__ volumeGpu, const float* __restrict__ mtx, float* __restrict__ mult, glm::uvec3 dataResolution)
{
	const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < dataResolution.x && y < dataResolution.y && t < dataResolution.z) 
    {
        const glm::uint spatialIndex = y * dataResolution.y + x;  

        float sum = 0.0f;
        for (glm::uint k = 0; k < dataResolution.z; ++k)
            sum += mtx[k * dataResolution.z + t] * volumeGpu[spatialIndex * dataResolution.z + k];

        mult[(y * dataResolution.y + x) * dataResolution.z + t] = sum;
    }
}