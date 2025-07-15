#pragma once

#include "stdafx.h"
#include "math.cuh"

#define CHECK_CUSPARSE(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSPARSE error at %s:%d - %s\n", __FILE__, __LINE__, cusparseGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

inline __forceinline__ __device__ glm::uint getKernelIdx(glm::uint x, glm::uint y, glm::uint t, const glm::uvec3& dataResolution)
{
	return x * dataResolution.y * dataResolution.z + y * dataResolution.z + t;
}

inline __global__ void computePSFKernel(float* __restrict__ psf, glm::uvec3 dataResolution, float slope)
{
    glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

    glm::vec3 grid = glm::vec3(
        -1.0f + (2.0f * static_cast<float>(x)) / (static_cast<float>(dataResolution.x) - 1),
        -1.0f + (2.0f * static_cast<float>(y)) / (static_cast<float>(dataResolution.y) - 1),
        0.0f + (2.0f * static_cast<float>(t)) / (static_cast<float>(dataResolution.z) - 1)
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
        psf[getKernelIdx(x, y, t, dataResolution)] = psf[getKernelIdx(x, y, t, dataResolution)] == minZ ? 1.0f : 0.0f;
}

inline __global__ void normalizePSF(float* __restrict__ psf, const float* __restrict__ centerSum, glm::uvec3 dataResolution)
{
    glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

	const glm::uint kernelIdx = getKernelIdx(x, y, t, dataResolution);
    psf[kernelIdx] *= safeRCP(*centerSum);
	//psf[kernelIdx] *= psf[kernelIdx];       // Prepare for L2 normalization
}

inline __global__ void l2NormPSF(float* __restrict__ psf, float sqrtNorm, glm::uvec3 dataResolution)
{
    glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

    const glm::uint kernelIdx = getKernelIdx(x, y, t, dataResolution);
    //psf[kernelIdx] *= safeRCP(psf[kernelIdx]);
    psf[kernelIdx] *= safeRCP(sqrtNorm);
}

inline __global__ void rollPSF(const float* __restrict__ psf, cufftComplex* __restrict__ rolledPsf, glm::uvec3 originalDataResolution, glm::uvec3 dataResolution)
{
    glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

    // Roll operation (only in x and y dimensions)
    glm::uint new_x = (x + originalDataResolution.x) % dataResolution.x;
    glm::uint new_y = (y + originalDataResolution.y) % dataResolution.y;

    rolledPsf[getKernelIdx(new_x, new_y, t, dataResolution)].x = psf[getKernelIdx(x, y, t, dataResolution)];
}

inline __global__ void padIntensityFFT(
    const float* __restrict__ H, cufftComplex* __restrict__ H_pad, glm::uint sliceSize, glm::uint nt, glm::uint nt_pad)
{
	const glm::uint xy = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y * blockDim.y + threadIdx.y;
	if (xy >= sliceSize || t >= nt) 
        return;

	H_pad[xy * nt_pad + t].x = H[xy * nt + t];
}

inline __global__ void wienerFilterPsf(cufftComplex* __restrict__ psf, glm::uvec3 dataResolution, float snr)
{
    const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) 
        return;

	const glm::uint kernelIdx = getKernelIdx(x, y, t, dataResolution);
	cufftComplex value = psf[kernelIdx], conjugate = { value.x, -value.y };
    float norm = value.x * value.x + value.y * value.y;

    if (norm > 0.0f)
    {
        float wienerFactor = safeRCP(norm + snr);
        psf[kernelIdx].x = wienerFactor * value.x;
        psf[kernelIdx].y = wienerFactor * value.y;
    }
    else
    {
        psf[kernelIdx] = { 0.0f, 0.0f }; // Avoid division by zero
	}
}

inline __global__ void scaleIntensity(float* __restrict__ intensity, const glm::uvec3 dataResolution, bool diffuse)
{
	const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) 
		return;

	float tempWeight = static_cast<float>(t) / (static_cast<float>(dataResolution.z) - 1.0f);
    if (diffuse)
		tempWeight = tempWeight * tempWeight * tempWeight * tempWeight;
    else
		tempWeight = tempWeight * tempWeight;

    intensity[getKernelIdx(x, y, t, dataResolution)] *= tempWeight;
}

inline __global__ void multiplyTransformTranspose(float* __restrict__ volumeGpu, float* __restrict__ mtx, float* __restrict__ mult, glm::uint XY, glm::uint M)
{
	const glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x, idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < XY && idy < M) {
        float sum = 0.0f;
        for (int k = 0; k < M; ++k) 
            sum += mtx[idy * M + k] * volumeGpu[idx * M + k];
        
        mult[idx * M + idy] = sum;
    }
}

// Resampling

inline __global__ void buildResamplingOperator(float* mtx, float* inverseMtx, glm::uint M)
{
    glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * M) return;

    // Build forward operator
    glm::uint col = static_cast<glm::uint>(ceilf(sqrtf(static_cast<float>(idx) + 1))) - 1;
    float val = 1.0f / sqrtf(static_cast<float>(idx) + 1);
    mtx[idx * M + col] = val;

    // Build inverse operator (transposed)
    if (idx < M) 
        for (glm::uint i = idx * idx; i < (idx + 1) * (idx + 1); ++i)
            if (i < M * M) 
                inverseMtx[idx * M * M + i] = val;
}

inline __global__ void downsampleOperator(float* mtx, float* inverseMtx, glm::uint M, glm::uint K)
{
    glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (glm::uint k = 0; k < K; k++)
    {
        glm::uint new_M = M >> (k + 1);
        if (idx < new_M) 
        {
            // forward operator
            for (glm::uint j = 0; j < M; j++)
            {
                float sum = 0.0f;
                for (glm::uint i = 2 * idx; i < 2 * (idx + 1); i++)
                    sum += mtx[i * M + j];

                mtx[idx * new_M + j] = 0.5f * sum;
            }

            // inverse operator
            for (glm::uint j = 0; j < new_M; j++)
            {
                float sum = 0.0f;
                for (glm::uint i = 2 * j; i < 2 * (j + 1); i++)
                    sum += inverseMtx[idx * M + i];

                inverseMtx[idx * new_M + j] = 0.5f * sum;
            }
        }
        __syncthreads();
    }
}