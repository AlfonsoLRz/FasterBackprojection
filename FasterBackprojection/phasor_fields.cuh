#pragma once

#include "stdafx.h"
#include "math.cuh"

// 
inline __forceinline__ __device__ glm::uint getKernelIdx(glm::uint x, glm::uint y, glm::uint t, const glm::uvec3& dataResolution)
{
    return x * dataResolution.y * dataResolution.z + y * dataResolution.z + t;
}

// PSF

inline __global__ void computePSFKernel_pf(float* __restrict__ psf, glm::uvec3 dataResolution, float slope)
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

inline __global__ void findMinimumBinarize_pf(float* __restrict__ psf, glm::uvec3 dataResolution)
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

inline __global__ void normalizePSF_pf(float* __restrict__ psf, const float* __restrict__ centerSum, glm::uvec3 dataResolution)
{
    glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

    const glm::uint kernelIdx = getKernelIdx(x, y, t, dataResolution);
    psf[kernelIdx] *= safeRCP(*centerSum);
    //psf[kernelIdx] *= psf[kernelIdx];       // Prepare for L2 normalization
}

inline __global__ void l2NormPSF_pf(float* __restrict__ psf, const float* __restrict__ sqrtNorm, glm::uvec3 dataResolution)
{
    glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

    const glm::uint kernelIdx = getKernelIdx(x, y, t, dataResolution);
    //psf[kernelIdx] *= safeRCP(psf[kernelIdx]);
    psf[kernelIdx] *= safeRCP(sqrtf(*sqrtNorm));
}

inline __global__ void rollPSF_pf(const float* __restrict__ psf, cufftComplex* __restrict__ rolledPsf, glm::uvec3 originalDataResolution, glm::uvec3 dataResolution)
{
    glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) return;

    // Roll operation (only in x and y dimensions)
    glm::uint new_x = (x + originalDataResolution.x) % dataResolution.x;
    glm::uint new_y = (y + originalDataResolution.y) % dataResolution.y;

    rolledPsf[getKernelIdx(new_x, new_y, t, dataResolution)].x = psf[getKernelIdx(x, y, t, dataResolution)];
}

inline __global__ void extractConjugate_pf(cufftComplex* __restrict__ psf, glm::uvec3 dataResolution)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x;
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z) 
		return;

    const glm::uint kernelIdx = getKernelIdx(x, y, t, dataResolution);
    const cufftComplex value = psf[kernelIdx], conjugate = { value.x, -value.y };
    psf[kernelIdx] = conjugate;
}

inline __global__ void multiplyPSF_pf(cufftComplex* __restrict__ d_H, const cufftComplex* __restrict__ d_K, glm::uint size)
{
    const glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    const cufftComplex h = d_H[idx], k = d_K[idx];
    d_H[idx] = { h.x * k.x - h.y * k.y, h.x * k.y + h.y * k.x };
}

inline __global__ void createVirtualWaves(
    float* __restrict__ waveCos, float* __restrict__ waveSin, 
    glm::uint numSamples, float numCycles, float alpha
)
{
    const glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) 
    {
        float stdev = static_cast<float>(numSamples - 1) / (2.0f * alpha);
        float n = -(static_cast<float>(numSamples) - 1) / 2.0f + static_cast<float>(idx);
		float window = expf(-0.5f * (n / stdev) * (n / stdev));
        float angle = 2.0f * PI * numCycles * static_cast<float>(idx + 1) / static_cast<float>(numSamples);

        waveSin[idx] = sinf(angle) * window;
		waveCos[idx] = cosf(angle) * window;
    }
}

__global__ void convolve(
    float* u, const float* v, float* result, glm::uint bufferSize, glm::uint kernelSize
)
{
    extern __shared__ float sharedMemory[];
    float* kernelShared = sharedMemory;

    const glm::uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bufferSize)
        return;

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < kernelSize; i++)
            kernelShared[i] = v[kernelSize - 1 - i];
    }
    __syncthreads();

    float sum = .0f;
    int centerOffset = (kernelSize % 2 == 0) ? (kernelSize / 2 - 1) : (kernelSize / 2);

    #pragma unroll
    for (glm::uint j = 0; j < kernelSize; j++)
    {
        int convT = static_cast<int>(tid) - centerOffset + static_cast<int>(j);
        if (convT >= 0 && convT < bufferSize)
            sum += u[convT] * kernelShared[j];
    }

    result[tid] = sum;
}

__global__ void convolveVirtualWaves(
	const float* data, glm::uvec3 dataResolution,
    const float* virtualWaveSin, 
    const float* virtualWaveCos, 
    float* waveCos, float* waveSin, 
    glm::uint numSamples
)
{
    extern __shared__ float sharedMemory[];
    float* virtualSinShared = sharedMemory;
    float* virtualCosShared = sharedMemory + numSamples;

	const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;

    if (t >= dataResolution.z || y >= dataResolution.y || x >= dataResolution.x) 
		return;

    // Load virtual waves into shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) 
    {
        for (int i = 0; i < numSamples; i++) 
        {
            virtualSinShared[i] = virtualWaveSin[numSamples - 1 - i];
            virtualCosShared[i] = virtualWaveCos[numSamples - 1 - i];
        }
    }
    __syncthreads();

    size_t dataIdx = getKernelIdx(x, y, t, dataResolution);
    float tmpReal = 0.0f;
    float tmpImg = 0.0f;
    int centerOffset = (numSamples % 2 == 0) ? (numSamples / 2 - 1) : (numSamples / 2);

	#pragma unroll
    for (glm::uint j = 0; j < numSamples; j++)
    {
        int convT = static_cast<int>(t) - centerOffset + static_cast<int>(j);
        if (convT >= 0 && convT < static_cast<int>(dataResolution.z))
        {
            size_t convIdx = getKernelIdx(x, y, convT, dataResolution);
            tmpReal += data[convIdx] * virtualSinShared[j];
            tmpImg += data[convIdx] * virtualCosShared[j];
        }
    }

    waveCos[dataIdx] = tmpReal;
    waveSin[dataIdx] = tmpImg;
}

inline __global__ void multiplyTransformTranspose(
    const float* __restrict__ phasorCos, const float* __restrict__ phasorSin,
    const float* __restrict__ mtx, 
    cufftComplex* __restrict__ phasorDataCos, cufftComplex* __restrict__ phasorDataSin,
    glm::uvec3 dataResolution, glm::uvec3 newResolution)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x;
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < dataResolution.x && y < dataResolution.y && t < dataResolution.z)
    {
        float sumCos = .0f, sumSin = .0f;

		#pragma unroll
        for (glm::uint k = 0; k < dataResolution.z; ++k)
        {
            const glm::uint idx = getKernelIdx(x, y, k, dataResolution);
            sumCos += mtx[k * dataResolution.z + t] * phasorCos[idx];
            sumSin += mtx[k * dataResolution.z + t] * phasorSin[idx];
        }

        phasorDataCos[getKernelIdx(x, y, t, newResolution)].x = sumCos;
        phasorDataSin[getKernelIdx(x, y, t, newResolution)].x = sumSin;
    }
}

inline __global__ void convolveBackprojectionKernel(
    cufftComplex* __restrict__ phasorDataCos, cufftComplex* __restrict__ phasorDataSin,
    const cufftComplex* __restrict__ psf,
    glm::uvec3 dataResolution
)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x;
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= dataResolution.x || y >= dataResolution.y || t >= dataResolution.z)
        return;

    const glm::uint kernelIdx = getKernelIdx(x, y, t, dataResolution);
    cufftComplex psfValue = psf[kernelIdx];

    phasorDataCos[kernelIdx] = complexMul(phasorDataCos[kernelIdx], psfValue);
    phasorDataSin[kernelIdx] = complexMul(phasorDataSin[kernelIdx], psfValue);
}

inline __global__ void computePhasorFieldMagnitude(
    const cufftComplex* __restrict__ phasorDataCos, const cufftComplex* __restrict__ phasorDataSin,
    float* __restrict__ result,
    glm::uvec3 resolution, glm::uvec3 newResolution
)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x;
    const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < newResolution.x && y < newResolution.y && t < newResolution.z)
    {
        const glm::uint prevIdx = getKernelIdx(x, y, t, resolution);
        float phasorCos = phasorDataCos[prevIdx].x, phasorSin = phasorDataSin[prevIdx].x;
        result[getKernelIdx(x, y, t, newResolution)] = sqrtf(phasorCos * phasorCos + phasorSin * phasorSin);
    }
}

inline __global__ void multiplyTransformTransposeInv(
    const float* __restrict__ vol,
    const float* __restrict__ mtx,
    float* __restrict__ result,
    glm::uvec3 dataResolution)
{
    const glm::uint t = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < dataResolution.x && y < dataResolution.y && t < dataResolution.z)
    {
        float sum = .0f;
		#pragma unroll
        for (glm::uint k = 0; k < dataResolution.z; ++k)
            sum += mtx[t * dataResolution.z + k] * vol[getKernelIdx(x, y, k, dataResolution)];

        result[getKernelIdx(x, y, t, dataResolution)] = fmax(sum, .0f);
    }
}
