#pragma once

#include "stdafx.h"
#include <cufft.h>

#include "CudaHelper.h"

// Error checking macro for cuFFT
#define CUFFT_CHECK(err) \
    do { \
        cufftResult err_ = (err); \
        if (err_ != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT error " << err_ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)


// Kernel for padding real input array
__global__ void padKernel(
    const float* input, float* output,
    glm::uint inputTimeBins, glm::uint paddedTimeBins, glm::uint otherDimsProduct, glm::uint padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= paddedTimeBins * otherDimsProduct) return;

    int time_pos = idx % paddedTimeBins;
    int other_pos = idx / paddedTimeBins;

    if (time_pos < padding || time_pos >= (padding + inputTimeBins)) {
        // Padding area - set to zero
        output[idx] = 0.0f;
    }
    else {
        // Copy from input
        int input_idx = (time_pos - padding) + inputTimeBins * other_pos;
        output[idx] = input[input_idx];
    }
}

// Kernel for element-wise multiplication (complex)
__global__ void multiplyKernel(cufftComplex* H_fft, const cufftComplex* K_fft, int time_dim, int other_dims_product, const int* K_shape)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= time_dim * other_dims_product) return;

    int time_pos = idx % time_dim;
    int other_pos = idx / time_dim;

    // Apply K_fft (assuming K_shape matches the required dimensions)
    H_fft[idx] = cuCmulf(H_fft[idx], K_fft[time_pos % K_shape[0]]);
}

// Kernel for removing padding and handling borders (real output)
__global__ void removePadding(
    const float* input, float* output,
    glm::uint inputTimeBins, glm::uint paddedTimeBins, glm::uint otherDimsProduct, glm::uint padding,
    int border_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= paddedTimeBins * otherDimsProduct) return;

    int time_pos = idx % paddedTimeBins;
    int other_pos = idx / paddedTimeBins;

    int input_idx = (time_pos + padding) + inputTimeBins * other_pos;

    if (border_type == 1) { // 'erase'
        if (time_pos < padding / 2 || time_pos >= (paddedTimeBins - padding / 2)) 
            output[idx] = 0.0f;
        else 
            output[idx] = input[input_idx];
    }
    else 
        output[idx] = input[input_idx];
}

// Helper kernel for scaling real array after inverse FFT
__global__ void scaleKernelReal(float* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    data[idx] *= scale;
}

inline void filter_H_real_cuda(const float* d_H, float* d_HoK,
    const cufftComplex* d_K_fft, const int* d_K_shape,
    int nt, int nt_pad, int padding,
    int otherDimsProduct, int border_type) {
    // Calculate dimensions
    int inputTimeBins = nt;
    int paddedTimeBins = nt_pad;
    int outputTimeBins = nt_pad - 2 * padding;

    // Allocate device memory
    float* d_H_pad_real;
    cufftComplex* d_H_fft;
    float* d_H_ifft_real;

    CudaHelper::checkError(cudaMalloc(&d_H_pad_real, paddedTimeBins * otherDimsProduct * sizeof(float)));
    CudaHelper::checkError(cudaMalloc(&d_H_fft, (paddedTimeBins / 2 + 1) * otherDimsProduct * sizeof(cufftComplex)));
    CudaHelper::checkError(cudaMalloc(&d_H_ifft_real, paddedTimeBins * otherDimsProduct * sizeof(float)));

    // Create cuFFT plans
    cufftHandle plan_forward, plan_inverse;
    CUFFT_CHECK(cufftPlan1d(&plan_forward, paddedTimeBins, CUFFT_R2C, otherDimsProduct));
    CUFFT_CHECK(cufftPlan1d(&plan_inverse, paddedTimeBins, CUFFT_C2R, otherDimsProduct));

    // Step 1: Pad the real input
    int blockSize = 256;
    int gridSize = (paddedTimeBins * otherDimsProduct + blockSize - 1) / blockSize;
    padKernel<<<gridSize, blockSize>>>(d_H, d_H_pad_real, inputTimeBins, paddedTimeBins, otherDimsProduct, padding);
    CudaHelper::checkError(cudaDeviceSynchronize());

    // Step 2: Forward R2C FFT
    CUFFT_CHECK(cufftExecR2C(plan_forward, d_H_pad_real, d_H_fft));
    CudaHelper::checkError(cudaDeviceSynchronize());

    // Step 3: Multiply with K_fft (complex)
    gridSize = ((paddedTimeBins / 2 + 1) * otherDimsProduct + blockSize - 1) / blockSize;
    multiplyKernel << <gridSize, blockSize >> > (d_H_fft, d_K_fft, (paddedTimeBins / 2 + 1), otherDimsProduct, d_K_shape);
    CudaHelper::checkError(cudaDeviceSynchronize());

    // Step 4: Inverse C2R FFT
    CUFFT_CHECK(cufftExecC2R(plan_inverse, d_H_fft, d_H_ifft_real));
    CudaHelper::checkError(cudaDeviceSynchronize());

    // Normalize after inverse FFT
    float scale = 1.0f / paddedTimeBins;
    scaleKernelReal<<<gridSize, blockSize>>>(d_H_ifft_real, paddedTimeBins * otherDimsProduct, scale);
    CudaHelper::checkError(cudaDeviceSynchronize());

    // Step 5: Remove padding (real output)
    gridSize = (outputTimeBins * otherDimsProduct + blockSize - 1) / blockSize;
    removePadding<<<gridSize, blockSize>>>(d_H_ifft_real, d_HoK,
        paddedTimeBins, outputTimeBins,
        otherDimsProduct, padding,
        border_type);
    CudaHelper::checkError(cudaDeviceSynchronize());

    // Cleanup
    CUFFT_CHECK(cufftDestroy(plan_forward));
    CUFFT_CHECK(cufftDestroy(plan_inverse));
    CudaHelper::free(d_H_pad_real);
    CudaHelper::free(d_H_fft);
    CudaHelper::free(d_H_ifft_real);
}