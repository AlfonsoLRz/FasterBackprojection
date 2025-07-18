#pragma once

#include "stdafx.h"

#include <cufft.h>

// 3D texture for interpolation (x, y, z order)
//texture<float, 3, cudaReadModeElementType> tex_f_data;

inline __device__ glm::uint getKernelIdx(glm::uint x, glm::uint y, glm::uint t, const glm::uvec3& dataResolution)
{
    return t + dataResolution.z * (y + dataResolution.y * x);
}

inline __global__ void padIntensityFFT(const float* H, cufftComplex* H_pad, glm::uvec3 currentResolution, glm::uvec3 newResolution)
{
    const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
        return;

    float val = H[getKernelIdx(x, y, t, currentResolution)];
    //val *= val;
    H_pad[getKernelIdx(x, y, t, newResolution)].x = val;
}

inline __global__ void unpadIntensityFFT(float* H, const cufftComplex* H_pad, glm::uvec3 currentResolution, glm::uvec3 newResolution)
{
    const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
        return;

    H[getKernelIdx(x, y, t, currentResolution)] = H_pad[getKernelIdx(x, y, t, newResolution)].x;
}

inline __global__ void calculateSqrtTerm(float* sqrtTerm, glm::vec3 originalResolution, glm::uvec3 newResolution, float rangeWidth)
{
	const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= newResolution.x || y >= newResolution.y || t >= newResolution.z)
		return;

    glm::vec3 grid = (glm::vec3(x, y, t) - originalResolution) / originalResolution;
	sqrtTerm[getKernelIdx(x, y, t, newResolution)] = rangeWidth * (grid.x * grid.x + grid.y * grid.y) + grid.z * grid.z;
}

__device__ float trilinear_interpolate(float x, float y, float z, int N, int M, const float* f_data) {
    // Normalize coordinates to [0,1] range
    float xn = (x + N) / (2.0f * N);
    float yn = (y + N) / (2.0f * N);
    float zn = z / (2.0f * M);  // Assuming z was in [0, 2M-1]

    // Perform texture lookup (hardware-accelerated interpolation)
    //return tex3D(tex_f_data, xn, yn, zn);
    return .0f;
}

__global__ void stoltKernel(float* f_vol,
    const float* x_grid, const float* y_grid, const float* z_grid,
    int N, int M,
    float range, float width) {
    // x, y, z are the fastest to slowest varying indices (matches data layout)
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= 2 * N || y >= 2 * N || z >= 2 * M) return;

    // Linear index for input grids (x, y, z order)
    const int idx = x + (2 * N) * y + (2 * N) * (2 * N) * z;

    // Get normalized coordinates
    const float x_val = x_grid[idx] / N;
    const float y_val = y_grid[idx] / N;
    const float z_val = z_grid[idx] / M;

    // Stolt trick calculation
    const float ratio = (N * range) / (M * width * 4.0f);
    const float sqrt_term = sqrtf(ratio * ratio * (x_val * x_val + y_val * y_val) + z_val * z_val);

    // Interpolation and output
    if (z_val > 0.0f) {
        float interp_val = trilinear_interpolate(
            sqrt_term * N,  // Scale back to grid coordinates
            y_val * N,
            z_val * M,
            N, M, nullptr);  // nullptr since we use texture

        // Apply output conditions
        f_vol[idx] = interp_val * (fabsf(z_val) / sqrt_term);
    }
    else {
        f_vol[idx] = 0.0f;
    }
}

void launch_stolt_trick(float* d_f_vol,
    float* d_x_grid, float* d_y_grid, float* d_z_grid,
    float* d_f_data,
    int N, int M,
    float range, float width) {
    // Configure 3D texture
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(2 * N, 2 * N, 2 * M);
    cudaMalloc3DArray(&cuArray, &channelDesc, extent);

    // Copy data to 3D array (x, y, z order)
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr(d_f_data,
        2 * N * sizeof(float),
        2 * N, 2 * N);
    copyParams.dstArray = cuArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);

    // Bind texture
    //tex_f_data.normalized = false;  // Using explicit coordinates
    //tex_f_data.filterMode = cudaFilterModeLinear;
    //cudaBindTextureToArray(tex_f_data, cuArray, channelDesc);

    // Launch kernel with optimized block/grid layout
    dim3 block(8, 8, 4);  // Optimal for x,y,z memory access
    dim3 grid(
        (2 * N + block.x - 1) / block.x,
        (2 * N + block.y - 1) / block.y,
        (2 * M + block.z - 1) / block.z
    );

    //stolt_trick_kernel << <grid, block >> > (d_f_vol, d_x_grid, d_y_grid, d_z_grid,
    //    N, M, range, width);

    //// Cleanup
    //cudaUnbindTexture(tex_f_data);
    //cudaFreeArray(cuArray);
}