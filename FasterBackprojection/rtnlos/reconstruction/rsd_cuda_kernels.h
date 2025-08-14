#pragma once

#include "../stdafx.h"

#include <cufft.h>

#include "../math.cuh"

// RsdKernel, fills kernel with real and imaginary components of RSD kernel multiplied with expontial harmonic
inline __global__ void rsdKernel(cufftComplex* __restrict__ kernel, float z_hat2, float mulSquare, glm::uint width, glm::uint height)
{
	const glm::uint y = blockIdx.x * blockDim.x + threadIdx.x, x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;

	float g_x = static_cast<float>(x) - static_cast<float>(width) / 2;
	float g_y = static_cast<float>(y) - static_cast<float>(height) / 2;
	float r = sqrt(1 + g_x * g_x / z_hat2 + g_y * g_y / z_hat2);
	float phi = 2 * PI * r * z_hat2 / (mulSquare * static_cast<float>(width));
	glm::uint idx = y * width + x;

	// Convert phase to complex
	kernel[idx].x = cosf(phi);
	kernel[idx].y = sinf(phi);
}

// MulSpectrum (not used, this operation has been incorporated into rsdKernel)
inline __global__ void multiplySpectrumExpHarmonic(cufftComplex* __restrict__ ker1, float omega, float t, glm::uint width, glm::uint height)
{
	const glm::uint y = blockIdx.x * blockDim.x + threadIdx.x, x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;

	glm::uint idx = y * width + x;
	cufftComplex& k1 = ker1[idx];
	const cufftComplex k2 = { cos(omega * t), sin(omega * t) };
	k1 = cuCmulf(k1, k2);
}

// Performs element-wise multiplication (convolution) on many images all at once
inline __global__ void multiplySpectrumMany(
	const cufftComplex* __restrict__ ker1, const cufftComplex* __restrict__ ker2, cufftComplex* __restrict__ dest, const float* __restrict__ weights,
	glm::uint numFrequencies, glm::uint sliceSize)
{
	const glm::uint xy = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint wave = blockIdx.y * blockDim.y + threadIdx.y;
	if (xy >= sliceSize || wave >= numFrequencies)
		return;

	glm::uint idx = wave * sliceSize + xy;
	cufftComplex mult = cuCmulf(ker1[idx], ker2[idx]);
	atomicAdd(&dest[xy].x, weights[wave] * mult.x);
	atomicAdd(&dest[xy].y, weights[wave] * mult.y);
}

// Absolute value of a grid of complex numbers
inline __global__ void abs(const cufftComplex* __restrict__ complex, float* __restrict__ abs, glm::uint sliceSize)
{
	const glm::uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= sliceSize)
		return;

	abs[tid] = cuCabsf(complex[tid]);
}

inline __global__ void addScale(
	cufftComplex* dest, const cufftComplex* __restrict__ src, const float* __restrict__ weights,
	glm::uint numFrequencies, glm::uint sliceSize)
{
	const glm::uint xy = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint wave = blockIdx.y * blockDim.y + threadIdx.y;
	if (xy >= sliceSize || wave >= numFrequencies)
		return;

	cufftComplex src2 = src[wave * sliceSize + xy];
	atomicAdd(&dest[xy].x, weights[wave] * src2.x);
	atomicAdd(&dest[xy].y, weights[wave] * src2.y);
}

inline __global__ void addScaleSM(
	cufftComplex* __restrict__ dest,
	const cufftComplex* __restrict__ src,
	const float* __restrict__ weights,
	glm::uint numFrequencies,
	glm::uint sliceSize)
{
	extern __shared__ cufftComplex sdata[]; // sdata layout: threadIdx.y stores partial sums for xy

	const glm::uint xy = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint ty = threadIdx.y; // Used for wave dimension
	const glm::uint blockWave = blockDim.y;

	if (xy >= sliceSize) return;

	// Each thread computes a partial sum over its assigned waves
	cufftComplex partial = make_cuComplex(0.0f, 0.0f);
	for (glm::uint wave = ty; wave < numFrequencies; wave += blockWave) {
		cufftComplex val = src[wave * sliceSize + xy];
		float w = weights[wave];
		partial.x += w * val.x;
		partial.y += w * val.y;
	}

	// Store partial sum in shared memory
	sdata[ty] = partial;
	__syncthreads();

	// Parallel reduction in shared memory
	for (unsigned int s = blockWave / 2; s > 0; s >>= 1) 
	{
		if (ty < s) 
		{
			sdata[ty].x += sdata[ty + s].x;
			sdata[ty].y += sdata[ty + s].y;
		}
		__syncthreads();
	}

	// Thread 0 writes the final sum for this xy
	if (ty == 0) 
		dest[xy] = sdata[0];
}

// Depth Dependent Averaging for 1 depth
inline __global__ void DDA(
	float* __restrict__ cubeImages, int cur, const float* __restrict__ weights,
	glm::uint sliceSize, glm::uint cubeSize, glm::uint numDepths)
{
	const glm::uint pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint depth = blockIdx.y * blockDim.y + threadIdx.y;
	if (pixelIdx >= sliceSize || depth >= numDepths)
		return;

	// For the current depth slice, fill cubeImages[3] with the weighted sums of cubeImages[0], [1] and [2]
	float acc = 0.f;
	for (int i = 0; i < 3; i++) 
	{
		int cubeIdx = (cur - 1 + i + 3) % 3;
		acc += weights[numDepths * i + depth] * cubeImages[cubeSize * cubeIdx + sliceSize * depth + pixelIdx];
	}

	cubeImages[cubeSize * 3 + sliceSize * depth + pixelIdx] = acc;
}

// For a given pixel, find the max z value across all depths of that pixel.
inline __global__ void maxZ(
	const float* __restrict__ cube, float* image,
	glm::uint width, glm::uint height, glm::uint numDepths, glm::uint sliceSize,
	glm::uvec2 shift)
{
	const glm::uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;

	// Circular shift
	const glm::uint xs = (x + shift.x) % width;
	const glm::uint ys = (y + shift.y) % height;
	const glm::uint pixelID = ys * width + xs;

	float m = cube[pixelID];
	#pragma unroll 8
	for (glm::uint i = 1; i < numDepths; i++)
		m = fmaxf(m, cube[i * sliceSize + pixelID]); 

	image[y * width + x] = m; // Write to original grid location
}

inline __global__ void writeImage(const float* __restrict__ image, glm::uint numPixels, float4* texture)
{
	const glm::uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numPixels)
		return;

	float gray = image[tid];
	texture[tid] = make_float4(gray, gray, gray, 1.0f);
}
