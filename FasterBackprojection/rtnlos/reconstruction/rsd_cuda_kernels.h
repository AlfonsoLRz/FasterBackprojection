#pragma once

#include "../stdafx.h"

#include <cufft.h>

#include "../math.cuh"

// rsdKernel, fills kernel with real and imaginary components of RSD kernel multiplied with expontial harmonic
inline __global__ void rsdKernel(cufftComplex* kernel, float z_hat2, float mulSquare, glm::uint width, glm::uint height)
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
inline __global__ void multiplySpectrumExpHarmonic(cufftComplex* ker1, float omega, float t, glm::uint width, glm::uint height)
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
	const cufftComplex* ker1, const cufftComplex* ker2, cufftComplex* dest,
	glm::uvec3 dataResolution, glm::uint sliceSize)
{
	const glm::uint wave = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= dataResolution.x || y >= dataResolution.y || wave >= dataResolution.z)
		return;

	glm::uint idx = wave * sliceSize + y * dataResolution.x + x;
	cuFloatComplex d = cuCmulf(ker1[idx], ker2[idx]);
	dest[idx] = d;
}

// Absolute value of a grid of complex numbers
inline __global__ void abs(const cufftComplex* complex, float* abs, glm::uvec3 dataResolution)
{
	const glm::uint y = blockIdx.x * blockDim.x + threadIdx.x, x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dataResolution.x || y >= dataResolution.y)
		return;

	glm::uint idx = y * dataResolution.x + x;
	abs[idx] = cuCabsf(complex[idx]);
}

inline __global__ void addScale(
	cufftComplex* dest, const cufftComplex* src, const float* weights, 
	glm::uvec3 dataResolution, glm::uint sliceSize)
{
	const glm::uint wave = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= dataResolution.x || y >= dataResolution.y || wave >= dataResolution.z)
		return;

	glm::uint idx3d = wave * sliceSize + y * dataResolution.x + x;
	glm::uint idx2d = y * dataResolution.x + x;

	cufftComplex src2 = src[idx3d];
	atomicAdd(&dest[idx2d].x, weights[wave] * src2.x);
	atomicAdd(&dest[idx2d].y, weights[wave] * src2.y);
}

// Depth Dependent Averaging for 1 depth
inline __global__ void DDA(
	float* cubeImages, int cur, const float* weights, 
	glm::uint width, glm::uint height, glm::uint sliceSize, glm::uint cubeSize, glm::uint numDepths)
{
	const glm::uint depth = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const glm::uint x = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= width || y >= height || depth >= numDepths)
		return;

	const glm::uint pixelIdx = y * width + x; 
	float* dst = &cubeImages[cubeSize * 3 + sliceSize * depth + pixelIdx];

	// For the current depth slice, fill cubeImages[3] with the weighted sums of cubeImages[0], [1] and [2]
	*dst = 0.f;
	for (int i = 0; i < 3; i++) 
	{
		int cubeIdx = (cur - 1 + i) % 3;
		if (cubeIdx < 0)
			cubeIdx += 3;

		float src = cubeImages[cubeSize * cubeIdx + sliceSize * depth + pixelIdx]; 
		float wt = weights[numDepths * i + depth];
		*dst += wt * src;
	}
}

// For a given pixel, find the max z value across all depths of that pixel.
inline __global__ void maxZ(
	const float* cube, float* image,
	glm::uint numDepths, glm::uint sliceSize)
{
	const glm::uint pixelID = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixelID >= sliceSize) 
		return;

	float m = cube[pixelID];
	for (glm::uint i = 1; i < numDepths; i++)
		m = fmaxf(m, cube[i * sliceSize + pixelID]); // fmaxf preferred in CUDA

	image[pixelID] = m;
}

// MagnitudeMax
inline __global__ void MagnitudeMax2(float* cmplx, float* re_out)
{
	// dest += scale * src * scale

	const int x_id = threadIdx.x;
	const int y_id = blockIdx.x;
	const int x_dim = blockDim.x;

	// 1 load/stores per complex component
	int idx = y_id * x_dim + x_id;
	float2 cmplx2 = reinterpret_cast<float2*>(cmplx)[idx];
	float m2 = cmplx2.x * cmplx2.x + cmplx2.y * cmplx2.y;
	float re = re_out[idx];
	re_out[idx] = (m2 > re) ? m2 : re;

	// 2 load/stores per complex component
	//int idx = y_id * x_dim + x_id;
	//float m2 = cmplx[idx * 2] * cmplx[idx * 2] + cmplx[idx * 2 + 1] * cmplx[idx * 2 + 1];
	//re_out[idx] = (m2 > re_out[idx]) ? m2 : re_out[idx];
}

// Sqrt
inline __global__ void Sqrt(float* m)
{
	// m = sqrt(m)
	const int x_id = threadIdx.x;
	const int y_id = blockIdx.x;
	const int x_dim = blockDim.x;

	int idx = y_id * x_dim + x_id;
	m[idx] = sqrt(m[idx]);
}


