#include "rsd_cuda_kernels.h"
#include <stdio.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// RSD_Kernel, fills ker with real and imag component of RSD kernel multiplied with exp harmonic
__global__ void RSD_Kernel(float* ker, float z_hat2, float mul_square) {
	const int x_id = threadIdx.x;
	const int y_id = blockIdx.x;
	const int x_dim = blockDim.x;
	const int y_dim = gridDim.x;

	float g_x = x_id - x_dim / 2;
	float g_y = y_id - y_dim / 2;

	float r = sqrt(1 + g_x * g_x / z_hat2 + g_y * g_y / z_hat2);
	float phi = 2 * M_PI * r * z_hat2 / (mul_square * x_dim);

	int idx = (x_id * y_dim + y_id) * 2;
	// convert phase to complex and store as 2 channel matrix
	ker[idx + 0] = cos(phi);
	ker[idx + 1] = sin(phi);
}

// MulSpectrum (not used, this operation has been incorporated into RSD_Kernel)
__global__ void MulSpectrumExpHarmonic(float* ker1, float omega, float t) {
	const int x_id = threadIdx.x;
	const int y_id = blockIdx.x;
	const int x_dim = blockDim.x;

	int idx = (y_id * x_dim + x_id) * 2;

	float* k1 = &ker1[idx];
	float k2[] = { cos(omega * t), sin(omega * t) };

	float re = k1[0] * k2[0] - k1[1] * k2[1];
	float im = k1[0] * k2[1] + k1[1] * k2[0];

	k1[0] = re;
	k1[1] = im;
}

// MulSpectrumMany
// performs element-wise multiplication (convolution) on many images all at once.
__global__ void MulSpectrumMany(float* ker1, float* ker2, float* dest, int num_pixels) {
	const int x_id = threadIdx.x;
	const int y_id = blockIdx.x;
	const int x_dim = blockDim.x;
	const int wave_num = blockIdx.y;

	if (y_id * x_dim + x_id >= num_pixels)
		return;

	// 1 load/store per complex component
	const int wave_size = num_pixels;
	int idx = wave_num * wave_size + (y_id * x_dim + x_id);
	cuFloatComplex k1 = reinterpret_cast<cuFloatComplex*>(ker1)[idx];
	cuFloatComplex k2 = reinterpret_cast<cuFloatComplex*>(ker2)[idx];
	cuFloatComplex d = cuCmulf(k1, k2);
	reinterpret_cast<cuFloatComplex*>(dest)[idx] = d;

	// 2 loads/stores per complex component
	//const int wave_size = num_pixels * 2;
	//int idx = wave_num * wave_size + ((y_id * x_dim + x_id) * 2);
	//float* k1 = &ker1[idx];
	//float* k2 = &ker2[idx];
	//float* d = &dest[idx];
	//d[0] = k1[0] * k2[0] - k1[1] * k2[1];
	//d[1] = k1[0] * k2[1] + k1[1] * k2[0];
}

// AddScale
__global__ void AddScale(float* dest, float* src, float scale) {
	// dest += scale * src * scale
	const int x_id = threadIdx.x;
	const int y_id = blockIdx.x;
	const int x_dim = blockDim.x;

	// 1 load/store per complex component
	int idx = y_id * x_dim + x_id;	
	float2 dst2 = reinterpret_cast<float2*>(dest)[idx];
	float2 src2 = reinterpret_cast<float2*>(src)[idx];
	dst2.x += scale * src2.x;
	dst2.y += scale * src2.y;
	reinterpret_cast<float2*>(dest)[idx] = dst2;

	// 2 load/stores per complex component
	//int idx = (y_id * x_dim + x_id) * 2;
	//dest[idx + 0] += scale * src[idx + 0];
	//dest[idx + 1] += scale * src[idx + 1];
}

// AddScaleMany
// this is a parallel reduction algorithm that sums up pointwise elements of num_waves number of matrices
// In testing, calling this once was slower than calling kernel 'AddScale' num_waves times sequentially.
// I think there are more opportunities to optimize this kernel...but for now sequentially calling AddScale()
// is still pretty fast.
__global__ void AddScaleMany(float* dest, float* src, float* scale, int num_waves) {
	__shared__ float sdata[128 * 2];

	const int tid = threadIdx.x;
	const int x_id = blockIdx.x;
	const int y_id = blockIdx.y;
	const int x_dim = gridDim.x;
	const int y_dim = gridDim.y;
	const int wave_size = x_dim * y_dim * 2;

	float sc = scale[tid];
	int idx_dest = (y_id * x_dim + x_id) * 2;
	int idx_src = (wave_size * tid) + idx_dest;

	// load shared memory with what we're going to sum
	if (tid < num_waves)
	{
		sdata[tid * 2 + 0] = sc * src[idx_src + 0];
		sdata[tid * 2 + 1] = sc * src[idx_src + 1];
	}
	else
	{
		sdata[tid * 2 + 0] = 0.0f;
		sdata[tid * 2 + 1] = 0.0f;
	}

	__syncthreads();

	// now do a standard reduction
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid * 2 + 0] += sdata[(tid + s) * 2 + 0];
			sdata[tid * 2 + 1] += sdata[(tid + s) * 2 + 1];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		src[idx_dest + 0] = sdata[0];
		src[idx_dest + 1] = sdata[1];
	}
}

// MagnitudeMax
__global__ void MagnitudeMax2(float* cmplx, float* re_out) {
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

// absolute value of a grid of complex numbers
__global__ void Abs(float* cmplx, float* abs) {
	// dest += scale * src * scale

	const int x_id = threadIdx.x;
	const int y_id = blockIdx.x;
	const int x_dim = blockDim.x;

	// 1 load/stores per complex component
	int idx = y_id * x_dim + x_id;
	float2 c = reinterpret_cast<cuComplex*>(cmplx)[idx];
	float z = cuCabsf(c);
	abs[idx] = z;
}

// Sqrt
__global__ void Sqrt(float* m) {
	// m = sqrt(m)
	const int x_id = threadIdx.x;
	const int y_id = blockIdx.x;
	const int x_dim = blockDim.x;

	int idx = y_id * x_dim + x_id;
	m[idx] = sqrt(m[idx]);
}

// Depth Dependent Averaging for 1 depth
__global__ void DDA(float* cube_images, int cur, float* weights)
{
	const int x_id = threadIdx.x;
	const int y_id = blockIdx.x;
	const int depth = blockIdx.y;
	const int x_dim = blockDim.x;
	const int y_dim = gridDim.x;
	const int num_depths = gridDim.y;

	const int slice_sz = x_dim * y_dim;
	const int cube_sz = slice_sz * num_depths;
	const int idx = y_id * x_dim + x_id; // (x,y) at depth 
	float* dst = cube_images +
		(cube_sz * 3) + // offset to dst cube
		(slice_sz * depth) + // offset to cur depth
		idx; // offset to pixel at current depth

	// for the current depth slice,
	// fill cube_images[3] with the weighted sums of cube_images[0], [1], and [2]
	*dst = 0.f;
	for (int i = 0; i < 3; i++) {
		int cube_idx = ((cur - 1) + i) % 3;
		if (cube_idx < 0)
			cube_idx += 3;
		float src = *(cube_images +
			(cube_sz * cube_idx) + // offset to dst cube
			(slice_sz * depth) + // offset to cur depth
			idx); // offset to pixel at current depth
		float wt = *(weights + (num_depths * i) + depth);
		*dst += wt * src;
	}
}

// for a given pixel, find the max z value across all depths of that pixel.
__global__ void MaxZ(float* img_3d, int num_depths, float* img_2d)
{
	const int x_id = threadIdx.x;
	const int y_id = blockIdx.x;
	const int x_dim = blockDim.x;
	const int y_dim = gridDim.x;

	const int slice_sz = x_dim * y_dim;
	const int idx = y_id * x_dim + x_id;

	float m = img_3d[idx];
	for (int i = 1; i < num_depths; i++) {
		float cur = img_3d[i * slice_sz + idx];
		m = cur > m ? cur : m;
	}
	img_2d[idx] = m;
}
