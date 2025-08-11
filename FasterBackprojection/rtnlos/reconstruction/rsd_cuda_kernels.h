#pragma once

#include <cuda_runtime.h>

__global__ void RSD_Kernel(float* ker, float z_hat2, float mul_square);
__global__ void MulSpectrumExpHarmonic(float* ker1, float omega, float t);
__global__ void MulSpectrumMany(float* ker1, float* ker2, float* dest, int num_pixels);
__global__ void AddScale(float* dest, float* src, float scale);
__global__ void AddScaleMany(float* dest, float* src, float* scale, int num_waves);
__global__ void MagnitudeMax2(float* cmplx, float* re_out);
__global__ void Abs(float* cmplx, float* abs);
__global__ void Sqrt(float* m);
__global__ void DDA(float* cube_images, int cur, float* weights);
__global__ void MaxZ(float *img_3d, int num_depths, float *img_2d);


