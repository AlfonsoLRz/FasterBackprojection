#pragma once

#include "stdafx.h"

//

inline __global__ void writeImage(const float4* __restrict__ image, glm::uint width, glm::uint numPixels, cudaSurfaceObject_t surfaceObj)
{
	const glm::uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numPixels)
		return;

	const glm::uint x = tid % width, y = tid / width;
	surf2Dwrite(image[tid], surfaceObj, x * sizeof(float4), y);
}