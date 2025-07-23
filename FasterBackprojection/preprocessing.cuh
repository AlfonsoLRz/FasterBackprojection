#pragma once

#include "stdafx.h"

#include "GpuStructs.cuh"
#include "math.cuh"

inline __global__ void compensateLaserPosition(
	float* __restrict__ intensity, glm::uint numSensorTargets, glm::uint spatialSize, glm::uint numTimeBins)
{
	const glm::uint xy = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint timeBin = blockIdx.y * blockDim.y + threadIdx.y;

	if (xy >= spatialSize || timeBin >= numTimeBins)
		return;

	const glm::uint laserTargetIdx = xy / numSensorTargets;
	const glm::vec3 w_i = rtRecInfo._laserPosition - laserTargets[laserTargetIdx];
	const float d = glm::length(w_i);
	const float cosTerm = glm::dot(w_i * safeRCP(d), laserTargetsNormals[laserTargetIdx]);
	intensity[laserTargetIdx * numTimeBins + timeBin] /= (cosTerm * safeRCP(d * d));
}
