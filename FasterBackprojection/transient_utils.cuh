#pragma once

#include "stdafx.h"
#include "GpuStructs.cuh"

__forceinline__ __device__ glm::uint getConfocalTransientIndex(glm::uint laserTargetIdx, glm::uint timeBin)
{
	return timeBin * rtRecInfo._numLaserTargets + laserTargetIdx;
}

__forceinline__ __device__ glm::uint getExhaustiveTransientIndex(glm::uint laserTargetIdx, glm::uint sensorTargetIdx, glm::uint timeBin)
{
	return laserTargetIdx * rtRecInfo._numSensorTargets * rtRecInfo._numTimeBins + sensorTargetIdx * rtRecInfo._numTimeBins + timeBin;
}

// Random number generation 

__forceinline__ __device__ float getUniformRandom(const float* __restrict__ noise, glm::uint noiseBufferSize, glm::uint& randomState)
{
	randomState = randomState * 747796405u + 2891336453u;
	return noise[randomState % noiseBufferSize];
}

inline __device__ int getMISIndex(glm::uint& idx, const float* __restrict__ sum, glm::uint size, const float* __restrict__ noise, glm::uint noiseBufferSize)
{
	const float r = getUniformRandom(noise, noiseBufferSize, idx);

	// Binary search over the CDF
	int left = 0, right = static_cast<int>(size) - 1;
	for (int i = static_cast<int>(std::log2(static_cast<float>(size))); i >= 0; --i) {
		int mid = (left + right) >> 1;

		// Avoid branching using a mask
		bool moveRight = r >= sum[mid];
		left = moveRight ? mid + 1 : left;
		right = moveRight ? right : mid;
	}

	return left;
}

inline __device__ glm::uint getMISIndexAlias(
	glm::uint& idx, const glm::uint* __restrict__ aliasTable, const float* __restrict__ probTable, glm::uint size, 
	const float* __restrict__ noise, glm::uint noiseBufferSize)
{
	const float r1 = getUniformRandom(noise, noiseBufferSize, idx);
	const float r2 = getUniformRandom(noise, noiseBufferSize, idx);
	glm::uint k = static_cast<glm::uint>(r1 * size);

	if (r2 < probTable[k])
		return k;

	return aliasTable[k];
}