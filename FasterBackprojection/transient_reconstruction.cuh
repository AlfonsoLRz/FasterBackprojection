#pragma once

#include "GpuStructs.cuh"
#include "math.cuh"
#include "transient_utils.cuh"

#define BLOCK_X_CONFOCAL 16
#define BLOCK_Y_CONFOCAL 16

#define BLOCK_X_EXHAUSTIVE 16
#define BLOCK_Y_EXHAUSTIVE 2

//#define ACCUMULATE_VOXEL_SCATTERING

__global__ void backproject(float* __restrict__ activation, const double* __restrict__ depths, glm::uint numDepths, glm::uint checkPosIdx)
{
	glm::uint l = blockIdx.x * blockDim.x + threadIdx.x;
	glm::uint s = blockIdx.y * blockDim.y + threadIdx.y;
	glm::uint d = blockIdx.z * blockDim.z + threadIdx.z;

	if (l >= rtRecInfo._numLaserTargets || s >= rtRecInfo._numSensorTargets || d >= numDepths)
		return;

	const glm::vec3 lPos = laserTargets[l];
	const glm::vec3 sPos = laserTargets[s];
	const glm::vec3 checkPos = laserTargets[checkPosIdx] + rtRecInfo._relayWallNormal * static_cast<float>(depths[d]);

	float traversedDistance = glm::distance(lPos, checkPos) + glm::distance(checkPos, sPos) - rtRecInfo._timeOffset;
	if (rtRecInfo._discardFirstLastBounces == 0)
		traversedDistance += glm::distance(lPos, rtRecInfo._laserPosition) + glm::distance(sPos, rtRecInfo._sensorPosition);

	const int timeBin = static_cast<int>(traversedDistance / rtRecInfo._timeStep);
	if (timeBin < 0 || timeBin >= rtRecInfo._numTimeBins)
		return;

	float intensity = intensityCube[getExhaustiveTransientIndex(l, s, timeBin)];
	if (intensity > EPS)
		atomicAdd(&activation[checkPosIdx * numDepths + d], intensity);
}

__global__ void backprojectConfocal(float* __restrict__ activation, const double* __restrict__ depths, glm::uint numDepths)
{
	glm::uint l = blockIdx.x * blockDim.x + threadIdx.x;
	glm::uint c = blockIdx.y * blockDim.y + threadIdx.y;
	glm::uint d = blockIdx.z * blockDim.z + threadIdx.z;

	if (l >= rtRecInfo._numLaserTargets || c >= rtRecInfo._numLaserTargets || d >= numDepths)
		return;

	glm::uint localC = threadIdx.y;
	glm::uint localD = threadIdx.z;
	__shared__ float shActivation[BLOCK_X_CONFOCAL][BLOCK_Y_CONFOCAL];

	if (localC < BLOCK_X_CONFOCAL && localD < BLOCK_Y_CONFOCAL)
		shActivation[localC][localD] = 0.0f;

	__syncthreads();

	const glm::vec3 lPos = laserTargets[l];
	const glm::vec3 checkPos = laserTargets[c] + rtRecInfo._relayWallNormal * static_cast<float>(depths[d]);

	float traversalTime = glm::distance(lPos, checkPos) * 2.0f - rtRecInfo._timeOffset;
	if (rtRecInfo._discardFirstLastBounces == 0)
		traversalTime += glm::distance(lPos, rtRecInfo._laserPosition) + glm::distance(lPos, rtRecInfo._sensorPosition);

	const int timeBin = static_cast<int>(traversalTime / rtRecInfo._timeStep);
	if (timeBin < 0 || timeBin >= rtRecInfo._numTimeBins)
		return;

	float intensity = intensityCube[getConfocalTransientIndex(l, timeBin)];
	if (intensity > EPS)
		atomicAdd(&shActivation[localC][localD], intensity);

	__syncthreads();

	// From shared memory back to global memory
	if (threadIdx.x == 0) 
		atomicAdd(&activation[c * numDepths + d], shActivation[localC][localD]);
}

__global__ void backprojectExhaustive(float* __restrict__ activation, const double* __restrict__ depths, glm::uint numDepths)
{
	glm::uint ls = blockIdx.x * blockDim.x + threadIdx.x;
	glm::uint c = blockIdx.y * blockDim.y + threadIdx.y;
	glm::uint d = blockIdx.z * blockDim.z + threadIdx.z;

	if (ls >= rtRecInfo._numLaserTargets * rtRecInfo._numSensorTargets || c >= rtRecInfo._numLaserTargets || d >= numDepths)
		return;

	glm::uint localC = threadIdx.y;
	glm::uint localD = threadIdx.z;
	__shared__ float shActivation[BLOCK_X_EXHAUSTIVE][BLOCK_Y_EXHAUSTIVE];

	if (localC < BLOCK_X_EXHAUSTIVE && localD < BLOCK_Y_EXHAUSTIVE)
		shActivation[localC][localD] = 0.0f;

	__syncthreads();

	const glm::uint l = ls % rtRecInfo._numLaserTargets;
	const glm::uint s = ls / rtRecInfo._numLaserTargets;
	const glm::vec3 lPos = laserTargets[l];
	const glm::vec3 sPos = laserTargets[s];
	const glm::vec3 checkPos = laserTargets[c] + rtRecInfo._relayWallNormal * static_cast<float>(depths[d]);

	float traversedDistance = glm::distance(lPos, checkPos) + glm::distance(checkPos, sPos) - rtRecInfo._timeOffset;
	if (rtRecInfo._discardFirstLastBounces == 0)
		traversedDistance += glm::distance(lPos, rtRecInfo._laserPosition) + glm::distance(sPos, rtRecInfo._sensorPosition);

	const int timeBin = static_cast<int>(traversedDistance / rtRecInfo._timeStep);
	if (timeBin < 0 || timeBin >= rtRecInfo._numTimeBins)
		return;

	float intensity = intensityCube[getExhaustiveTransientIndex(l, s, timeBin)];
	if (intensity > EPS)
		atomicAdd(&shActivation[localC][localD], intensity);

	__syncthreads();

	// From shared memory back to global memory
	if (threadIdx.x == 0)
		atomicAdd(&activation[c * numDepths + d], shActivation[localC][localD]);
}

__global__ void backprojectConfocalVoxel(float* __restrict__ activation, glm::uint sliceSize)
{
	glm::uint v = blockIdx.x * blockDim.x + threadIdx.x; 
	glm::uint l = blockIdx.y * blockDim.y + threadIdx.y; 
	glm::uint voxelZ = blockIdx.z * blockDim.z + threadIdx.z;

	if (l >= rtRecInfo._numLaserTargets || v >= sliceSize || voxelZ >= rtRecInfo._voxelResolution.z)
		return;

	const glm::uint vx = v % rtRecInfo._voxelResolution.x;
	const glm::uint vy = v / rtRecInfo._voxelResolution.x;

	const glm::vec3 lPos = laserTargets[l];
	const glm::vec3 checkPos = rtRecInfo._hiddenVolumeMin + glm::vec3(vx, voxelZ, vy) * rtRecInfo._hiddenVolumeVoxelSize + 0.5f * rtRecInfo._hiddenVolumeVoxelSize;

	float dist = glm::distance(lPos, checkPos) * 2.0f - rtRecInfo._timeOffset;
	if (rtRecInfo._discardFirstLastBounces == 0)
		dist += glm::distance(lPos, rtRecInfo._laserPosition) + glm::distance(lPos, rtRecInfo._sensorPosition);

#ifdef ACCUMULATE_VOXEL_SCATTERING
	float voxelExtent = glm::length(rtRecInfo._hiddenVolumeVoxelSize) / 2.0f;
	float minDist = glm::max(.0f, dist - voxelExtent), maxDist = dist + voxelExtent;
	float d = minDist;

	while (d < maxDist)
	{
		const int timeBin = static_cast<int>(d / rtRecInfo._timeStep);
		if (timeBin < 0 || timeBin >= rtRecInfo._numTimeBins)
			return;
		const float backscatteredLight = intensityCube[getConfocalTransientIndex(l, timeBin)];
		if (backscatteredLight > EPS)
			atomicAdd(&activation[sliceSize * voxelZ + v], backscatteredLight * safeRCP(dist * dist));
		d += rtRecInfo._timeStep;
	}
#else
	const int timeBin = static_cast<int>(dist / rtRecInfo._timeStep);
	if (timeBin < 0 || timeBin >= rtRecInfo._numTimeBins)
		return;

	const float backscatteredLight = intensityCube[getConfocalTransientIndex(l, timeBin)];
	if (backscatteredLight > EPS)
		atomicAdd(&activation[sliceSize * voxelZ + v], backscatteredLight * safeRCP(dist * dist));
#endif
}

__global__ void backprojectExhaustiveVoxel(float* __restrict__ activation, glm::uint sliceSize)
{
	glm::uint v = blockIdx.x * blockDim.x + threadIdx.x;
	glm::uint l = blockIdx.y * blockDim.y + threadIdx.y;
	glm::uint voxelZ = blockIdx.z * blockDim.z + threadIdx.z;

	if (l >= rtRecInfo._numLaserTargets || v >= sliceSize || voxelZ >= rtRecInfo._voxelResolution.z)
		return;

	const glm::uint vx = v % rtRecInfo._voxelResolution.x;
	const glm::uint vy = v / rtRecInfo._voxelResolution.x;

	const glm::vec3 lPos = laserTargets[l];
	const glm::vec3 checkPos = rtRecInfo._hiddenVolumeMin + glm::vec3(vx, voxelZ, vy) * rtRecInfo._hiddenVolumeVoxelSize + 0.5f * rtRecInfo._hiddenVolumeVoxelSize;

	float dist = glm::distance(lPos, checkPos) * 2.0f - rtRecInfo._timeOffset;
	if (rtRecInfo._discardFirstLastBounces == 0)
		dist += glm::distance(lPos, rtRecInfo._laserPosition) + glm::distance(lPos, rtRecInfo._sensorPosition);

	const int timeBin = static_cast<int>(dist / rtRecInfo._timeStep);
	if (timeBin < 0 || timeBin >= rtRecInfo._numTimeBins)
		return;

	const float backscatteredLight = intensityCube[getConfocalTransientIndex(l, timeBin)];
	if (backscatteredLight > EPS)
		atomicAdd(&activation[sliceSize * voxelZ + v], backscatteredLight * safeRCP(dist * dist));
}

__global__ void precomputeDataConfocal(
	float* __restrict__ activation,
	glm::vec3* __restrict__ means, float* __restrict__ radius, float* __restrict__ colour,
	uint32_t* tilesTouched, const double* __restrict__ depths, glm::uint numDepths,
	glm::vec3 relayWallMin, glm::vec3 relayWallMax, glm::vec3 relayWallPixelSize, glm::vec3 relayWallMask,
	glm::vec4* debugBuffer)
{
	const glm::uint l = blockIdx.x * blockDim.x + threadIdx.x;
	const glm::uint d = blockIdx.y * blockDim.y + threadIdx.y;

	if (l >= rtRecInfo._numLaserTargets || d >= numDepths)
		return;

	//if (l != 128 * 256 + 128)
	//	return;

	const glm::vec3 relayWallPos = laserTargets[l];
	const float depth = static_cast<float>(depths[d]);

	// Get ellipsoid center at depth
	const glm::vec3 center = relayWallPos;

	// Compute max extent (radius) of splat (simplified Gaussian spread assumption)
	const float maxRadius = depth; // conservative bound
	glm::vec3 extent = glm::vec3(maxRadius) * relayWallMask;

	// Clamp to reconstruction volume AABB
	glm::vec3 minWorld = glm::max(center - extent, relayWallMin);
	glm::vec3 maxWorld = glm::min(center + extent, relayWallMax);

	// Convert to image grid coordinates
	glm::vec3 minCoord = (minWorld - relayWallMin) * safeRCP(relayWallPixelSize);
	glm::vec3 maxCoord = (maxWorld - relayWallMin) * safeRCP(relayWallPixelSize);
	glm::ivec3 rectMin = glm::clamp(glm::ivec3(glm::floor(minCoord)), glm::ivec3(0), glm::ivec3(256 - 1));
	glm::ivec3 rectMax = glm::clamp(glm::ivec3(glm::ceil(maxCoord)), glm::ivec3(0), glm::ivec3(256 - 1));

	const int tiles = (rectMax.x - rectMin.x) * (rectMax.z - rectMin.z);
	debugBuffer[l] = glm::vec4(tiles, maxCoord);
	if (tiles == 0)
		return;

	// Store mean, radius, colour
	const int idx = d * rtRecInfo._numLaserTargets + l;
	means[idx] = center;
	radius[idx] = maxRadius;
	tilesTouched[idx] = tiles;

	// DEBUG: Write to debug image (1.0f where touched)
	for (int y = 0; y < 256; ++y) {
		for (int x = 0; x < 256; ++x) {
			const int debugIdx = y * 256 + x;

			glm::vec3 lPos = laserTargets[l];
			const glm::vec3 checkPos = laserTargets[debugIdx] + rtRecInfo._relayWallNormal * static_cast<float>(depths[d]);

			float traversalTime = glm::distance(lPos, checkPos) + glm::distance(checkPos, lPos) - rtRecInfo._timeOffset;
			if (rtRecInfo._discardFirstLastBounces == 0)
				traversalTime += glm::distance(lPos, rtRecInfo._laserPosition) + glm::distance(lPos, rtRecInfo._sensorPosition);

			const int timeBin = static_cast<int>(traversalTime / rtRecInfo._timeStep);
			if (timeBin < 0 || timeBin >= rtRecInfo._numTimeBins)
				return;

			activation[d * rtRecInfo._numLaserTargets + debugIdx] += intensityCube[getConfocalTransientIndex(l, timeBin)];
		}
	}
}

__global__ void normalizeReconstruction(float* v, glm::uint size, const float* maxValue, const float* minValue)
{
	const glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	v[idx] = (v[idx] - *minValue) * safeRCP(*maxValue - *minValue);
}

