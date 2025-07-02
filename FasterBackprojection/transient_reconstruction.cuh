#pragma once

#include "GpuStructs.cuh"
#include "math.cuh"
#include "transient_utils.cuh"

#define BLOCK_C 16
#define BLOCK_D 16

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

	// Local indices within the block tile
	glm::uint localC = threadIdx.y;
	glm::uint localD = threadIdx.z;

	// Allocate shared memory per (c, d) tile
	__shared__ float shActivation[BLOCK_C][BLOCK_D];

	// Initialize shared memory
	if (localC < BLOCK_C && localD < BLOCK_D)
		shActivation[localC][localD] = 0.0f;

	__syncthreads();

	if (l >= rtRecInfo._numLaserTargets || c >= rtRecInfo._numLaserTargets || d >= numDepths)
		return;

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

	// Write shared memory back to global memory
	if (threadIdx.x == 0 && c < rtRecInfo._numLaserTargets && d < numDepths) 
		atomicAdd(&activation[c * numDepths + d], shActivation[localC][localD]);
}

__global__ void backprojectExhaustive(float* __restrict__ activation, const double* __restrict__ depths, glm::uint numDepths)
{
	glm::uint lc = blockIdx.x * blockDim.x + threadIdx.x;
	glm::uint l = lc % 256;
	glm::uint s = lc / 256;
	glm::uint c = blockIdx.y * blockDim.y + threadIdx.y; 
	glm::uint d = blockIdx.z * blockDim.z + threadIdx.z;

	if (l >= rtRecInfo._numLaserTargets || c >= rtRecInfo._numLaserTargets || d >= numDepths)
		return;

	// Compute traversal time
	const glm::vec3 lPos = laserTargets[l];
	const glm::vec3 checkPos = laserTargets[c] + rtRecInfo._relayWallNormal * static_cast<float>(depths[d]);
	float traversalTime = glm::distance(lPos, checkPos) * 2.0f - rtRecInfo._timeOffset;

	if (rtRecInfo._discardFirstLastBounces == 0) 
		traversalTime += glm::distance(lPos, rtRecInfo._laserPosition) + glm::distance(lPos, rtRecInfo._sensorPosition);

	// Time bin check
	const int timeBin = static_cast<int>(traversalTime / rtRecInfo._timeStep);
	if (timeBin < 0 || timeBin >= rtRecInfo._numTimeBins)
		return;

	// Atomic update in shared memory
	float intensity = intensityCube[getExhaustiveTransientIndex(l, s, timeBin)];
	if (intensity > EPS)
		atomicAdd(&activation[c * numDepths + d], intensity);
}

__global__ void backprojectVoxel(float* __restrict__ activation, glm::uint sliceSize, glm::uint voxelZ)
{
	glm::uint l = blockIdx.x * blockDim.x + threadIdx.x;
	glm::uint v = blockIdx.y * blockDim.y + threadIdx.y;

	if (l >= rtRecInfo._numLaserTargets || v >= sliceSize)
		return;

	const glm::uint vx = v % rtRecInfo._voxelResolution.x;
	const glm::uint vy = v / rtRecInfo._voxelResolution.x;

	const glm::vec3 lPos = laserTargets[l];
	const glm::vec3 checkPos = rtRecInfo._hiddenVolumeMin + glm::vec3(vx, voxelZ, vy) * rtRecInfo._hiddenVolumeVoxelSize + 0.5f * rtRecInfo._hiddenVolumeVoxelSize;

	float dist = glm::distance(lPos, checkPos) * 2.0f - rtRecInfo._timeOffset;
	if (rtRecInfo._discardFirstLastBounces == 0)
		dist += glm::distance(lPos, rtRecInfo._laserPosition) + glm::distance(lPos, rtRecInfo._sensorPosition);

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
			atomicAdd(&activation[sliceSize * voxelZ + v], backscatteredLight);
		d += rtRecInfo._timeStep;
	}

	//const int timeBin = static_cast<int>(dist / rtRecInfo._timeStep);
	//if (timeBin < 0 || timeBin >= rtRecInfo._numTimeBins)
	//	return;

	//const float backscatteredLight = rtRecInfo.intensityCube[getConfocalTransientIndex(l, timeBin)];
	//if (backscatteredLight < EPS)
	//	return;

	//atomicAdd(&activation[sliceSize * voxelZ + v], backscatteredLight);

	//float traversedDistance = float(t) * rtRecInfo._timeStep - rtRecInfo._timeOffset;
	//if (rtRecInfo._discardFirstLastBounces == 0)
	//	traversedDistance -= (glm::distance(lPos, rtRecInfo._laserPosition) + glm::distance(lPos, rtRecInfo._sensorPosition));

	//const float backscatteredLight = rtRecInfo.intensityCube[getConfocalTransientIndex(l, t)];
	//const float extension = 0.001f;
	//const float sigma = extension * glm::length(rtRecInfo._hiddenVolumeVoxelSize);
	//const float error = glm::abs(traversedDistance - glm::distance(lPos, checkPos));
	//const float weight = expf(-(error * error) / (2.0f * sigma * sigma));

	//if (weight > EPS && backscatteredLight > EPS)
	//	atomicAdd(&activation[voxelZ * sliceSize + v], backscatteredLight);

	//const float maxRadius = 0.5f * distance;
	//const float invSquaredDistance = safeRCP(distance * distance);
	//const float backscatteredLight = rtRecInfo.intensityCube[getConfocalTransientIndex(l, s, t)];
	//glm::vec3 minWorld = lPos - glm::vec3(maxRadius);
	//glm::vec3 maxWorld = lPos + glm::vec3(maxRadius);

	// Clamp to scene AABB
	//minWorld = glm::max(minWorld, rtRecInfo._hiddenVolumeMin);
	//maxWorld = glm::min(maxWorld, rtRecInfo._hiddenVolumeMax);

	//glm::ivec3 minVoxel = glm::floor((minWorld - rtRecInfo._hiddenVolumeMin) / rtRecInfo._hiddenVolumeVoxelSize);
	//glm::ivec3 maxVoxel = glm::ceil((maxWorld - rtRecInfo._hiddenVolumeMin) / rtRecInfo._hiddenVolumeVoxelSize);

	//// Clamp to valid voxel indices
	//minVoxel = glm::clamp(minVoxel, glm::ivec3(0), glm::ivec3(voxelResolution) - 1);
	//maxVoxel = glm::clamp(maxVoxel, glm::ivec3(0), glm::ivec3(voxelResolution) - 1);

	//for (int z = minVoxel.z; z <= maxVoxel.z; ++z)
	//	for (int y = minVoxel.y; y <= maxVoxel.y; ++y)
	//		for (int x = minVoxel.x; x <= maxVoxel.x; ++x) 
	//		{
	//			glm::vec3 voxelCenter = rtRecInfo._hiddenVolumeMin + (glm::vec3(x, y, z) + 0.5f) * rtRecInfo._hiddenVolumeVoxelSize;

	//			const float estimatedDistance = glm::distance(lPos, voxelCenter) + glm::distance(voxelCenter, sPos);
	//			const float error = estimatedDistance - distance;

	//			const float extension = .3f;
	//			const float sigma = extension * glm::length(rtRecInfo._hiddenVolumeVoxelSize);
	//			const float weight = expf(-(error * error) / (2.0f * sigma * sigma));

	//			if (weight > 0.01f)
	//			{
	//				int index = z * voxelResolution.y * voxelResolution.x + y * voxelResolution.x + x;
	//				atomicAdd(&activation[index], weight * backscatteredLight * invSquaredDistance);
	//			}
	//		}
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

