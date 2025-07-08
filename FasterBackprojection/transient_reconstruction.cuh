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
	glm::uint c = blockIdx.x * blockDim.x + threadIdx.x;
	glm::uint l = blockIdx.y * blockDim.y + threadIdx.y;
	glm::uint d = blockIdx.z * blockDim.z + threadIdx.z;

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
		atomicAdd(&activation[c * numDepths + d], intensity);
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

// MIS

__global__ void spatioTemporalPrefixSum(
	const float* __restrict__ temporalPrefixSum, float* __restrict__ spatialPrefixSum, 
	glm::uint numSpatialElements, glm::uint numTimeBins)
{
	glm::uint spatialIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (spatialIdx >= numSpatialElements)
		return;

	spatialPrefixSum[spatialIdx] = temporalPrefixSum[spatialIdx * numTimeBins + numTimeBins - 1];
}

__global__ void normalizeSpatioTemporalPrefixSum(
	float* __restrict__ spatioTemporalPrefixSum, glm::uint numTimeBins, glm::uint size)
{
	const glm::uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	glm::uint spatialIdx = idx / numTimeBins, temporalIdx = idx % numTimeBins;
	float minValue = spatioTemporalPrefixSum[spatialIdx * numTimeBins], maxValue = spatioTemporalPrefixSum[spatialIdx * numTimeBins + numTimeBins - 1];

	spatioTemporalPrefixSum[idx] = (spatioTemporalPrefixSum[idx] - minValue) * safeRCP(maxValue - minValue);
}

__global__ void backprojectConfocalVoxelMIS(
	const float* __restrict__ spatialSum, float* __restrict__ activation, const float* __restrict__ noise,
	const glm::uint* __restrict__ aliasTable, const float* __restrict__ probTable, 
	glm::uint sliceSize, glm::uint numSamples, glm::uint noiseBufferSize)
{
	glm::uint v = blockIdx.x * blockDim.x + threadIdx.x;
	glm::uint s = blockIdx.y * blockDim.y + threadIdx.y;
	glm::uint voxelZ = blockIdx.z * blockDim.z + threadIdx.z;

	if (s >= numSamples || v >= sliceSize || voxelZ >= rtRecInfo._voxelResolution.z)
		return;

	const glm::uint vx = v % rtRecInfo._voxelResolution.x;
	const glm::uint vy = v / rtRecInfo._voxelResolution.x;

	glm::uint randomState = sliceSize * voxelZ + v + s;

	// Randomize pdf selection
	glm::uint l;
	float pdf, globalPDF;
	if (getUniformRandom(noise, noiseBufferSize, randomState) < 0.5f)
	{
		l = getMISIndexAlias(randomState, aliasTable, probTable, rtRecInfo._numLaserTargets, noise, noiseBufferSize);
		pdf = l > 0 ? spatialSum[l] - spatialSum[l - 1] : spatialSum[l];
		globalPDF = pdf + 1.0f * safeRCP(static_cast<float>(rtRecInfo._numLaserTargets));
	}
	else
	{
		l = static_cast<glm::uint>(getUniformRandom(noise, noiseBufferSize, randomState) * static_cast<float>(rtRecInfo._numLaserTargets));
		pdf = 1.0f * safeRCP(static_cast<float>(rtRecInfo._numLaserTargets));
		globalPDF = pdf + (l > 0 ? spatialSum[l] - spatialSum[l - 1] : spatialSum[l]);
	}

	// Check distance to the laser target
	const glm::vec3 lPos = laserTargets[l];
	const glm::vec3 checkPos = rtRecInfo._hiddenVolumeMin + glm::vec3(vx, voxelZ, vy) * rtRecInfo._hiddenVolumeVoxelSize + 0.5f * rtRecInfo._hiddenVolumeVoxelSize;

	float dist = glm::distance(lPos, checkPos) * 2.0f - rtRecInfo._timeOffset;
	if (rtRecInfo._discardFirstLastBounces == 0)
		dist += glm::distance(lPos, rtRecInfo._laserPosition) + glm::distance(lPos, rtRecInfo._sensorPosition);

	const int timeBin = static_cast<int>(dist / rtRecInfo._timeStep);
	if (timeBin < 0 || timeBin >= rtRecInfo._numTimeBins)
		return;

	// MIS
	const float misWeight = pdf * safeRCP(globalPDF);
	const float backscatteredLight = intensityCube[getConfocalTransientIndex(l, timeBin)];
	if (backscatteredLight > EPS)
		atomicAdd(&activation[sliceSize * voxelZ + v], misWeight * backscatteredLight * safeRCP(dist * dist));
}