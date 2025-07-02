#pragma once

#include "stdafx.h"
#include "GpuStructs.cuh"

__forceinline__ __device__ glm::uint getConfocalTransientIndex(glm::uint laserTargetIdx, glm::uint timeBin)
{
	return timeBin * rtRecInfo._numLaserTargets + laserTargetIdx;
}

__forceinline__ __device__ glm::uint getExhaustiveTransientIndex(glm::uint laserTargetIdx, glm::uint sensorTargetIdx, glm::uint timeBin)
{
	return
		timeBin * rtRecInfo._numLaserTargets * rtRecInfo._numSensorTargets +
		laserTargetIdx * rtRecInfo._numSensorTargets +
		sensorTargetIdx;
}