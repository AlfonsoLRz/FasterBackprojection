#include "stdafx.h"
#include "GpuStructs.cuh"

__constant__ __device__ ReconstructionInfo rtRecInfo;

__constant__ __device__ glm::vec3*	laserTargets;
__constant__ __device__ glm::vec3*	sensorTargets;
__constant__ __device__ float*		intensityCube;

// Noise
__constant__ __device__ float*		noiseBuffer;

// MIS
__constant__ __device__ float*		spatialSum;
__constant__ __device__ glm::uint*	aliasTable;
__constant__ __device__ float*		probTable;