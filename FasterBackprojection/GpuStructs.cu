#include "stdafx.h"
#include "GpuStructs.cuh"

__constant__ __device__ ReconstructionInfo rtRecInfo;

__device__ glm::vec3* laserTargets;
__device__ glm::vec3* sensorTargets;
__device__ float* intensityCube;