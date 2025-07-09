#pragma once

enum CaptureSystem : glm::uint
{
	Confocal = 0, Exhaustive = 1
};

struct ReconstructionInfo
{
	glm::vec3		_sensorPosition;
	CaptureSystem	_captureSystem;

	glm::vec3		_laserPosition;
	float			_timeOffset;

	glm::uint 		_numLaserTargets;
	glm::uint		_numSensorTargets;
	glm::uint		_numTimeBins;
	float			_timeStep;

	glm::vec3		_hiddenVolumeMin;
	glm::uint		_discardFirstLastBounces;

	glm::vec3		_hiddenVolumeMax;
	glm::vec3		_hiddenVolumeSize;
	glm::vec3		_hiddenVolumeVoxelSize;

	glm::vec3		_relayWallMinPosition;
	glm::vec3		_relayWallSize;
	glm::vec3		_relayWallNormal;

	glm::uvec3		_voxelResolution;

	__host__ float getFocusDepth() const
	{
		return glm::abs(_relayWallMinPosition[1] - _hiddenVolumeMin[1]);
	}
};

struct ReconstructionBuffers
{
	glm::vec3* _laserTargets;
	glm::vec3* _sensorTargets;
	float* _intensity;
};

__constant__ extern __device__ ReconstructionInfo rtRecInfo;

// NLOS things
extern __device__ glm::vec3* laserTargets;
extern __device__ glm::vec3* sensorTargets;
extern __device__ float* intensityCube;

// Noise
extern __device__ float* noiseBuffer;

// MIS
extern __device__ float* spatialSum;
extern __device__ glm::uint* aliasTable;
extern __device__ float* probTable;