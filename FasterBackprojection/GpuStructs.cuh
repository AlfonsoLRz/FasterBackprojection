#pragma once

enum CaptureSystem : glm::uint
{
	Confocal = 0, Exhaustive = 1
};

struct __align__(16) ReconstructionInfo
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
	bool			_discardFirstLastBounces;

	glm::vec3		_hiddenVolumeMax;
	glm::vec3		_hiddenVolumeSize;
	glm::vec3		_hiddenVolumeVoxelSize;

	//glm::vec3		_relayWallMinPosition;
	//glm::vec3		_relayWallSize;
	//glm::vec3		_relayWallNormal;

	glm::uvec3		_voxelResolution;

	__host__ float getFocusDepth() const
	{
		return glm::abs((_hiddenVolumeMin[1] + _hiddenVolumeMax[1]) * 0.5f);
	}
};

struct ReconstructionBuffers
{
	glm::vec3* _laserTargets;
	glm::vec3* _sensorTargets;

	glm::vec3* _laserTargetsNormals;

	float* _intensity;
};

__constant__ extern __device__ ReconstructionInfo rtRecInfo;

// NLOS things
__constant__ extern __device__ glm::vec3* laserTargets;
__constant__ extern __device__ glm::vec3* sensorTargets;

__constant__ extern __device__ glm::vec3* laserTargetsNormals;

__constant__ extern __device__ float* intensityCube;

// Noise
__constant__ extern __device__ float* noiseBuffer;

// MIS
__constant__ extern __device__ float* spatialSum;
__constant__ extern __device__ glm::uint* aliasTable;
__constant__ extern __device__ float* probTable;