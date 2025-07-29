#version 460

#extension GL_ARB_compute_variable_group_size : enable
#extension GL_NV_gpu_shader5 : enable
layout(local_size_variable) in;

#include <assets/shaders/compute/constants.glsl>

layout(std430, binding = 0) coherent buffer Activation		{ uint	activation[]; };
layout(std430, binding = 1) coherent buffer IntensityVolume	{ float intensityCube[]; };
layout(std430, binding = 2) coherent buffer LaserTargets		{ vec3	laserTargets[]; };

uniform uint	sliceSize;

uniform uvec3	voxelResolution;
uniform vec3	hiddenVolumeMin, hiddenVolumeVoxelSize;

uniform uint	numLaserTargets, numTimeBins;
uniform vec3	sensorPosition, laserPosition;

uniform	float	timeStep, timeOffset;
uniform uint	discardFirstLastBounces;	

uint getConfocalTransientIndex(uint laserTargetIdx, uint timeBin)
{
	return laserTargetIdx * numTimeBins + timeBin;
}

void main()
{
	uvec3 gid = gl_GlobalInvocationID;
	uint v = gid.x;
	uint l = gid.y;
	uint vx = gid.z;

	if (l >= numLaserTargets || v >= sliceSize || vx >= voxelResolution.x)
		return;

	const uint vz = v % voxelResolution.x;
	const uint vy = v / voxelResolution.x;

	const vec3 lPos = laserTargets[l];
	const vec3 checkPos = hiddenVolumeMin + vec3(vx, vz, vy) * hiddenVolumeVoxelSize + 0.5f * hiddenVolumeVoxelSize;

	float dist = distance(lPos, checkPos) * 2.0f - timeOffset;
	if (discardFirstLastBounces == 0)
		dist += distance(lPos, laserPosition) + distance(lPos, sensorPosition);

#ifdef ACCUMULATE_VOXEL_SCATTERING
	float voxelExtent = length(hiddenVolumeVoxelSize) / 2.0f;
	float minDist = max(.0f, dist - voxelExtent), maxDist = dist + voxelExtent;
	float d = minDist, sum = .0f;

	while (d < maxDist)
	{
		const int timeBin = static_cast<int>(d / timeStep);
		if (timeBin < 0 || timeBin >= numTimeBins)
			return;
		sum += intensityCube[getConfocalTransientIndex(l, timeBin)];
		d += timeStep + EPS;
	}

	if (sum > EPS)
		atomicAdd(&activation[sliceSize * voxelZ + v], sum);
#else
	const int timeBin = int(dist / timeStep);
	if (timeBin < 0 || timeBin >= numTimeBins)
		return;

	const uint backscatteredLight = uint(intensityCube[getConfocalTransientIndex(l, timeBin)] * 10000.0f);
	if (backscatteredLight > EPS)
		atomicAdd(activation[vx * sliceSize + vy * voxelResolution.z + vz], 1u);
#endif
}