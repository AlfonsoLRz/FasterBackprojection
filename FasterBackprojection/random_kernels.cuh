#pragma once

#include "stdafx.h"

__forceinline__ __device__ float getUniformRandom(State* __restrict__ state)
{
	state->_randomState = state->_randomState * 747796405u + 2891336453u;
	return rtBufferInfo._noiseBuffer[state->_randomState % rtBufferInfo._noiseBufferSize];
}

__forceinline__ __device__ glm::uint getUniformRandomUint(State* __restrict__ state, glm::uint resolution)
{
	glm::uint randomValue = glm::uint(getUniformRandom(state) * float(resolution));
	return randomValue % resolution;
}

__forceinline__ __device__ float getRandomNormalDistribution(State* __restrict__ state)
{
	float theta = 2.0f * glm::pi<float>() * getUniformRandom(state);
	float rho = sqrt(-2.0f * log(getUniformRandom(state)));
	return rho * cos(theta);
}

__forceinline__ __device__ glm::vec3 getRandomDirection(State* __restrict__ state)
{
	float x = getRandomNormalDistribution(state);
	float y = getRandomNormalDistribution(state);
	float z = getRandomNormalDistribution(state);
	return glm::normalize(glm::vec3(x, y, z));
}

__forceinline__ __device__ glm::vec3 getRandomHemisphereDirection(State* __restrict__ state, const glm::vec3& normal)
{
	glm::vec3 direction = getRandomDirection(state);
	return direction * glm::sign(dot(direction, normal));
}

__forceinline__ __device__ glm::vec2 getRandomPointInCircle(State* __restrict__ state)
{
	float angle = getUniformRandom(state) * 2.0f * glm::pi<float>();
	float radius = sqrt(getUniformRandom(state));
	return glm::vec2(radius * cos(angle), radius * sin(angle));
}