#pragma once

#include "stdafx.h"

#define PI			glm::pi<float>()
#define INV_PI		0.31830988618379067f
#define TWO_PI		6.28318530717958648f
#define INV_TWO_PI	0.15915494309189533f
#define INV_4_PI	0.07957747154594766f
#define LIGHT_SPEED 299792458.0f
#define EPS			0.00001f

__forceinline  __device__ float square(float a)
{
	return a * a;
}

__forceinline  __device__ glm::uint square(glm::uint a)
{
	return a * a;
}

__forceinline__ __device__ float safeRCP(const float x)
{
	if (x > EPS || x < -EPS)
		return 1.0f / x;

	return x >= 0 ? FLT_MAX : -FLT_MAX;
}

__forceinline__ __device__ glm::vec3 safeRCP(const glm::vec3& x)
{
	return glm::vec3(
		safeRCP(x.x),
		safeRCP(x.y),
		safeRCP(x.z));
}