#include "stdafx.h"
#include "Sphere.h"

//

Sphere::Sphere(const glm::vec3& center, const float radius) : _center(center), _radius(radius)
{
	// Add a single component to store material properties
	_components.push_back({});
	_components.front()._aabb = AABB(center - glm::vec3(radius), center + glm::vec3(radius));
}

Sphere::~Sphere()
= default;

AABB Sphere::getAABB() const
{
	return AABB(_center - glm::vec3(_radius), _center + glm::vec3(_radius));
}
