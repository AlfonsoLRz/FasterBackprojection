#include "stdafx.h"
#include "AABB.h"

// Public methods

AABB::AABB(const glm::vec3& min, const glm::vec3& max) : _max(max), _min(min)
{
}

AABB::AABB(const AABB& aabb) 
= default;

AABB::~AABB()
= default;

AABB& AABB::operator=(const AABB& aabb)
= default;

AABB AABB::dot(const glm::mat4& matrix) const
{
	return AABB(matrix * glm::vec4(_min, 1.0f), matrix * glm::vec4(_max, 1.0f));
}

void AABB::update(const AABB& aabb)
{
	this->update(aabb.maxPoint());
	this->update(aabb.minPoint());
}

void AABB::update(const glm::vec3& point)
{
	_min.x = std::min(point.x, _min.x);
	_min.y = std::min(point.y, _min.y);
	_min.z = std::min(point.z, _min.z);

	_max.x = std::max(point.x, _max.x);
	_max.y = std::max(point.y, _max.y);
	_max.z = std::max(point.z, _max.z);
}

std::ostream& operator<<(std::ostream& os, const AABB& aabb)
{
	os << "Maximum corner: " << aabb.maxPoint().x << ", " << aabb.maxPoint().y << ", " << aabb.maxPoint().z << "\n";
	os << "Minimum corner: " << aabb.minPoint().x << ", " << aabb.minPoint().y << ", " << aabb.minPoint().z << "\n";

	return os;
}
