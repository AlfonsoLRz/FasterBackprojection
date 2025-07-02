#pragma once

#include "Model3D.h"

class Sphere : public Model3D
{
private:
	glm::vec3	_center;
	float		_radius;

public:
	Sphere(const glm::vec3& center, const float radius);
	~Sphere() override;

	// Getters
	glm::vec3 getCenter() const { return _center; }
	float getRadius() const { return _radius; }
	virtual AABB getAABB() const override;
};

