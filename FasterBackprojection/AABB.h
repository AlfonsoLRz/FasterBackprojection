#pragma once

class AABB
{
protected:
	glm::vec3	_max, _min;

public:
	explicit AABB(const glm::vec3& min = glm::vec3(FLT_MAX), const glm::vec3& max = glm::vec3(-FLT_MAX));
	AABB(const AABB& aabb);
	virtual ~AABB();
	AABB& operator=(const AABB& aabb);

	glm::vec3 center() const { return (_max + _min) / 2.0f; }
	AABB dot(const glm::mat4& matrix) const;
	glm::vec3 extent() const { return _max - center(); }
	glm::vec3 maxPoint() const { return _max; }
	glm::vec3 minPoint() const { return _min; }
	glm::vec3 size() const { return _max - _min; }

	void update(const AABB& aabb);
	void update(const glm::vec3& point);

	friend std::ostream& operator<<(std::ostream& os, const AABB& aabb);
};

