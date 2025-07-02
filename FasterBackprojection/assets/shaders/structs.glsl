struct Vertex
{
	vec3	position;
	vec3	normal;
	vec2	uv;
};

struct Mesh
{
	vec3	diffuseColor;
	uint	startIndex;

	vec3	emissionColor;
	float	emissionStrength;

	vec3	specularColor;
	float	metallic;

	vec3    max;
	uint	length;

	vec3    min;
	float	smoothness;

	vec3	padding;
	int		diffuseTextureIndex;
};

struct Ray {
	vec3	origin;
	vec3	direction;
};

struct HitInfo {
	vec3	position;
	float	t;

	vec3	normal;
	uint 	materialIndex;

	int		hit;
	uint	triangleIndex;
};

struct Sphere
{
	vec3	center;
	float	radius;
};

struct BvhNode
{
	vec3	maxPoint;
	uint	prevIndex1;

	vec3	minPoint;
	uint	prevIndex2;

	uint	triangleIndex;
	uint	numTriangles;
	uint	meshIndex;
	uint	padding;
};