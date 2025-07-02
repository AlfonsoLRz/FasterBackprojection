#pragma once

#define RESTART_PRIMITIVE_INDEX 0xFFFFFFFF

class Vao
{
public:
	enum VBO
	{
		POSITION = 0, NORMAL = 1, TEXTURE_COORDS = 2,
		TRANSLATION = 3, SCALE = 4, ROTATION = 5, INTENSITY = 6,
		NUM_VBOS = 7
	};

	enum IBO 
	{
		POINT = 0,
		LINE = 1,
		TRIANGLE = 2
	};

	struct Vertex
	{
		glm::vec3	_position;
		glm::vec3	_normal;
		glm::vec2	_texCoords;
	};

public:
	GLuint				_vao;
	std::vector<GLuint> _vbo;
	std::vector<GLuint> _ibo;

private:
	static void createInterleavedVBO(GLuint vboId);
	static void createVBO(GLuint vboId, GLsizeiptr structSize, GLuint elementType, uint8_t slot);

public:
	Vao(bool interleaved);
	virtual ~Vao();

	void drawObject(IBO ibo, GLuint openGLPrimitive, GLsizei numIndices) const;
	void drawObject(IBO ibo, GLuint openGLPrimitive, GLsizei numIndices, GLsizei numInstances) const;
	void drawObject(IBO ibo, GLuint openGLPrimitive, GLsizei numIndices, GLsizei numInstances, GLsizei offset) const;
	template<typename T, typename Z>
	int createMultiInstanceVBO(VBO vbo, T dataExample, Z dataPrimitive, GLuint openGLBasicType);
	template<typename T, typename Z>
	int createMultiInstanceIntVBO(VBO vbo, T dataExample, Z dataPrimitive, GLuint openGLBasicType);

	template<typename T>
	void setVBOData(VBO vbo, T* geometryData, GLsizei size, GLuint changeFrequency = GL_STATIC_DRAW);
	void setVBOData(const std::vector<Vertex>& vertices, GLuint changeFrequency = GL_STATIC_DRAW) const;
	void setIBOData(IBO ibo, const std::vector<GLuint>& indices, GLuint changeFrequency = GL_STATIC_DRAW) const;
};

template<typename T, typename Z>
inline int Vao::createMultiInstanceVBO(VBO vbo, T dataExample, Z dataPrimitive, GLuint openGLBasicType)
{
	glBindVertexArray(_vao);
	glGenBuffers(1, &_vbo[vbo]);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo[vbo]);
	glEnableVertexAttribArray(vbo);
	glVertexAttribPointer(vbo, sizeof(dataExample) / sizeof(dataPrimitive), openGLBasicType, GL_FALSE, sizeof(dataExample), (GLubyte*)nullptr);
	glVertexAttribDivisor(vbo, 1);

	return vbo;
}

template <typename T, typename Z>
int Vao::createMultiInstanceIntVBO(VBO vbo, T dataExample, Z dataPrimitive, GLuint openGLBasicType)
{
	glBindVertexArray(_vao);
	glGenBuffers(1, &_vbo[vbo]);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo[vbo]);
	glEnableVertexAttribArray(vbo);
	glVertexAttribIPointer(vbo, sizeof(dataExample) / sizeof(dataPrimitive), openGLBasicType, sizeof(dataExample), (GLubyte*)nullptr);
	glVertexAttribDivisor(vbo, 1);

	return vbo;
}

template<typename T>
inline void Vao::setVBOData(VBO vbo, T* geometryData, GLsizei size, GLuint changeFrequency)
{
	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo[vbo]);
	glBufferData(GL_ARRAY_BUFFER, size * sizeof(T), geometryData, changeFrequency);
}
