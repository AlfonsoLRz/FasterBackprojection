#include "stdafx.h"
#include "Vao.h"

// Public methods

Vao::Vao(bool interleaved)
{
	glGenVertexArrays(1, &_vao);
	glBindVertexArray(_vao);

	// VBOs
	_vbo.resize(NUM_VBOS);
	glGenBuffers(static_cast<GLsizei>(_vbo.size()), _vbo.data());

	if (not interleaved)
	{
		Vao::createVBO(_vbo[VBO::POSITION], sizeof(glm::vec3), GL_FLOAT, VBO::POSITION);
		Vao::createVBO(_vbo[VBO::NORMAL], sizeof(glm::vec3), GL_FLOAT, VBO::NORMAL);
		Vao::createVBO(_vbo[VBO::TEXTURE_COORDS], sizeof(glm::vec2), GL_FLOAT, VBO::TEXTURE_COORDS);
	}
	else
	{
		Vao::createInterleavedVBO(_vbo[0]);
	}

	_ibo.resize(3);
	glGenBuffers(static_cast<GLsizei>(_ibo.size()), _ibo.data());
}

Vao::~Vao()
{
	glDeleteBuffers(static_cast<GLsizei>(_vbo.size()), _vbo.data());
	glDeleteBuffers(static_cast<GLsizei>(_ibo.size()), _ibo.data());
	glDeleteVertexArrays(1, &_vao);
}

void Vao::drawObject(IBO ibo, GLuint openGLPrimitive, GLsizei numIndices) const
{
	glBindVertexArray(_vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo[ibo]);
	glDrawElements(openGLPrimitive, numIndices, GL_UNSIGNED_INT, nullptr);
}

void Vao::drawObject(IBO ibo, GLuint openGLPrimitive, GLsizei numIndices, GLsizei numInstances) const
{
	glBindVertexArray(_vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo[ibo]);
	glDrawElementsInstanced(openGLPrimitive, numIndices, GL_UNSIGNED_INT, nullptr, numInstances);
}

void Vao::drawObject(IBO ibo, GLuint openGLPrimitive, GLsizei numIndices, GLsizei numInstances, GLsizei offset) const
{
	glBindVertexArray(_vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo[ibo]);
	glDrawElementsInstancedBaseInstance(openGLPrimitive, numIndices, GL_UNSIGNED_INT, nullptr, numInstances, offset);
}

void Vao::setVBOData(const std::vector<Vertex>& vertices, GLuint changeFrequency) const
{
	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), changeFrequency);
}

void Vao::setIBOData(IBO ibo, const std::vector<GLuint>& indices, GLuint changeFrequency) const
{
	glBindVertexArray(_vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo[ibo]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), changeFrequency);
}

// Private methods

void Vao::createVBO(GLuint vboId, GLsizeiptr structSize, GLuint elementType, uint8_t slot)
{
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glBufferData(GL_ARRAY_BUFFER, structSize, nullptr, GL_STATIC_DRAW);
	glVertexAttribPointer(slot, static_cast<GLsizei>(structSize / sizeof(elementType)), elementType, GL_FALSE, static_cast<GLsizei>(structSize), static_cast<GLubyte*>(nullptr));
	glEnableVertexAttribArray(slot);
}

void Vao::createInterleavedVBO(GLuint vboId)
{
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	GLsizei structSize = sizeof(Vertex);

	glEnableVertexAttribArray(POSITION);
	glVertexAttribPointer(POSITION, 3, GL_FLOAT, GL_FALSE, structSize, reinterpret_cast<GLubyte*>(offsetof(Vertex, _position)));

	glEnableVertexAttribArray(NORMAL);
	glVertexAttribPointer(NORMAL, 3, GL_FLOAT, GL_FALSE, structSize, reinterpret_cast<GLubyte*>(offsetof(Vertex, _normal)));

	glEnableVertexAttribArray(TEXTURE_COORDS);
	glVertexAttribPointer(TEXTURE_COORDS, 2, GL_FLOAT, GL_FALSE, structSize, reinterpret_cast<GLubyte*>(offsetof(Vertex, _texCoords)));
}