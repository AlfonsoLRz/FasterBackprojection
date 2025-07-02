#pragma once

#include <cuda_gl_interop.h>

#include "CudaHelper.h"
#include "ShaderProgram.h"
#include "UnsignedImage.h"

class Texture
{
protected:
	const static GLint MIN_FILTER;
	const static GLint MAG_FILTER;
	const static GLint WRAP_S;
	const static GLint WRAP_T;
	const static GLint WRAP_R;

protected:
	GLuint		_id;
	glm::vec4	_color;

public:
	Texture(
		UnsignedImage* image, 
		GLint wrapS = WRAP_S, GLint wrapT = WRAP_T, 
		GLint minFilter = MIN_FILTER, GLint magFilter = MAG_FILTER);
	Texture(
		const float* image, int width, int height, int numChannels,
		GLint wrapS = WRAP_S, GLint wrapT = WRAP_T,
		GLint minFilter = MIN_FILTER, GLint magFilter = MAG_FILTER);
	Texture(const glm::vec4& color);
	virtual ~Texture();

	void applyTexture(const ShaderProgram* shader, const GLint id, const std::string& shaderVariable) const;

	glm::vec4 getColor() const { return _color; }
	GLuint getId() const { return _id; }
};