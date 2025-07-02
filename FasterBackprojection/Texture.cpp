#include "stdafx.h"
#include "Texture.h"

// [Static variables initialization]

const GLint Texture::MAG_FILTER = GL_LINEAR;
const GLint Texture::MIN_FILTER = GL_LINEAR_MIPMAP_NEAREST;
const GLint Texture::WRAP_S = GL_MIRRORED_REPEAT;
const GLint Texture::WRAP_T = GL_MIRRORED_REPEAT;
const GLint Texture::WRAP_R = GL_MIRRORED_REPEAT;

/// [Public methods]

Texture::Texture(UnsignedImage* image, GLint wrapS, GLint wrapT, GLint minFilter, const GLint magFilter): _color()
{
	unsigned char* bits = image->bits();
	const unsigned int width = image->getWidth(), height = image->getHeight();
	constexpr GLuint depthID[] = {GL_RED, GL_RED, GL_RGB, GL_RGBA};

	glGenTextures(1, &_id);
	glBindTexture(GL_TEXTURE_2D, _id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);

	glTexImage2D(GL_TEXTURE_2D, 0, depthID[image->getDepth() - 1], width, height, 0, depthID[image->getDepth() - 1],
	             GL_UNSIGNED_BYTE, bits);
	glGenerateMipmap(GL_TEXTURE_2D);
}

Texture::Texture(const float* image, int width, int height, int numChannels, GLint wrapS, GLint wrapT, GLint minFilter, GLint magFilter):
	_color()
{
	constexpr GLuint depthID[] = { GL_RED, GL_RED, GL_RGB, GL_RGBA };

	glGenTextures(1, &_id);
	glBindTexture(GL_TEXTURE_2D, _id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);

	glTexImage2D(GL_TEXTURE_2D, 0, depthID[numChannels - 1], width, height, 0, depthID[numChannels - 1], GL_FLOAT, image);

	glGenerateMipmap(GL_TEXTURE_2D);
}

Texture::Texture(const glm::vec4& color)
	: _id(-1), _color(color)
{
	constexpr int width = 1, height = 1;
	const unsigned char image[] = { 
		static_cast<unsigned char>(255.0f * color.x), static_cast<unsigned char>(255.0f * color.y), 
		static_cast<unsigned char>(255.0f * color.z), static_cast<unsigned char>(255.0f * color.a) 
	};

	glGenTextures(1, &_id);
	glBindTexture(GL_TEXTURE_2D, _id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, MIN_FILTER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, MAG_FILTER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, WRAP_S);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, WRAP_T);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
}

Texture::~Texture()
{
	glDeleteTextures(1, &_id);
}

void Texture::applyTexture(const ShaderProgram* shader, const GLint id, const std::string& shaderVariable) const
{
	assert(shader->setUniform(shaderVariable, id));
	glActiveTexture(GL_TEXTURE0 + id);
	glBindTexture(GL_TEXTURE_2D, _id);
}
