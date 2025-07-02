#include "stdafx.h"
#include "FloatImage.h"

#include "stb_image.h"
#include "Texture.h"

FloatImage::FloatImage(const std::string& filename): Image(0, 0, 0), _texture(nullptr)
{
	int width, height, channels;
	stbi_set_flip_vertically_on_load(true);
	float* image = stbi_loadf(filename.c_str(), &width, &height, &channels, 0);

	if (!image)
	{
		std::cout << "Error loading image: " << filename << '\n';
		_width = _height = _depth = 0;
		return;
	}

	_width = static_cast<uint16_t>(width);
	_height = static_cast<uint16_t>(height);
	_depth = static_cast<uint8_t>(channels);
	_image = std::vector(image, image + _width * _height * _depth);

	std::copy_n(image, _width * _height * _depth, _image.begin());
	stbi_image_free(image);

	_texture = new Texture(_image.data(), _width, _height, _depth);
}

FloatImage::FloatImage(const uint16_t width, const uint16_t height, const uint8_t depth) : Image(width, height, depth), _texture(nullptr)
{
	_texture = new Texture(_image.data(), width, height, depth, GL_CLAMP, GL_CLAMP, GL_NEAREST, GL_NEAREST);
}

FloatImage::~FloatImage()
{
	delete _texture;
}

void FloatImage::updateRenderingBuffers() const
{
	// Update texture
	const GLuint textureID = _texture->getId();
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_FLOAT, _image.data());
	glGenerateMipmap(GL_TEXTURE_2D);
}

void FloatImage::resize(uint16_t width, uint16_t height, uint8_t depth)
{
	Image::resize(width, height, depth);
	
	this->updateRenderingBuffers();
}

void FloatImage::writeAt(uint16_t x, uint16_t y, const glm::vec4& color)
{
	_image[y * _width * _depth + x * _depth + 0] = color.r;
	_image[y * _width * _depth + x * _depth + 1] = color.g;
	_image[y * _width * _depth + x * _depth + 2] = color.b;
	_image[y * _width * _depth + x * _depth + 3] = color.a;
}

uchar4* FloatImage::toUnsignedBits(float exposure, float gamma) const
{
	uchar4* image = new uchar4[_width * _height];
	float max = 1.0f;

	for (uint16_t y = 0; y < _height; ++y)
		for (uint16_t x = 0; x < _width; ++x)
			for (uint16_t z = 0; z < _depth; ++z)
				max = std::max(max, _image[y * _width * _depth + x * _depth + z]);

	for (uint16_t y = 0; y < _height; ++y)
	{
		unsigned char color[4] {};

		for (uint16_t x = 0; x < _width; ++x)
		{
			color[3] = 255;
			for (uint16_t z = 0; z < _depth; ++z)
			{
				color[z] = static_cast<unsigned char>(
					1.0f - exp(-glm::pow(_image[y * _width * _depth + x * _depth + z] / max, 1.0f / gamma) * exposure)
					* 255.0f
					);
			}

			image[y * _width + x] = *(reinterpret_cast<uchar4*>(color));
		}
	}

	return image;
}
