#include "stdafx.h"
#include "TransientImage.h"

#include "Texture.h"

TransientImage::TransientImage(uint16_t width, uint16_t height) : Image(width, height, 4)
{
}

TransientImage::~TransientImage() = default;

void TransientImage::save(const std::string& filename, const glm::uvec4* slice)
{
	#pragma omp parallel for
	for (int idx = 0; idx < _width * _height; ++idx)
	{
		_image[idx * 4 + 0] = static_cast<unsigned char>(glm::clamp(slice[idx].x, 0u, 255u));
		_image[idx * 4 + 1] = static_cast<unsigned char>(glm::clamp(slice[idx].y, 0u, 255u));
		_image[idx * 4 + 2] = static_cast<unsigned char>(glm::clamp(slice[idx].z, 0u, 255u));
		_image[idx * 4 + 3] = 255;
	}

	flipImageVertically(_image, _width, _height, _depth);
	SOIL_save_image(filename.c_str(), SOIL_SAVE_TYPE_PNG, _width, _height, _depth, _image.data());
}

void TransientImage::save(const std::string& filename, float* slice, const glm::uvec2& size, glm::uint stride, glm::uint offset, bool normalize, bool flip)
{
	if (normalize)
		normalizeImage(slice);

	#pragma omp parallel for
	for (int idx = 0; idx < _width * _height; ++idx)
	{
		_image[idx * 4 + 0] = _image[idx * 4 + 1] = _image[idx * 4 + 2] = 
			static_cast<unsigned char>(glm::clamp(static_cast<glm::uint>(slice[offset + idx * stride] * 255.0f), 0u, 255u));
		_image[idx * 4 + 3] = 255;
	}

	if (flip)
		flipImageVertically(_image, _width, _height, _depth);

	if (size != glm::uvec2(_width, _height))
		resize(size.x, size.y, Interpolation::BILINEAR);
	SOIL_save_image(filename.c_str(), SOIL_SAVE_TYPE_PNG, _width, _height, _depth, _image.data());
}

void TransientImage::normalizeImage(float* slice)
{
	float maxVal = .0f;
	const glm::uint size = this->getWidth() * this->getHeight();

	#pragma omp parallel for reduction(max:maxVal)
	for (glm::uint idx = 0; idx < size; ++idx)
		maxVal = glm::max(maxVal, slice[idx]);

	#pragma omp parallel for
	for (glm::uint idx = 0; idx < size; ++idx)
		slice[idx] /= maxVal;
}