#include "stdafx.h"
#include "UnsignedImage.h"

/// [Public methods]

UnsignedImage::UnsignedImage(const std::string& filename) : Image(0, 0, 4)
{
	int width, height, channels;

	if (unsigned char* image = SOIL_load_image(filename.c_str(), &width, &height, &channels, SOIL_LOAD_RGBA))
	{
		_width = width;
		_height = height;
		_depth = 4;
		const int size = _width * _height * _depth;
		_image = std::vector(image, image + size);
		SOIL_free_image_data(image);
	}
	else
	{
		_width = _height = _depth = 0;
		throw std::runtime_error("Failed to load image: " + filename);
	}
}

UnsignedImage::UnsignedImage(unsigned char* image, const uint16_t width, const uint16_t height, const uint8_t depth) :
	Image(width, height, depth)
{
	if (image)
	{
		const int size = width * height * depth;
		std::copy_n(image, size, _image.begin());
	}
	else
	{
		std::cout << "Empty image!\n";
		_width = _height = _depth = 0;
		_image.clear();
	}
}

UnsignedImage::UnsignedImage(uint16_t width, uint16_t height, uint8_t depth) : Image(width, height, depth)
{
}

UnsignedImage::~UnsignedImage() = default;
