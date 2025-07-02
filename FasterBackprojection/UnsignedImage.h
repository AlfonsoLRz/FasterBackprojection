#pragma once

#include "Image.h"

class UnsignedImage: public Image<unsigned char>
{
public:
	explicit UnsignedImage(const std::string& filename);
	UnsignedImage(unsigned char* image, uint16_t width, uint16_t height, uint8_t depth);
	UnsignedImage(uint16_t width, uint16_t height, uint8_t depth);
	~UnsignedImage() override;
};

