#pragma once

#include "Image.h"

class Texture;

class TransientImage : public Image<unsigned char>
{
protected:
	void normalizeImage(float* slice);

public:
	TransientImage(uint16_t width, uint16_t height);
	~TransientImage() override;

	void save(const std::string& filename, const glm::uvec4* slice);
	void save(const std::string& filename, float* slice, const glm::uvec2& size, glm::uint stride, glm::uint offset, bool normalize = false);
};

