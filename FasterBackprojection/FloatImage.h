#pragma once

#include "Image.h"

class Texture;

class FloatImage: public Image<float>
{
protected:
	Texture*	_texture;	

public:
	FloatImage(const std::string& filename);
	FloatImage(uint16_t width, uint16_t height, uint8_t depth);
	~FloatImage() override;

	void updateRenderingBuffers() const;
	void resize(uint16_t width, uint16_t height, uint8_t depth) override;
	void writeAt(uint16_t x, uint16_t y, const glm::vec4& color) override;

	// Converts the float image into unsigned bits. The main problem is we don't know the range of the float image, and
	// linear normalization may not be the best option. We can use the exposure and gamma parameters to modify the function.
	uchar4* toUnsignedBits(float exposure = 1.5f, float gamma = 1.8f) const;

	// ------- Getter --------

	Texture* getTexture() const { return _texture; }
};

