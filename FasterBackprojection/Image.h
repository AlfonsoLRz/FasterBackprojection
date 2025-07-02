#pragma once

#include "stdafx.h"

template<class T>
class Image
{
public:
	enum Interpolation
	{
		NEAREST,
		BILINEAR
	};

protected:
	std::vector<T>	_image;				//!< Image pixels
	uint16_t		_width, _height;
	uint8_t			_depth;				//!< Image dimensions

private:
	T interpolate(float x, float y, uint8_t depth) const;

public:
	Image(uint16_t width, uint16_t height, uint8_t depth);
	virtual ~Image();

	T* bits() { return _image.data(); }
	void flipImageVertically();
	void reset(T value = 0) { std::fill(_image.begin(), _image.end(), value); }
	virtual void resize(uint16_t width, uint16_t height, uint8_t depth);
	virtual void resize(uint16_t width, uint16_t height, Interpolation interpolation = BILINEAR);

	virtual glm::vec4 readAt(uint16_t x, uint16_t y) const;
	virtual void writeAt(uint16_t x, uint16_t y, const glm::vec4& color);

	// Getters
	uint16_t getDepth() const { return _depth; }
	uint16_t getHeight() const { return _height; }
	uint16_t getWidth() const { return _width; }

	// ----------- Static methods ------------
	static void flipImageVertically(std::vector<T>& image, const uint16_t width, const uint16_t height, const uint8_t depth);
};

template <class T>
T Image<T>::interpolate(float x, float y, uint8_t depth) const
{
	const int x0 = static_cast<int>(x);
	const int y0 = static_cast<int>(y);
	const int x1 = std::min(x0 + 1, _width - 1);
	const int y1 = std::min(y0 + 1, _height - 1);

	const float dx = x - static_cast<float>(x0);
	const float dy = y - static_cast<float>(y0);

	const T& p00 = _image[y0 * _width * _depth + x0 * _depth + depth];
	const T& p01 = _image[y0 * _width * _depth + x1 * _depth + depth];
	const T& p10 = _image[y1 * _width * _depth + x0 * _depth + depth];
	const T& p11 = _image[y1 * _width * _depth + x1 * _depth + depth];

	return static_cast<T>(
		(1.0f - dx) * (1.0f - dy) * p00 +
		dx * (1.0f - dy) * p01 +
		(1.0f - dx) * dy * p10 +
		dx * dy * p11);
}

template<class T>
Image<T>::Image(uint16_t width, uint16_t height, uint8_t depth) :
	_width(width), _height(height), _depth(depth)
{
	_image.resize(_width * _height * _depth);
	std::fill(_image.begin(), _image.end(), 0); // Initialize with zero
}

template<class T>
inline Image<T>::~Image() = default;

template<class T>
inline void Image<T>::flipImageVertically()
{
	Image::flipImageVertically(_image, _width, _height, _depth);
}

template<class T>
inline void Image<T>::resize(uint16_t width, uint16_t height, uint8_t depth)
{
	_width = width; _height = height; _depth = depth;
	_image.resize(_width * _height * _depth);
}

template <class T>
void Image<T>::resize(uint16_t width, uint16_t height, Interpolation interpolation)
{
	std::vector<T> newImage(width * height * _depth);

	const float xRatio = static_cast<float>(_width - 1) / static_cast<float>(width - 1);
	const float yRatio = static_cast<float>(_height - 1) / static_cast<float>(height - 1);

	if (interpolation == NEAREST)
	{
		#pragma omp parallel for
		for (int y = 0; y < static_cast<int>(height); ++y)
		{
			for (int x = 0; x < static_cast<int>(width); ++x)
			{
				const float xSample = x * xRatio;
				const float ySample = y * yRatio;
				const int xNearest = lround(xSample);
				const int yNearest = lround(ySample);

				#pragma omp parallel for
				for (int d = 0; d < _depth; ++d)
					newImage[y * width * _depth + x * _depth + d] = _image[yNearest * _width * _depth + xNearest * _depth + d];
			}
		}
	}
	else if (interpolation == BILINEAR)
	{
		#pragma omp parallel for
		for (int y = 0; y < static_cast<int>(height); ++y)
		{
			for (int x = 0; x < static_cast<int>(width); ++x)
			{
				const float xSample = x * xRatio;
				const float ySample = y * yRatio;

				#pragma omp parallel for
				for (int d = 0; d < _depth; ++d)
					newImage[y * width * _depth + x * _depth + d] = interpolate(xSample, ySample, d);
			}
		}
	}

	_image = std::move(newImage);
	_width = width;
	_height = height;
}

template <class T>
glm::vec4 Image<T>::readAt(uint16_t x, uint16_t y) const
{
	return {
		_image[y * _width * _depth + x * _depth + 0],
		_image[y * _width * _depth + x * _depth + 1],
		_image[y * _width * _depth + x * _depth + 2],
		_image[y * _width * _depth + x * _depth + 3]
	};
}

template <class T>
void Image<T>::writeAt(uint16_t x, uint16_t y, const glm::vec4& color)
{
	_image[y * _width * _depth + x * _depth + 0] = static_cast<T>(color.r * 255);
	_image[y * _width * _depth + x * _depth + 1] = static_cast<T>(color.g * 255);
	_image[y * _width * _depth + x * _depth + 2] = static_cast<T>(color.b * 255);
	_image[y * _width * _depth + x * _depth + 3] = static_cast<T>(color.a * 255);
}

template<class T>
inline void Image<T>::flipImageVertically(std::vector<T>& image, const uint16_t width, const uint16_t height, const uint8_t depth)
{
	int rowSize = width * depth;
	T* bits = image.data();
	T* tempBuffer = new T[rowSize];

	for (int i = 0; i < (height / 2); ++i)
	{
		T* source = bits + i * rowSize;
		T* target = bits + (height - i - 1) * rowSize;

		memcpy(tempBuffer, source, rowSize);					// Swap with help of temporary buffer
		memcpy(source, target, rowSize);
		memcpy(target, tempBuffer, rowSize);
	}

	delete[] tempBuffer;
}
