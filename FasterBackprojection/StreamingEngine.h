#pragma once

#include "ViewportSurface.h"

class StreamingEngine
{
protected:
	glm::uint			_imageWidth = 0, _imageHeight = 0;
	ViewportSurface		_viewportSurface;

public:
	StreamingEngine(const glm::uint imageWidth, const glm::uint imageHeight)
		: _imageWidth(imageWidth), _imageHeight(imageHeight)
	{
		_viewportSurface.init(_imageWidth, _imageHeight);
	}

	ViewportSurface& getViewportSurface() { return _viewportSurface; }
	glm::uint getImageWidth() const { return _imageWidth; }
	glm::uint getImageHeight() const { return _imageHeight; }
};
