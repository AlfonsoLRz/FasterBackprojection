#pragma once

#include "stdafx.h"

#include "TransientParameters.h"

class ApplicationState
{
public:
	// Application
	glm::vec3						_backgroundColor;
	bool							_enableSky;
	uint16_t						_numFps;
	uint8_t							_selectedCamera;
	glm::ivec2						_viewportSize;
	bool							_primitiveEnabled[3]; // Point, Line, Triangle

	// Screenshot
	char							_screenshotFilenameBuffer[60];
	float							_screenshotFactor;
	bool							_transparentScreenshot;

	ApplicationState()
	{
		_backgroundColor = glm::vec3(.6f);
		_enableSky = true;
		_numFps = 0;
		_selectedCamera = 0;
		_viewportSize = glm::vec3(0);

		strcpy_s(_screenshotFilenameBuffer, 60, "ScreenshotRGBA.png");
		_screenshotFactor = 2.2f;
		_transparentScreenshot = true;

		for (bool& enabled : _primitiveEnabled)
			enabled = true; // Enable all primitives by default
	}
};