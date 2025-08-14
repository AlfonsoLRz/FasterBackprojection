#pragma once

class RenderingShader;
class Vao;

#include "ApplicationState.h"
#include "Texture.h"

class CudaRenderer
{
private:
	ApplicationState*	_applicationState = nullptr;
	RenderingShader*	_quadShader = nullptr;
	Vao*				_quadVAO = nullptr;
	TextureResourceGPU  _viewportTexture;

public:
	CudaRenderer() = default;
	~CudaRenderer();

	void initialize(ApplicationState* applicationState);
	void render();
};

