#pragma once

class RenderingShader;
class StreamingEngine;
class Vao;

#include "Texture.h"

class CudaRenderer
{
private:
	// Cuda & OpenGL resources for interop
	RenderingShader*	_quadShader = nullptr;
	Vao*				_quadVAO = nullptr;
	TextureResourceGPU  _viewportTexture;

	// Streaming engine
	StreamingEngine*	_streamingEngine = nullptr;

public:
	CudaRenderer() = default;
	~CudaRenderer();

	void initialize();
	void setStreamingFocus(StreamingEngine* streamingEngine);
	void render();
};

