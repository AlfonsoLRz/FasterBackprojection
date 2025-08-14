#pragma once

#include "ApplicationState.h"
#include "CudaRenderer.h"
#include "SceneContent.h"
#include "Singleton.h"
#include "Texture.h"

class RenderingShader;
class Vao;
class ViewportSurface;

class Renderer: public Singleton<Renderer>, public ResizeListener, public ScreenshotListener
{
	friend class Singleton<Renderer>;

private:
	ApplicationState	_appState;
	CudaRenderer		_cudaRenderer;
	bool				_changedWindowSize;
	SceneContent		_content;
	RenderingShader*	_cubeShading;
	glm::uvec2			_newSize;

private:
	static void bindTexture(GLuint textureID, const ShaderProgram* shader, const std::string& uniformName, unsigned offset);
	void updateWindowBuffers();

public:
	Renderer();
	virtual ~Renderer();

	void createCamera() const;
	void createModels();
	void prepareOpenGL(uint16_t width, uint16_t height);
	void render();
	void resizeEvent(uint16_t width, uint16_t height) override;
	void screenshotEvent(const ScreenshotEvent& event) override;

	ApplicationState* getApplicationState() { return &_appState; }
	Camera* getCamera() const { return _content._camera[_appState._selectedCamera].get(); }
	SceneContent* getSceneContent() { return &_content; }
};

