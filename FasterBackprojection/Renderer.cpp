// ReSharper disable CppExpressionWithoutSideEffects
#include "stdafx.h"
#include "Renderer.h"

#include "ApplicationState.h"
#include "Camera.h"
#include "Model3D.h"
#include "Vao.h"

//

Renderer::Renderer():
	_changedWindowSize(false),
	_cubeShading(nullptr),
	_newSize(0),
	_quadShader(nullptr),
	_quadVAO(nullptr)
{
}

Renderer::~Renderer()
{
	delete _quadShader;
	delete _quadVAO;
}

void Renderer::createCamera() const
{
    if (!_content._model.empty())
    {
        //_content._camera[_appState._selectedCamera]->track(_content._model.front().get());
        _content._camera[_appState._selectedCamera]->saveCamera();
    }
}

void Renderer::createModels()
{
    _content.buildScenario();
}

void Renderer::prepareOpenGL(uint16_t width, uint16_t height)
{
    _appState._viewportSize = glm::ivec2(width, height);

    //
    glClearColor(_appState._backgroundColor.x, _appState._backgroundColor.y, _appState._backgroundColor.z, 1.0f);

    //
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_MULTISAMPLE);

    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(RESTART_PRIMITIVE_INDEX);

    glEnable(GL_PROGRAM_POINT_SIZE);

    glEnable(GL_POLYGON_OFFSET_FILL);

    _content._camera.push_back(std::make_unique<Camera>(width, height));
    this->createModels();
    this->createCamera();

    // Observer
    InputManager* inputManager = InputManager::getInstance();
    inputManager->subscribeResize(this);
    inputManager->subscribeScreenshot(this);

	_cubeShading = new RenderingShader();
	_cubeShading->createShaderProgram("assets/shaders/shading/cube");

	_quadShader = new RenderingShader();
    _quadShader->createShaderProgram("assets/shaders/shading/quad");

    // Vao
    const std::vector<glm::vec2> quadTextCoord{
        glm::vec2(0.0f, 0.0f),
        glm::vec2(1.0f, 0.0f),
        glm::vec2(0.0f, 1.0f),
        glm::vec2(1.0f, 1.0f)
    };				
    const std::vector<GLuint> triangleMesh{ 0, 1, 2, 1, 3, 2 };				

    _quadVAO = new Vao(false);
    //_quadVAO->setVBOData(Vao::TEXTURE_COORDS, quadTextCoord.data(), quadTextCoord.size());
    //_quadVAO->setIBOData(Vao::IBO::TRIANGLE, triangleMesh);

    this->resizeEvent(_appState._viewportSize.x, _appState._viewportSize.y);
}

void Renderer::render()
{
    if (_changedWindowSize)
    {
        _appState._viewportSize = _newSize;
        updateWindowBuffers();
        _changedWindowSize = false;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(_appState._backgroundColor.x, _appState._backgroundColor.y, _appState._backgroundColor.z, 1.0f);

    glPolygonOffset(1.0f, 1.0f);

    Model3D::MatrixRenderInformation matrixInformation;
    matrixInformation.setMatrix(Model3D::MatrixRenderInformation::MODEL, 
        glm::rotate(glm::mat4(1.0f), glm::pi<float>() / 2.0f, glm::vec3(.0f, .0f, 1.0f)) * 
        glm::scale(glm::mat4(1.0f), glm::vec3(1.0f)));
    matrixInformation.setMatrix(Model3D::MatrixRenderInformation::MODEL, glm::scale(glm::mat4(1.0f), glm::vec3(10.0f)));
    matrixInformation.setMatrix(Model3D::MatrixRenderInformation::VIEW, _content._camera.front()->getViewMatrix());
    matrixInformation.setMatrix(Model3D::MatrixRenderInformation::VIEW_PROJECTION, _content._camera.front()->getViewProjectionMatrix());

    for (auto& model : _content._model)
    {
        model->draw(&matrixInformation, &_appState);
    }

    glPolygonOffset(.0f, .0f);

    //_quadShader->use();
    //_quadShader->applyActiveSubroutines();

    //_quadVAO->drawObject(Vao::TRIANGLE, GL_TRIANGLES, 6);
}

void Renderer::resizeEvent(uint16_t width, uint16_t height)
{
    if (_content._camera.front())
        _content._camera.front()->setRaspect(width, height);

    glViewport(0, 0, width, height);

	_changedWindowSize = true;
    _newSize = glm::ivec2(width, height);
}

void Renderer::screenshotEvent(const ScreenshotEvent& event)
{
}

//

void Renderer::bindTexture(GLuint textureID, const ShaderProgram* shader, const std::string& uniformName, unsigned offset)
{
    assert(shader->setUniform(uniformName, offset));
    glActiveTexture(GL_TEXTURE0 + offset);
    glBindTexture(GL_TEXTURE_2D, textureID);
}

void Renderer::updateWindowBuffers()
{
}
