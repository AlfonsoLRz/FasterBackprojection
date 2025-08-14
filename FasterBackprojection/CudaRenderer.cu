#include "stdafx.h"
#include "CudaRenderer.h"

#include "renderer.cuh"
#include "RenderingShader.h"
#include "Vao.h"
#include "ViewportSurface.h"

//

CudaRenderer::~CudaRenderer()
{
	delete _quadShader;
	delete _quadVAO;
}

void CudaRenderer::initialize(ApplicationState* applicationState)
{
	_applicationState = applicationState;

    _quadShader = new RenderingShader();
    _quadShader->createShaderProgram("assets/shaders/shading/quad");

    // Texture for compute shader rendering
    _viewportTexture.init(applicationState->_viewportSize.x, applicationState->_viewportSize.y);
    _viewportTexture.mapPersistently();

    // Surface
    const std::vector<glm::vec2> quadTextCoord{
        glm::vec2(0.0f, 0.0f),
        glm::vec2(1.0f, 0.0f),
        glm::vec2(0.0f, 1.0f),
        glm::vec2(1.0f, 1.0f)
    };
    const std::vector<GLuint> triangleMesh{ 0, 1, 2, 1, 3, 2 };

    _quadVAO = new Vao(false);
    _quadVAO->setVBOData(Vao::TEXTURE_COORDS, quadTextCoord.data(), quadTextCoord.size());
    _quadVAO->setIBOData(Vao::IBO::TRIANGLE, triangleMesh);
}

void CudaRenderer::render()
{
    // Compute-based rendering
    ViewportSurface* viewportSurface = ViewportSurface::getInstance();
    float4* drawTexture = viewportSurface->acquirePresentSurface();

    _quadShader->use();
    _quadShader->applyActiveSubroutines();

    if (drawTexture)
    {
		std::cout << "Rendering with compute shader..." << std::endl;

        glm::uvec2 viewportSize = _applicationState->_viewportSize;
        glm::uint width = viewportSize.x, height = viewportSize.y;

        writeImage<<<CudaHelper::getNumBlocks(width * height, 512), 512>>>(
            drawTexture, width, width * height, _viewportTexture.getSurfaceObject()
            );
        CudaHelper::synchronize("writeImage");
        viewportSurface->presented();
    }

    _viewportTexture.bind();
    _quadVAO->drawObject(Vao::TRIANGLE, GL_TRIANGLES, 6);
}
