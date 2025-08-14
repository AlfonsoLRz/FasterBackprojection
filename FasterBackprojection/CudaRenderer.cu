#include "stdafx.h"
#include "CudaRenderer.h"

#include "renderer.cuh"
#include "RenderingShader.h"
#include "StreamingEngine.h"
#include "Vao.h"

//

CudaRenderer::~CudaRenderer()
{
	delete _quadShader;
	delete _quadVAO;
}

void CudaRenderer::initialize()
{
    _quadShader = new RenderingShader();
    _quadShader->createShaderProgram("assets/shaders/shading/quad");

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

void CudaRenderer::setStreamingFocus(StreamingEngine* streamingEngine)
{
	_streamingEngine = streamingEngine;

    _viewportTexture.init(streamingEngine->getImageWidth(), streamingEngine->getImageHeight());
    _viewportTexture.mapPersistently();
}

void CudaRenderer::render()
{
	assert(_streamingEngine != nullptr);

    // Compute-based rendering
    ViewportSurface& viewportSurface = _streamingEngine->getViewportSurface();
    float4* drawTexture = viewportSurface.acquirePresentSurface();

    _quadShader->use();
    _quadShader->applyActiveSubroutines();

    if (drawTexture)
    {
        glm::uint width = _streamingEngine->getImageWidth(),
    			  height = _streamingEngine->getImageHeight();

        writeImage<<<CudaHelper::getNumBlocks(width * height, 512), 512>>>(
            drawTexture, width, width * height, _viewportTexture.getSurfaceObject()
            );

        CudaHelper::synchronize("writeImage");
        viewportSurface.presented();
    }

    _viewportTexture.bind();
    _quadVAO->drawObject(Vao::TRIANGLE, GL_TRIANGLES, 6);
}
