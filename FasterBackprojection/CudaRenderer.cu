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
    using clock = std::chrono::high_resolution_clock;
    static auto lastTime = clock::now();
    static int imageFrameCount = 0;             // only count frames with valid images
    static double fps = 0.0;

    assert(_streamingEngine != nullptr);

    ViewportSurface& viewportSurface = _streamingEngine->getViewportSurface();
    float4* drawTexture = viewportSurface.acquirePresentSurface();

    _quadShader->use();
    _quadShader->applyActiveSubroutines();

    if (drawTexture)
    {
        imageFrameCount++; // count only if we got an image

        glm::uint width = _streamingEngine->getImageWidth();
        glm::uint height = _streamingEngine->getImageHeight();

        writeImageInSurface<<<CudaHelper::getNumBlocks(width * height, 512), 512>>>(
            drawTexture, width, width * height, _viewportTexture.getSurfaceObject()
            );

        viewportSurface.presented();
    }

    // Measure FPS for frames with images
    auto now = clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();

    if (elapsed > 1000) // every ~1s
    {
        fps = imageFrameCount * 1000.0 / elapsed;
        spdlog::info("Rendered images FPS: {:.2f}", fps);

        imageFrameCount = 0;
        lastTime = now;
    }

    _viewportTexture.bind();
    _quadVAO->drawObject(Vao::TRIANGLE, GL_TRIANGLES, 6);
}


