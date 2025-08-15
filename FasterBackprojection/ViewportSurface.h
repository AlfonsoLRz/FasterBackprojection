#pragma once

#include "SafeQueue.h"
#include "stdafx.h"
#include "Texture.h"

class ViewportSurface 
{
private:
    glm::uint                   _width = 0, _height = 0, _numSurfaces = 0;
    std::vector<float4*>        _surfaces;
    SafeQueue<glm::uint, 3>     _freeQueue, _presentQueue;
    glm::uint                   _drawIndex = 0, _presentIndex = 0;

public:
    ViewportSurface() = default;

    ~ViewportSurface()
	{
        destroySurfaces();
    }

    void init(
        glm::uint width,
        glm::uint height)
    {
	    _width = width;
        _height = height;
        _numSurfaces = 3;
		createSurfaces();
    }

    // Get a surface to draw into (non-blocking)
    float4* acquireDrawSurface()
	{
        // If no free surface is available, wait for one to complete
        if (!_freeQueue.Size())
            return nullptr;

        _freeQueue.Pop(_drawIndex);

        return _surfaces[_drawIndex];
    }

    // Mark the draw surface as ready for presentation
    void present()
	{
        _presentQueue.Push(_drawIndex);
		//spdlog::info("ViewportSurface: Presenting surface {}", _drawIndex);
    }

    // Get the surface ready to display
    float4* acquirePresentSurface()
	{
        if (!_presentQueue.Size())
            return nullptr;

        _presentQueue.Pop(_presentIndex);

        return _surfaces[_presentIndex];
    }

    void presented()
    {
        // After presentation, surface becomes free again
        _freeQueue.Push(_presentIndex);
    }

private:
    void createSurfaces()
	{
        _surfaces.resize(_numSurfaces);

        for (glm::uint i = 0; i < _numSurfaces; ++i) 
        {
            _surfaces[i] = nullptr;
			CudaHelper::initializeBuffer(_surfaces[i], _width * _height, static_cast<float4*>(nullptr), false);
            _freeQueue.Push(i); 
        }
    }

    void destroySurfaces()
	{
        for (glm::uint i = 0; i < _numSurfaces; ++i) 
            delete _surfaces[i];

        _surfaces.clear();
    }
};


