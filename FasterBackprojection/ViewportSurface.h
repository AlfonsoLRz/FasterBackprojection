#pragma once

#include "stdafx.h"
#include "Singleton.h"
#include "Texture.h"

class ViewportSurface 
{
private:
    glm::uint                   _width = 0, _height = 0, _numSurfaces = 0;
    std::vector<float4*>        _surfaces;
    std::queue<glm::uint>       _freeQueue, _presentQueue;
    glm::uint                   _drawIndex = 0, _presentIndex = 0;

public:
    ViewportSurface() = default;

    ~ViewportSurface()
	{
        destroySurfaces();
    }

    void init(
        glm::uint width,
        glm::uint height,
        glm::uint numSurfaces = 3)
    {
	    _width = width;
        _height = height;
        _numSurfaces = numSurfaces;
		createSurfaces();
    }

    // Get a surface to draw into (non-blocking)
    float4* acquireDrawSurface()
	{
        // If no free surface is available, wait for one to complete
        if (_freeQueue.empty())
            return nullptr;

        _drawIndex = _freeQueue.front();
        _freeQueue.pop();

        return _surfaces[_drawIndex];
    }

    // Mark the draw surface as ready for presentation
    void present()
	{
        _presentQueue.push(_drawIndex);
    }

    // Get the surface ready to display
    float4* acquirePresentSurface()
	{
        if (_presentQueue.empty())
            return nullptr;

        _presentIndex = _presentQueue.front();
        _presentQueue.pop();

        return _surfaces[_presentIndex];
    }

    void presented()
    {
        // After presentation, surface becomes free again
        _freeQueue.push(_presentIndex);
    }

private:
    void createSurfaces()
	{
        _surfaces.resize(_numSurfaces);

        for (glm::uint i = 0; i < _numSurfaces; ++i) 
        {
            _surfaces[i] = nullptr;
			CudaHelper::initializeBuffer(_surfaces[i], _width * _height);
            _freeQueue.push(i); 
        }
    }

    void destroySurfaces()
	{
        for (glm::uint i = 0; i < _numSurfaces; ++i) 
            delete _surfaces[i];

        _surfaces.clear();
    }
};


