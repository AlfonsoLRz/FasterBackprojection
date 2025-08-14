#pragma once

#include "stdafx.h"
#include "Singleton.h"
#include "Texture.h"

class ViewportSurface : public Singleton<ViewportSurface>
{
	friend class Singleton<ViewportSurface>;

private:
    glm::uint                           _width = 0, _height = 0, _numSurfaces = 0;
    std::vector<TextureResourceGPU*>    _surfaces;
    std::vector<cudaEvent_t>            _surfaceEvents;
    std::queue<glm::uint>               _freeQueue, _presentQueue;
    glm::uint                           _drawIndex = 0, _presentIndex = 0;

private:
    ViewportSurface() = default;

public:
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
    TextureResourceGPU* acquireDrawSurface()
	{
        // If no free surface is available, wait for one to complete
        if (_freeQueue.empty())
            return nullptr;

        glm::uint idx = _freeQueue.front();
        _freeQueue.pop();

        _drawIndex = idx;
        return _surfaces[idx];
    }

    // Mark the draw surface as ready for presentation
    void present(cudaStream_t stream = nullptr)
	{
        _presentQueue.push(_drawIndex);
    }

    // Get the surface ready to display
    TextureResourceGPU* acquirePresentSurface()
	{
        if (_presentQueue.empty())
            return nullptr;

        _presentIndex = _presentQueue.front();
        _presentQueue.pop();

        // After presentation, surface becomes free again
        _freeQueue.push(_presentIndex);

        _surfaces[_presentIndex]->mapPersistently();
        return _surfaces[_presentIndex];
    }

private:
    void createSurfaces()
	{
        _surfaces.resize(_numSurfaces);
        _surfaceEvents.resize(_numSurfaces);

        for (glm::uint i = 0; i < _numSurfaces; ++i) 
        {
            _surfaces[i] = new TextureResourceGPU(_width, _height);
            _surfaces[i]->init();
            cudaEventCreateWithFlags(&_surfaceEvents[i], cudaEventDisableTiming);
            _freeQueue.push(i); // Initially all free
        }
    }

    void destroySurfaces()
	{
        for (glm::uint i = 0; i < _numSurfaces; ++i) 
        {
            delete _surfaces[i];
            cudaEventDestroy(_surfaceEvents[i]);
        }

        _surfaces.clear();
        _surfaceEvents.clear();
    }

    void waitForSurface()
	{
        // Wait for the oldest surface in use to finish
        glm::uint idx = (_drawIndex + 1) % _numSurfaces;
        cudaEventSynchronize(_surfaceEvents[idx]);
        _freeQueue.push(idx);
    }

    void waitForPresent()
	{
        // Block until something is presentable
        if (!_presentQueue.empty())
            return;

        // Wait for the last presented frame
        glm::uint idx = _drawIndex;
        cudaEventSynchronize(_surfaceEvents[idx]);
        _presentQueue.push(idx);
    }
};


