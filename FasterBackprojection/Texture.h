#pragma once

#include <cuda_gl_interop.h>

#include "CudaHelper.h"
#include "ShaderProgram.h"
#include "UnsignedImage.h"

class Texture
{
protected:
	const static GLint MIN_FILTER;
	const static GLint MAG_FILTER;
	const static GLint WRAP_S;
	const static GLint WRAP_T;
	const static GLint WRAP_R;

protected:
	GLuint		_id;
	glm::vec4	_color;

public:
	Texture(
		UnsignedImage* image, 
		GLint wrapS = WRAP_S, GLint wrapT = WRAP_T, 
		GLint minFilter = MIN_FILTER, GLint magFilter = MAG_FILTER);
	Texture(
		const float* image, int width, int height, int numChannels,
		GLint wrapS = WRAP_S, GLint wrapT = WRAP_T,
		GLint minFilter = MIN_FILTER, GLint magFilter = MAG_FILTER);
	Texture(const glm::vec4& color);
	virtual ~Texture();

	void applyTexture(const ShaderProgram* shader, const GLint id, const std::string& shaderVariable) const;

	glm::vec4 getColor() const { return _color; }
	GLuint getId() const { return _id; }
};

class TextureResourceGPU
{
public:
    GLuint                  _id = 0;
    cudaGraphicsResource*   _cudaResource = nullptr;
    cudaSurfaceObject_t     _surfaceObj = 0;
    GLuint                  _width, _height;
    bool                    _isMapped = false;

public:
    TextureResourceGPU(GLuint width, GLuint height) : _width(width), _height(height) {}
    ~TextureResourceGPU() { release(); }

    void init(unsigned int flags = cudaGraphicsRegisterFlagsWriteDiscard)
    {
        glGenTextures(1, &_id);
        glBindTexture(GL_TEXTURE_2D, _id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _width, _height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Direct texture registration (optimal)
        CudaHelper::checkError(cudaGraphicsGLRegisterImage(&_cudaResource, _id, GL_TEXTURE_2D, flags | cudaGraphicsRegisterFlagsSurfaceLoadStore));
    }

    // Persistent mapping interface
    void mapPersistently()
    {
        if (!_isMapped)
        {
            CudaHelper::checkError(cudaGraphicsMapResources(1, &_cudaResource, 0));
            createSurfaceObject();
            _isMapped = true;
        }
    }

    void unmapPersistently()
    {
        if (_isMapped)
        {
            cudaGraphicsUnmapResources(1, &_cudaResource, 0);
            _isMapped = false;
        }
    }

    // For backward compatibility
    float4* mapCudaPointer()
	{
        if (!_isMapped)
            mapPersistently();

        return nullptr;
    }

    void unmapCudaPointer()
    {
        if (_surfaceObj) {
            cudaDestroySurfaceObject(_surfaceObj);
            _surfaceObj = 0;
        }
    }

    // Surface object access (for direct texture writes)
    cudaSurfaceObject_t getSurfaceObject()
    {
        if (!_surfaceObj)
            return _surfaceObj; // already created by mapPersistently

        if (!_surfaceObj)
        {
            if (!_isMapped)
                mapPersistently();
            createSurfaceObject();
        }

        return _surfaceObj;
    }

    void release()
    {
        unmapPersistently();

        if (_surfaceObj)
        {
            cudaDestroySurfaceObject(_surfaceObj);
            _surfaceObj = 0;
        }

        if (_cudaResource)
        {
            cudaGraphicsUnregisterResource(_cudaResource);
            _cudaResource = nullptr;
        }

        if (_id)
            glDeleteTextures(1, &_id);
    }

    void bind(GLenum unit = GL_TEXTURE0) const
    {
        glActiveTexture(unit);
        glBindTexture(GL_TEXTURE_2D, _id);
    }

    void resize(GLuint width, GLuint height)
    {
        if (width == _width && height == _height) return;

        release();
        _width = width;
        _height = height;
        init();
    }

private:
    void createSurfaceObject()
    {
        if (_surfaceObj) return;

        cudaArray* array = nullptr;
        CudaHelper::checkError(cudaGraphicsSubResourceGetMappedArray(
            &array, _cudaResource, 0, 0));

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = array;
        CudaHelper::checkError(cudaCreateSurfaceObject(&_surfaceObj, &resDesc));
    }
};