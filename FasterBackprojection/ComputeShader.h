#pragma once

#include "ShaderProgram.h"

#define TRACK_GPU_MEMORY false

class ComputeShader : public ShaderProgram
{
public:
	enum WorkGroupAxis { X_AXIS, Y_AXIS, Z_AXIS };

protected:
	static std::vector<GLint> MAX_WORK_GROUP_SIZE;					//!< This value can be useful since the number of groups is not as limited as group size
	static std::vector<GLint> MAX_NUM_WORK_GROUPS;					//!< Maximum number of invocable work groups

public:
	class MemoryFootprint
	{
	public:
		std::unordered_map<GLuint, size_t> inBuffers;				//!< Size of input buffers
		size_t size;												//!< Size of allocated GPU buffers

	public:
		MemoryFootprint() : size(0) {}
		virtual ~MemoryFootprint() {}

		void addBuffer(GLuint bufferID, const size_t size) { inBuffers[bufferID] = size; this->size += size; }
		void removeBuffer(GLuint bufferID) { size -= inBuffers[bufferID]; inBuffers.erase(bufferID); }
		void updateBuffer(GLuint bufferID, const size_t size) { this->size -= inBuffers[bufferID]; inBuffers[bufferID] = size; this->size += size; }
	};

	static MemoryFootprint _memoryFootprint;						//!< Memory footprint of all the compute shaders

public:
	ComputeShader();
	virtual ~ComputeShader();

	virtual void applyActiveSubroutines();
	static void bindBuffers(const std::vector<GLuint>& bufferID);
	virtual GLuint createShaderProgram(const char* filename);
	static void execute(GLuint numGroups_x, GLuint numGroups_y, GLuint numGroups_z, GLuint workGroup_x, GLuint workGroup_y, GLuint workGroup_z);

	static void deleteBuffer(const GLuint bufferID);
	static void deleteBuffers(const std::vector<GLuint>& bufferID);

	static std::vector<GLint> getMaxGlobalSize();
	static std::vector<GLint> getMaxLocalSize();
	static GLint getMaxGroupSize(const WorkGroupAxis axis = X_AXIS) { return MAX_WORK_GROUP_SIZE[axis]; }
	static GLint getMaxSSBOSize(unsigned elementSize);
	template<typename T>
	static float getBufferSize(const GLuint size, const T& dataType);
	static int getNumGroups(const unsigned arraySize, const WorkGroupAxis axis = X_AXIS);
	static int getWorkGroupSize(const unsigned numGroups, const unsigned arraySize);
	static void initializeMaximumSize();

	template<typename T>
	static T* readData(GLuint bufferID, const T& dataType);
	template<typename T>
	static T* readNamedBuffer(GLuint bufferID, const T& dataType, size_t offset, size_t length);
	template<typename T>
	static T* readData(GLuint bufferID, const T& dataType, size_t offset, size_t length);

	void setImageUniform(const GLint id, const std::string& shaderVariable) const;
	template<typename T>
	static GLuint setReadBuffer(const std::vector<T>& data, const GLuint flags = GL_DYNAMIC_STORAGE_BIT);
	template<typename T>
	static GLuint setReadBuffer(const T* data, const unsigned arraySize, const GLuint flags = GL_DYNAMIC_STORAGE_BIT);
	template<typename T>
	static GLuint setReadNamedBuffer(const T* data, const unsigned arraySize, const GLuint flags = GL_DYNAMIC_STORAGE_BIT);
	template<typename T>
	static GLuint setReadData(const T& data, const GLuint flags = GL_DYNAMIC_STORAGE_BIT);
	template<typename T>
	static GLuint setWriteBuffer(const T& dataType, const GLuint arraySize, const GLuint flags = GL_DYNAMIC_STORAGE_BIT);
	template<typename T>
	static GLuint setNamedBuffer(const T& dataType, const GLuint arraySize, const GLuint flags = GL_DYNAMIC_STORAGE_BIT);

	template<typename T>
	static void updateReadBuffer(const GLuint id, const T* data, const unsigned arraySize, const GLuint flags = GL_DYNAMIC_STORAGE_BIT);
	template<typename T>
	static void updateWriteBuffer(const GLuint id, const T& dataType, const unsigned arraySize, const GLuint flags = GL_DYNAMIC_STORAGE_BIT);
	template<typename T>
	static void updateReadBufferSubset(const GLuint id, const T* data, const unsigned offset, const unsigned arraySize);
	template<typename T>
	static void updateNamedBuffer(const GLuint id, const T* data, const unsigned arraySize, const GLuint flags = GL_DYNAMIC_STORAGE_BIT);
};

template <typename T>
float ComputeShader::getBufferSize(const GLuint size, const T& dataType)
{
	return size * sizeof(T) / 1024.0 / 1024.0;
}

template<typename T>
inline T* ComputeShader::readData(GLuint bufferID, const T& dataType)
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID);																	
	T* data = (T*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	return data;
}

template<typename T>
inline T* ComputeShader::readNamedBuffer(GLuint bufferID, const T& dataType, size_t offset, size_t length)
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID);
	T* data = (T*)glMapNamedBufferRange(GL_SHADER_STORAGE_BUFFER, offset, length, GL_MAP_READ_BIT);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	return data;
}

template<typename T>
inline T* ComputeShader::readData(GLuint bufferID, const T& dataType, size_t offset, size_t length)
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID);
	T* data = (T*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, offset, length, GL_MAP_READ_BIT);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	return data;
}

template<typename T>
inline GLuint ComputeShader::setReadBuffer(const std::vector<T>& data, const GLuint flags)
{
	GLuint id;
	glGenBuffers(1, &id);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * data.size(), data.data(), flags);

#if TRACK_GPU_MEMORY
	_memoryFootprint.addBuffer(id, sizeof(T) * data.size());
#endif

	return id;
}

template<typename T>
inline GLuint ComputeShader::setReadBuffer(const T* data, const unsigned arraySize, const GLuint flags)
{
	GLuint id;
	glGenBuffers(1, &id);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * arraySize, data, flags);

#if TRACK_GPU_MEMORY
	_memoryFootprint.addBuffer(id, sizeof(T) * arraySize);
#endif

	return id;
}

template<typename T>
inline GLuint ComputeShader::setReadNamedBuffer(const T* data, const unsigned arraySize, const GLuint flags)
{
	GLuint id;
	glCreateBuffers(1, &id);
	glNamedBufferStorage(id, sizeof(T) * arraySize, data, flags);

#if TRACK_GPU_MEMORY
	_memoryFootprint.addBuffer(id, sizeof(T) * arraySize);
#endif

	return id;
}

template<typename T>
inline GLuint ComputeShader::setReadData(const T& data, const GLuint flags)
{
	GLuint id;
	glGenBuffers(1, &id);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T), &data, flags);

#if TRACK_GPU_MEMORY
	_memoryFootprint.addBuffer(id, sizeof(T));
#endif

	return id;
}

template<typename T>
inline GLuint ComputeShader::setWriteBuffer(const T& dataType, const GLuint arraySize, const GLuint flags)
{
	GLuint id;
	glGenBuffers(1, &id);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(dataType) * arraySize, nullptr, flags);

#if TRACK_GPU_MEMORY
	_memoryFootprint.addBuffer(id, sizeof(T) * arraySize);
#endif

	return id;
}

template<typename T>
inline GLuint ComputeShader::setNamedBuffer(const T& dataType, const GLuint arraySize, const GLuint flags)
{
	GLuint id;
	glCreateBuffers(1, &id);
	glNamedBufferStorage(id, sizeof(dataType) * arraySize, nullptr, flags);

#if TRACK_GPU_MEMORY
	_memoryFootprint.addBuffer(id, sizeof(T) * arraySize);
#endif

	return id;
}

template<typename T>
inline void ComputeShader::updateReadBuffer(const GLuint id, const T* data, const unsigned arraySize, const GLuint flags)
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * arraySize, data, flags);

#if TRACK_GPU_MEMORY
	_memoryFootprint.updateBuffer(id, sizeof(T) * arraySize);
#endif
}

template<typename T>
inline void ComputeShader::updateWriteBuffer(const GLuint id, const T& dataType, const unsigned arraySize, const GLuint flags)
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(dataType) * arraySize, nullptr, flags);
}

template<typename T>
inline void ComputeShader::updateReadBufferSubset(const GLuint id, const T* data, const unsigned offset, const unsigned arraySize)
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, sizeof(T) * arraySize, data);
}

template<typename T>
inline void ComputeShader::updateNamedBuffer(const GLuint id, const T* data, const unsigned arraySize, const GLuint flags)
{
	glNamedBufferStorage(id, sizeof(T) * arraySize, data, flags);
}

