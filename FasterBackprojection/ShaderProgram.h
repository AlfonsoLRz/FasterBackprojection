#pragma once

#define DEBUG_UNIFORMS true

class ShaderProgram
{
public:
	enum ShaderTypes : uint8_t
	{
		VERTEX_SHADER,
		FRAGMENT_SHADER,
		GEOMETRY_SHADER,
		COMPUTE_SHADER
	};

protected:
	const static std::string MODULE_HEADER;
	const static std::string MODULE_FILE_CHAR_1;
	const static std::string MODULE_FILE_CHAR_2;

protected:
	static std::unordered_map<std::string, std::string> _moduleCode;			//!< Modules that are already loaded

protected:
	std::vector<GLuint> _activeSubroutineUniform[COMPUTE_SHADER + 1];			//!< Active uniform for each subroutine for each shader type
	GLuint				_handler;												//!< Shader program id in GPU
	bool				_linked;												//!< Flag which tell us if the shader has been linked correctly
	std::string			_logString;												//!< Error message got from the last operation with the shader

protected:
	virtual GLuint compileShader(const char* filename, const GLenum shaderType);
	static bool fileExists(const std::string& fileName);
	static ShaderTypes fromOpenGLToShaderTypes(const GLenum shaderType);
	bool includeLibraries(std::string& shaderContent);
	virtual bool loadFileContent(const std::string& filename, std::string& content);
	static bool showErrorMessage(const std::string& variableName);

public:
	ShaderProgram();
	virtual ~ShaderProgram();

	virtual void applyActiveSubroutines() = 0;
	virtual GLuint createShaderProgram(const char* filename) = 0;

	bool setSubroutineUniform(const GLenum shaderType, const std::string& subroutine, const std::string& functionName);
	bool setUniform(const std::string& name, const GLfloat value) const;
	bool setUniform(const std::string& name, const GLint value) const;
	bool setUniform(const std::string& name, const GLuint value) const;
	bool setUniform(const std::string& name, const glm::mat4& value) const;
	bool setUniform(const std::string& name, const std::vector<glm::mat4>& values) const;
	bool setUniform(const std::string& name, const glm::vec2& value) const;
	bool setUniform(const std::string& name, const glm::ivec2& value) const;
	bool setUniform(const std::string& name, const glm::uvec2& value) const;
	bool setUniform(const std::string& name, const glm::vec3& value) const;
	bool setUniform(const std::string& name, const glm::uvec3& value) const;
	bool setUniform(const std::string& name, const glm::ivec3& value) const;
	bool setUniform(const std::string& name, const glm::vec4& value) const;
	bool setUniform(const std::string& name, const std::vector<float>& values) const;
	bool setUniformBlock(const std::string& name, const GLuint bufferID) const;

	bool use() const;
};
