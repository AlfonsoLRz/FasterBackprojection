#include "stdafx.h"
#include "RenderingShader.h"

RenderingShader::RenderingShader(): ShaderProgram()
{
}

RenderingShader::~RenderingShader()
{
}

void RenderingShader::applyActiveSubroutines()
{
	glUniformSubroutinesuiv(GL_VERTEX_SHADER, static_cast<uint32_t>(_activeSubroutineUniform[VERTEX_SHADER].size()), _activeSubroutineUniform[VERTEX_SHADER].data());
	glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, static_cast<uint32_t>(_activeSubroutineUniform[FRAGMENT_SHADER].size()), _activeSubroutineUniform[FRAGMENT_SHADER].data());
	glUniformSubroutinesuiv(GL_GEOMETRY_SHADER, static_cast<uint32_t>(_activeSubroutineUniform[GEOMETRY_SHADER].size()), _activeSubroutineUniform[GEOMETRY_SHADER].data());
}

GLuint RenderingShader::createShaderProgram(const char* filename)
{
	if (_handler <= 0) {										// Shader identifier
		_handler = glCreateProgram();

		if (_handler == 0) {
			fprintf(stderr, "Cannot create shader program: %s!\n", filename);
			return 0;
		}
	}

	// [Vertex shader]
	char fileNameComplete[256];
	strcpy_s(fileNameComplete, _countof(fileNameComplete), filename);
	strcat_s(fileNameComplete, _countof(fileNameComplete), "-vs.glsl");

	const GLuint vertexShaderObject = compileShader(fileNameComplete, GL_VERTEX_SHADER);
	if (vertexShaderObject == 0) {
		return 0;
	}

	// [Fragment shader]
	strcpy_s(fileNameComplete, _countof(fileNameComplete), filename);
	strcat_s(fileNameComplete, _countof(fileNameComplete), "-fs.glsl");

	const GLuint fragmentShaderObject = compileShader(fileNameComplete, GL_FRAGMENT_SHADER);
	if (fragmentShaderObject == 0) {
		return 0;
	}

	// [Geometry shader, optional]
	strcpy_s(fileNameComplete, _countof(fileNameComplete), filename);
	strcat_s(fileNameComplete, _countof(fileNameComplete), "-gs.glsl");
	const GLuint geometryShaderObject = compileShader(fileNameComplete, GL_GEOMETRY_SHADER);

	glAttachShader(_handler, vertexShaderObject);						// Associate shaders with shader program
	glAttachShader(_handler, fragmentShaderObject);
	if (geometryShaderObject != 0) glAttachShader(_handler, geometryShaderObject);

	glLinkProgram(_handler);											// Link shader program to check errors
	GLint linkSuccess = 0;
	glGetProgramiv(_handler, GL_LINK_STATUS, &linkSuccess);

	if (linkSuccess == GL_FALSE)
	{
		GLint logLen = 0;
		glGetProgramiv(_handler, GL_INFO_LOG_LENGTH, &logLen);

		if (logLen > 0) {
			char* cLogString = new char[logLen];
			GLint written = 0;
			glGetProgramInfoLog(_handler, logLen, &written, cLogString);
			_logString.assign(cLogString);
			delete[] cLogString;

			std::cout << "Could not link shader " << filename << "!" << '\n' << _logString << '\n';
		}

		return 0;
	}
	else
	{
		_linked = true;
	}

	// Allocate space for subroutines
	GLint numSubroutines;
	glGetProgramStageiv(_handler, GL_VERTEX_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORMS, &numSubroutines);
	_activeSubroutineUniform[VERTEX_SHADER].resize(numSubroutines);
	std::ranges::fill(_activeSubroutineUniform[VERTEX_SHADER], -1);				// Non valid ID

	glGetProgramStageiv(_handler, GL_FRAGMENT_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORMS, &numSubroutines);
	_activeSubroutineUniform[FRAGMENT_SHADER].resize(numSubroutines);
	std::ranges::fill(_activeSubroutineUniform[FRAGMENT_SHADER], -1);

	glGetProgramStageiv(_handler, GL_GEOMETRY_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORMS, &numSubroutines);
	_activeSubroutineUniform[GEOMETRY_SHADER].resize(numSubroutines);
	std::ranges::fill(_activeSubroutineUniform[GEOMETRY_SHADER], -1);

	glDeleteShader(vertexShaderObject);
	glDeleteShader(fragmentShaderObject);
	if (geometryShaderObject != 0) glDeleteShader(geometryShaderObject);

	return _handler;
}
