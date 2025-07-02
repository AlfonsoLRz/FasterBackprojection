#pragma once

#include "ShaderProgram.h"

class RenderingShader: public ShaderProgram
{
public:
	RenderingShader();
	~RenderingShader() override;
	void applyActiveSubroutines() override;
	GLuint createShaderProgram(const char* filename) override;
};

