#pragma once

#include "RenderingShader.h"
#include "Singleton.h"

class ShaderProgramDB : public Singleton<ShaderProgramDB>
{
	friend class Singleton<ShaderProgramDB>;

private:
	static std::unordered_map<uint8_t, std::string>							RENDERING_SHADER_PATH;
	static std::unordered_map<uint8_t, std::unique_ptr<RenderingShader>>	_renderingShader;

public:
	enum RenderingShaderId
	{
		POINT_RENDERING,
		LINE_RENDERING,
		TRIANGLE_RENDERING,
		MULTI_INSTANCE_TRIANGLE_RENDERING,
		BACKPROJECTION_RENDERING
	};

private:
	ShaderProgramDB();

public:
	virtual ~ShaderProgramDB();
	static RenderingShader* getShader(RenderingShaderId shaderId);
};

