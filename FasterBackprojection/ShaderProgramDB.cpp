#include "stdafx.h"
#include "ShaderProgramDB.h"

// Static attributes

std::unordered_map<uint8_t, std::string> ShaderProgramDB::RENDERING_SHADER_PATH {
		{RenderingShaderId::LINE_RENDERING, "assets/shaders/shading/line"},
		{RenderingShaderId::POINT_RENDERING, "assets/shaders/shading/point"},
		{RenderingShaderId::TRIANGLE_RENDERING, "assets/shaders/shading/triangle"},
		{RenderingShaderId::MULTI_INSTANCE_TRIANGLE_RENDERING, "assets/shaders/shading/multi-triangle"},
		{RenderingShaderId::BACKPROJECTION_RENDERING, "assets/shaders/shading/backprojection"},
		{RenderingShaderId::BACKPROJECTION_MESH_RENDERING, "assets/shaders/shading/backprojection_mesh"}
};

std::unordered_map<uint8_t, std::unique_ptr<RenderingShader>> ShaderProgramDB::_renderingShader;

// Private methods

ShaderProgramDB::ShaderProgramDB() = default;

ShaderProgramDB::~ShaderProgramDB() = default;

// Public methods

RenderingShader* ShaderProgramDB::getShader(RenderingShaderId shaderId)
{
	uint8_t shaderId8 = static_cast<uint8_t>(shaderId);

	if (!_renderingShader[shaderId8].get())
	{
		RenderingShader* shader = new RenderingShader();
		shader->createShaderProgram(RENDERING_SHADER_PATH.at(shaderId8).c_str());

		_renderingShader[shaderId8].reset(shader);
	}

	return _renderingShader[shaderId8].get();
}
