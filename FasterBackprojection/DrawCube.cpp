// ReSharper disable All
#include "stdafx.h"
#include "DrawCube.h"

#include "ShaderProgramDB.h"

DrawCube::DrawCube(const AABB& aabb)
{
	_components.resize(1);
	Component* component = &_components.front();

	const glm::vec3 min = aabb.minPoint();
	const glm::vec3 max = aabb.maxPoint();
	component->_vertices = {
		Vao::Vertex {._position = { min.x, min.y, min.z}},
		Vao::Vertex {._position = { max.x, min.y, min.z}},
		Vao::Vertex {._position = { max.x, max.y, min.z}},
		Vao::Vertex {._position = { min.x, max.y, min.z}},
		Vao::Vertex {._position = { min.x, min.y, max.z}},
		Vao::Vertex {._position = { max.x, min.y, max.z}},
		Vao::Vertex {._position = { max.x, max.y, max.z}},
		Vao::Vertex {._position = { min.x, max.y, max.z}}
	};
	component->_indices[Vao::TRIANGLE] = {
		0, 1, 2, 0, 2, 3,
		4, 5, 6, 4, 6, 7,
		0, 1, 5, 0, 5, 4,
		2, 3, 7, 2, 7, 6,
		0, 3, 7, 0, 7, 4,
		1, 2, 6, 1, 6, 5
	};
	component->generateWireframe();
	component->generatePointCloud();
	component->buildVao();
	this->calculateAABB();

	_pointShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::POINT_RENDERING);
	_lineShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::LINE_RENDERING);
	_triangleShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::TRIANGLE_RENDERING);
}

void DrawCube::draw(MatrixRenderInformation* matrixInformation, ApplicationState* appState)
{
	const glm::mat4 modelMatrix =
		matrixInformation->_matrix[MatrixRenderInformation::VIEW_PROJECTION] *
		matrixInformation->_matrix[MatrixRenderInformation::MODEL] *
		this->_modelMatrix;

	Component* component = &_components.front();

	if (appState->_primitiveEnabled[Vao::TRIANGLE] && component->_enabled && component->_vao)
	{
		_triangleShader->use();
		_triangleShader->setUniform("mModelViewProj", modelMatrix);
		_triangleShader->setUniform("Kd", component->_material._kdColor);
		_triangleShader->applyActiveSubroutines();
		component->_vao->drawObject(Vao::TRIANGLE, GL_TRIANGLES, static_cast<GLuint>(component->_indices[Vao::TRIANGLE].size()));
	}

	if (appState->_primitiveEnabled[Vao::LINE] && component->_enabled && component->_vao)
	{
		_lineShader->use();
		_lineShader->setUniform("mModelViewProj", modelMatrix);
		_lineShader->setUniform("lineColor", component->_material._lineColor);
		_lineShader->applyActiveSubroutines();
		component->_vao->drawObject(Vao::LINE, GL_LINES, static_cast<GLuint>(component->_indices[Vao::LINE].size()));
	}

	if (appState->_primitiveEnabled[Vao::POINT] && component->_enabled && component->_vao)
	{
		_pointShader->use();
		_pointShader->setUniform("mModelViewProj", modelMatrix);
		_pointShader->setUniform("pointColor", component->_material._pointColor);
		_pointShader->setUniform("pointSize", component->_pointSize);
		_pointShader->applyActiveSubroutines();
		component->_vao->drawObject(Vao::POINT, GL_POINTS, static_cast<GLuint>(component->_indices[Vao::POINT].size()));
	}
}
