// ReSharper disable All
#include "stdafx.h"
#include "DrawPointCloud.h"

#include "ShaderProgramDB.h"

// Public methods

DrawPointCloud::DrawPointCloud (const std::vector<glm::vec3>& points)
{
    _components.resize(1);
	Component* component = &_components.front();

	for (const glm::vec3& point : points)
        component->_vertices.push_back(Vao::Vertex{ ._position= point});

    this->calculateAABB();
    component->generatePointCloud();
    component->buildVao();

	_pointCloudShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::POINT_RENDERING);
}

void DrawPointCloud::draw(MatrixRenderInformation* matrixInformation, ApplicationState* appState)
{
	if (appState->_primitiveEnabled[0] == false) return; // Points are not enabled

    for (auto& component : _components)
    {
        if (component._enabled && component._vao)
        {
			const glm::mat4 modelMatrix = 
                matrixInformation->_matrix[MatrixRenderInformation::VIEW_PROJECTION] * 
                matrixInformation->_matrix[MatrixRenderInformation::MODEL] * 
                this->_modelMatrix;

            _pointCloudShader->use();
			_pointCloudShader->setUniform("pointColor", component._material._pointColor);
            _pointCloudShader->setUniform("pointSize", component._pointSize);
            _pointCloudShader->setUniform("mModelViewProj", modelMatrix);
            _pointCloudShader->applyActiveSubroutines();

            component._vao->drawObject(Vao::POINT, GL_POINTS, static_cast<GLuint>(component._indices[Vao::POINT].size()));
        }
    }
}
