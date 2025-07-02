#pragma once

#include "Model3D.h"

class DrawCube : public Model3D
{
protected:
	RenderingShader* _pointShader;
	RenderingShader* _lineShader;
	RenderingShader* _triangleShader;

public:
	DrawCube(const AABB& aabb);
	virtual ~DrawCube() = default;

	void draw(MatrixRenderInformation* matrixInformation, ApplicationState* appState) override;
};

