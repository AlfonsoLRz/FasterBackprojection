#pragma once

#include "Model3D.h"

class DrawEllipsoids : public Model3D
{
protected:
	RenderingShader*	_lineShader;
	RenderingShader*	_triangleShader;
	RenderingShader*	_multiInstanceTriangleShader;
	RenderingShader*	_backprojectionShader;

	NLosData*			_nlosData;
	glm::uint			_numInstances;

protected:
	static void createEllipsoid(Component* component, int stacks, int sectors);
	static void createHalfEllipsoid(Component* component, int stacks, int sectors);

	void MIS(std::vector<glm::vec3>& ellipsoidPositions, std::vector<glm::vec3>& ellipsoidScale, std::vector<float>& ellipsoidIntensities, glm::uint numSamples);
	void reorder(std::vector<glm::vec3>& ellipsoidPositions, std::vector<glm::vec3>& ellipsoidScale, std::vector<float>& ellipsoidIntensities, const AABB& relayWall);

public:
	DrawEllipsoids(NLosData* nlosData);
	~DrawEllipsoids() override;

	void draw(MatrixRenderInformation* matrixInformation, ApplicationState* appState) override;

	void solveBackprojection();
};

