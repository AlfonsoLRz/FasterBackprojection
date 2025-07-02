#pragma once

#include "AABB.h"
#include "ApplicationState.h"

class Camera;
class Model3D;

class SceneContent
{
public:
	std::vector<std::unique_ptr<Camera>>	_camera;
	std::vector<std::unique_ptr<Model3D>>	_model;
	AABB									_sceneAABB;

	glm::uint								_numVertices;
	glm::uint								_numMeshes;
	glm::uint								_numTextures;
	glm::uint								_numTriangles;

public:
	SceneContent();
	virtual ~SceneContent();

	void addNewCamera(ApplicationState* appState);
	void addNewModel(Model3D* model);
	void buildScenario();
};
