#pragma once

#include "AABB.h"
#include "ApplicationState.h"
#include "SPADStreamingEngine.h"

class Camera;
class Model3D;

class SceneContent
{
public:
	std::vector<std::unique_ptr<Camera>>	_camera;
	std::vector<std::unique_ptr<Model3D>>	_model;
	rtnlos::ReconstructionEngine*			_reconstructionEngine;
	AABB									_sceneAABB;

public:
	SceneContent();
	virtual ~SceneContent();

	void addNewCamera(ApplicationState* appState);
	void addNewModel(Model3D* model);
	void buildScenario();
};
