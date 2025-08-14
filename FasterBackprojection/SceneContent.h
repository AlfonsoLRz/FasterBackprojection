#pragma once

#include "AABB.h"
#include "ApplicationState.h"
#include "NlosStreamingEngine.h"

class Camera;
class Model3D;

class SceneContent
{
	using ReconstructionEngine = rtnlos::NlosStreamingEngine<NUMBER_OF_ROWS, NUMBER_OF_COLS, NUMBER_OF_FREQUENCIES>;
public:
	std::vector<std::unique_ptr<Camera>>	_camera;
	std::vector<std::unique_ptr<Model3D>>	_model;
	AABB									_sceneAABB;

	glm::uint								_numVertices;
	glm::uint								_numMeshes;
	glm::uint								_numTextures;
	glm::uint								_numTriangles;

	ReconstructionEngine*					_reconstructionEngine;

public:
	SceneContent();
	virtual ~SceneContent();

	void addNewCamera(ApplicationState* appState);
	void addNewModel(Model3D* model);
	void buildScenario(cudaSurfaceObject_t cudaSurface);
};
