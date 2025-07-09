#include "stdafx.h"
#include "SceneContent.h"

#include "Camera.h"
#include "Laser.cuh"
#include "Model3D.h"
#include "TriangleMesh.h"

#include "NLosData.h"
#include "NLosDataVisualizer.h"

// ----------------------------- BUILD YOUR SCENARIO HERE -----------------------------------

void SceneContent::buildScenario()
{
	TransientParameters transientParameters;

	// Use real data and reconstruct the shape
	//NLosData* transientVoxels = new NLosData("assets/transient/z/Z_l[0.00,-1.00,0.00]_r[1.57,0.00,3.14]_v[0.81,0.01,0.81]_s[16]_l[16]_gs[1.00].hdf5");
	NLosData* transientVoxels = new NLosData("assets/transient/z/Z_l[0.00,-0.50,0.00]_r[1.57,0.00,3.14]_v[0.81,0.01,0.81]_s[256]_l[256]_gs[1.00]_conf.hdf5");
	//NLosData* transientVoxels = new NLosData("assets/transient/t/T_l[0.00,-0.40,0.00]_r[1.57,0.00,3.14]_v[0.40]_s[256]_l[256]_gs[1.50]_conf.hdf5");
	//NLosData* transientVoxels = new NLosData("assets/transient/usaf/usaf_l[0.00,-0.50,0.00]_r[1.57,0.00,3.14]_v[0.57,0.01,0.60]_s[256]_l[256]_gs[1.00]_conf.hdf5");

	//transientVoxels.saveImages("output/");
	NLosDataVisualizer* nlosVisualizer = new NLosDataVisualizer(transientVoxels);
	this->addNewModel(nlosVisualizer);

	Laser laser(transientVoxels);
	laser.reconstructShape(transientParameters);
}

SceneContent::SceneContent() : _numVertices(0), _numMeshes(0), _numTextures(0), _numTriangles(0)
{
}

SceneContent::~SceneContent()
{
    _camera.clear();
    _model.clear();
}

void SceneContent::addNewCamera(ApplicationState* appState)
{
    _camera.push_back(std::make_unique<Camera>(appState->_viewportSize.x, appState->_viewportSize.y, false));
}

void SceneContent::addNewModel(Model3D* model)
{
    _sceneAABB.update(model->getAABB());
    _model.push_back(std::unique_ptr<Model3D>(model));
}
