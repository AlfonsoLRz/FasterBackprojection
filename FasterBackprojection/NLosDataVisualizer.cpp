#include "stdafx.h"
#include "NLosDataVisualizer.h"

#include "DrawEllipsoids.h"
#include "DrawPointCloud.h"
#include "NLosData.h"

NLosDataVisualizer::NLosDataVisualizer(NLosData* nlosData) :
	_relayWallLaserTargets(nullptr), _relayWallCameraTargets(nullptr), _hiddenGeometryCube(nullptr), _ellipsoids(nullptr)
{
	assert(!nlosData->_laserGridPositions.empty());
	_relayWallLaserTargets = new DrawPointCloud(nlosData->_laserGridPositions);
	_relayWallLaserTargets->setPointColor(glm::vec3(0.0f, 1.0f, 0.0f));

	assert(!nlosData->_cameraGridPositions.empty());
	_relayWallCameraTargets = new DrawPointCloud(nlosData->_cameraGridPositions);
	_relayWallCameraTargets->setPointColor(glm::vec3(1.0f, 0.0f, 0.0f));

	_hiddenGeometryCube = new DrawCube(nlosData->_hiddenGeometry);
	_hiddenGeometryCube->setTriangleColor(glm::vec4(0.5f));

	_ellipsoids = new DrawEllipsoids(nlosData);
	_ellipsoids->setTriangleColor({ 1.0f, 0.0f, 1.0f, 0.8f });
	//_ellipsoids->solveBackprojection();
}

NLosDataVisualizer::~NLosDataVisualizer()
{
	delete _relayWallLaserTargets;
	delete _relayWallCameraTargets;
	delete _hiddenGeometryCube;
	delete _ellipsoids;
}

void NLosDataVisualizer::draw(MatrixRenderInformation* matrixInformation, ApplicationState* appState)
{
	//_relayWallCameraTargets->draw(matrixInformation, appState);
	//_hiddenGeometryCube->draw(matrixInformation, appState);
	_ellipsoids->draw(matrixInformation, appState);
	//_ellipsoids->solveBackprojection();
}
