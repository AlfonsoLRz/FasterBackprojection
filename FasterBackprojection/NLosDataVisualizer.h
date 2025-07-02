#pragma once

#include "DrawCube.h"
#include "Model3D.h"

class DrawEllipsoids;
class NLosData;
class DrawPointCloud;

class NLosDataVisualizer : public Model3D
{
protected:
	DrawPointCloud* _relayWallLaserTargets;
	DrawPointCloud* _relayWallCameraTargets;
	DrawCube*		_hiddenGeometryCube;
	DrawEllipsoids* _ellipsoids;

public:
	NLosDataVisualizer(NLosData* nlosData);
	~NLosDataVisualizer() override;

	void draw(MatrixRenderInformation* matrixInformation, ApplicationState* appState) override;
};

