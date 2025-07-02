#pragma once

#include "Camera.h"
#include "InterfaceAdapter.h"

class CameraGuiAdapter : public InterfaceAdapter
{
private:
	Camera* _activeCamera;
	Camera* _camera;

public:
	CameraGuiAdapter() : _activeCamera(nullptr), _camera(nullptr) {}
	virtual ~CameraGuiAdapter() = default;

	void	renderGuiObject(bool& changed) override;
	void	setCamera(Camera* camera, Camera* activeCamera) { _activeCamera = activeCamera; _camera = camera; }
};


