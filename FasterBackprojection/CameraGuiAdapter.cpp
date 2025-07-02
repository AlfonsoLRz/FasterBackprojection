#include "stdafx.h"
#include "CameraGuiAdapter.h"

#include "GuiUtilities.h"

void CameraGuiAdapter::renderGuiObject(bool& changed)
{
	changed = false;

	const char* projectionTitle[] = { "Perspective", "Orthographic", "Fisheye", "Stereographic", "Equirectangular" };
	changed |= ImGui::Combo("Camera Type", &_camera->_properties._cameraType, projectionTitle, IM_ARRAYSIZE(projectionTitle));

	//GuiUtilities::leaveSpace(2);
	//ImGui::Text("Current information");
	//ImGui::Separator();
	//GuiUtilities::leaveSpace(2);

	GuiUtilities::leaveSpace(4);
	ImGui::Text("Camera");
	ImGui::Separator();
	GuiUtilities::leaveSpace(2);
	changed |= ImGui::InputFloat3("Eye", &_camera->_properties._eye[0]);
	changed |= ImGui::InputFloat3("Look at", &_camera->_properties._lookAt[0]);
	changed |= ImGui::SliderFloat("Z near", &_camera->_properties._zNear, 0.1f, _camera->_properties._zFar);
	changed |= ImGui::SliderFloat("Z far", &_camera->_properties._zFar, _camera->_properties._zNear, 1000.0f);
	changed |= ImGui::SliderFloat("FoV X", &_camera->_properties._fovX, 0.1f, glm::pi<float>() / 2);
	changed |= ImGui::SliderFloat("Blur strength", &_camera->_properties._blurStrength, 0.0f, 3.0f);
	changed |= ImGui::SliderFloat("Defocus angle", &_camera->_properties._defocusAngle, 0.0f, 4.0f);
	changed |= ImGui::SliderFloat("Focus point", &_camera->_properties._focusPoint, 0.0f, 15.0f);

	if (changed)
		_camera->updateMatrices();
}
