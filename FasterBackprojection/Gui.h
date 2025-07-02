#pragma once

#include "CameraGuiAdapter.h"
#include "Singleton.h"

class ApplicationState;
class SceneContent;

class GuiChangeListener
{
protected:
	~GuiChangeListener() = default;

public:
	virtual void cameraChangeEvent() {}
	virtual void settingsChangeEvent() {}
	virtual void materialChangeEvent() {}
};

class Gui: public Singleton<Gui>
{
	friend class Singleton<Gui>;

protected:
	enum MenuButtons { RENDERING, MODELS, CAMERA, SCREENSHOT, NUM_GUI_MENU_BUTTONS };

protected:
	CameraGuiAdapter*				_cameraGuiAdapter;
	ApplicationState*				_renderingState;
	bool*							_showMenuButtons;

	// ImGuizmo
	ImGuizmo::OPERATION				_currentGizmoOperation;
	ImGuizmo::MODE					_currentGizmoMode;

	// Observer pattern
	std::vector<GuiChangeListener*>	_changeListeners;

protected:
	static std::string checkName(const std::string& filename, const std::string& extension);
	static void editTransform(ImGuizmo::OPERATION& operation, ImGuizmo::MODE& mode);
	static void loadFonts();
	static void loadImGuiStyle();
	void showCameraMenu(const SceneContent* sceneContent) const;
	void showModelMenu(const SceneContent* sceneContent);
	void showRenderingMenu(SceneContent* sceneContent) const;
	void showScreenshotMenu(SceneContent* sceneContent) const;

protected:
	Gui();

public:
	virtual ~Gui();

	void initialize(GLFWwindow* window, const int openGLMinorVersion) const;
	void render(SceneContent* sceneContent);

	static uint16_t getFrameRate() { return static_cast<uint16_t>(ImGui::GetIO().Framerate); }
	static bool isMouseActive() { return ImGui::GetIO().WantCaptureMouse; }
	void subscribeChangeListener(GuiChangeListener* listener);
};

