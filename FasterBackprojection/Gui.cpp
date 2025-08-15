#include "stdafx.h"
#include "Gui.h"

#include "ApplicationState.h"
#include "fonts/font_awesome.hpp"
#include "fonts/lato.hpp"
#include "fonts/IconsFontAwesome5.h"
#include "GuiUtilities.h"
#include "InputManager.h"
#include "Renderer.h"
#include "SceneContent.h"

// Public methods

Gui::Gui()
{
	_cameraGuiAdapter = new CameraGuiAdapter;
	_currentGizmoOperation = ImGuizmo::TRANSLATE;
	_currentGizmoMode = ImGuizmo::WORLD;
	_renderingState = Renderer::getInstance()->getApplicationState();

	_showMenuButtons = new bool[NUM_GUI_MENU_BUTTONS];
	for (int idx = 0; idx < NUM_GUI_MENU_BUTTONS; ++idx) _showMenuButtons[idx] = false;
}

Gui::~Gui()
{
	delete[]	_showMenuButtons;

	ImGui::DestroyContext();
}

std::string Gui::checkName(const std::string& filename, const std::string& extension)
{
	if (filename.find(extension) == std::string::npos)
		return filename + "." + extension;

	return filename;
}

void Gui::editTransform(ImGuizmo::OPERATION& operation, ImGuizmo::MODE& mode)
{
	if (ImGui::RadioButton("Translate", operation == ImGuizmo::TRANSLATE))
	{
		operation = ImGuizmo::TRANSLATE;
	}

	ImGui::SameLine();
	if (ImGui::RadioButton("Rotate", operation == ImGuizmo::ROTATE))
	{
		operation = ImGuizmo::ROTATE;
	}

	ImGui::SameLine();
	if (ImGui::RadioButton("Scale", operation == ImGuizmo::SCALE))
	{
		operation = ImGuizmo::SCALE;
	}

	if (operation != ImGuizmo::SCALE)
	{
		if (ImGui::RadioButton("Local", mode == ImGuizmo::LOCAL))
		{
			mode = ImGuizmo::LOCAL;
		}

		ImGui::SameLine();
		if (ImGui::RadioButton("World", mode == ImGuizmo::WORLD))
		{
			mode = ImGuizmo::WORLD;
		}
	}
}

void Gui::loadFonts()
{
	ImFontConfig cfg;
	ImGuiIO& io = ImGui::GetIO();

	std::copy_n("Lato", 5, cfg.Name);
	io.Fonts->AddFontFromMemoryCompressedBase85TTF(LatoFont::lato_compressed_data_base85, 13.0f, &cfg);

	static const ImWchar iconRanges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
	cfg.MergeMode = true;
	cfg.PixelSnapH = true;
	cfg.GlyphMinAdvanceX = 20.0f;
	cfg.GlyphMaxAdvanceX = 20.0f;
	std::copy_n("FontAwesome", 12, cfg.Name);

	io.Fonts->AddFontFromFileTTF("assets/fonts/fa-regular-400.ttf", 12.0f, &cfg, iconRanges);
	io.Fonts->AddFontFromFileTTF("assets/fonts/fa-solid-900.ttf", 12.0f, &cfg, iconRanges);
}

void Gui::loadImGuiStyle()
{
	ImGui::StyleColorsDark();
	Gui::loadFonts();
}

void Gui::initialize(GLFWwindow* window, const int openGLMinorVersion) const
{
	const std::string openGLVersion = "#version 4" + std::to_string(openGLMinorVersion) + "0 core";

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	this->loadImGuiStyle();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(openGLVersion.c_str());
}

void Gui::render(SceneContent* sceneContent)
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

	for (int menuButtonIdx = 0; menuButtonIdx < NUM_GUI_MENU_BUTTONS; ++menuButtonIdx)
	{
		MenuButtons button = static_cast<MenuButtons>(menuButtonIdx);

		if (_showMenuButtons[button])
		{
			switch (button)
			{
			case MenuButtons::RENDERING:
				this->showRenderingMenu(sceneContent);
				break;
			case MenuButtons::CAMERA:
				this->showCameraMenu(sceneContent);
				break;
			case MenuButtons::MODELS:
				this->showModelMenu(sceneContent);
				break;
			case MenuButtons::SCREENSHOT:
				this->showScreenshotMenu(sceneContent);
				break;
			default:
				break;
			}
		}
	}

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu(ICON_FA_COG "Settings"))
		{
			ImGui::MenuItem(ICON_FA_DRAW_POLYGON "Rendering", NULL, &_showMenuButtons[MenuButtons::RENDERING]);
			ImGui::MenuItem(ICON_FA_CUBE "Models", NULL, &_showMenuButtons[MenuButtons::MODELS]);
			ImGui::MenuItem(ICON_FA_CAMERA_RETRO "Camera", NULL, &_showMenuButtons[MenuButtons::CAMERA]);
			ImGui::MenuItem(ICON_FA_CAMERA "Screenshot", NULL, &_showMenuButtons[MenuButtons::SCREENSHOT]);
			ImGui::EndMenu();
		}

		ImGui::SameLine(0, 20);
		ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
		ImGui::SameLine(0, 20);
		ImGui::Text("CUDA Allocated memory (GB): %.3f", CudaHelper::getAllocatedMemory() / (1024.0f * 1024.0f * 1024.0f));
		ImGui::EndMainMenuBar();
	}

	ImGui::Render();

	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Gui::subscribeChangeListener(GuiChangeListener* listener)
{
	_changeListeners.push_back(listener);
}

void Gui::showCameraMenu(const SceneContent* sceneContent) const
{
	static Camera* cameraSelected = nullptr;

	if (ImGui::Begin("Cameras", &this->_showMenuButtons[MenuButtons::CAMERA], ImGuiWindowFlags_None))
	{
		ImGui::BeginChild("Camera List", ImVec2(200, 0), true);

		for (int cameraIdx = 0; cameraIdx < sceneContent->_camera.size(); ++cameraIdx)
		{
			const std::string cameraName = "Camera " + std::to_string(cameraIdx);
			if (ImGui::Selectable(cameraName.c_str(), cameraSelected == sceneContent->_camera[cameraIdx].get()))
			{
				cameraSelected = sceneContent->_camera[cameraIdx].get();
				_renderingState->_selectedCamera = cameraIdx;
			}
		}

		ImGui::EndChild();
		ImGui::SameLine();

		ImGui::BeginGroup();
		ImGui::BeginChild("Model Component View", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()));		// Leave room for 1 line below us

		if (cameraSelected)
		{
			bool cameraChanged = false;

			_cameraGuiAdapter->setCamera(cameraSelected, sceneContent->_camera[_renderingState->_selectedCamera].get());
			_cameraGuiAdapter->renderGuiObject(cameraChanged);

			if (cameraChanged)
			{
				for (GuiChangeListener* listener : _changeListeners)
					listener->cameraChangeEvent();
			}
		}

		ImGui::EndChild();
		ImGui::EndGroup();

		ImGui::End();
	}
}

void Gui::showModelMenu(const SceneContent* sceneContent)
{
	ImGui::SetNextWindowSize(ImVec2(800, 440), ImGuiCond_FirstUseEver);

	if (ImGui::Begin("Models", &this->_showMenuButtons[MenuButtons::MODELS], ImGuiWindowFlags_None))
	{
		ImGui::BeginChild("Components", ImVec2(200, 0), true);

		ImGui::EndChild();
		ImGui::SameLine();

		ImGui::BeginGroup();
		ImGui::BeginChild("Mesh view", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()));		

		ImGui::EndChild();
		ImGui::EndGroup();
		ImGui::End();
	}
}

void Gui::showRenderingMenu(SceneContent* sceneContent) const
{
	if (ImGui::Begin("Rendering Settings", &_showMenuButtons[RENDERING]))
	{
		bool changedSettings = false;

		ImGui::Separator();
		ImGui::Text("Background");
		changedSettings |= ImGui::ColorEdit3("Background color", &_renderingState->_backgroundColor[0]);

		if (changedSettings)
		{
			for (GuiChangeListener* listener : _changeListeners)
				listener->settingsChangeEvent();
		}

		ImGui::End();
	}
}

void Gui::showScreenshotMenu(SceneContent* sceneContent) const
{
	auto fixName = [=](const std::string& name, const std::string& defaultName, const std::string& extension) -> std::string
	{
		if (name.empty())
			return defaultName + extension;
		else if (name.find(extension) == std::string::npos)
			return name + extension;

		return name;
	};

	if (ImGui::Begin("Screenshot Settings", &_showMenuButtons[SCREENSHOT]))
	{
		ImGui::SliderFloat("Size multiplier", &_renderingState->_screenshotFactor, 1.0f, 10.0f);
		ImGui::Checkbox("Transparent", &_renderingState->_transparentScreenshot);
		ImGui::InputText("Filename", _renderingState->_screenshotFilenameBuffer, IM_ARRAYSIZE(_renderingState->_screenshotFilenameBuffer));

		GuiUtilities::leaveSpace(2);

		if (ImGui::Button("Take screenshot (RGBA)"))
		{
			std::string filename = _renderingState->_screenshotFilenameBuffer;
			filename = fixName(filename, "RGB", ".png");
			InputManager::getInstance()->pushScreenshotEvent(ScreenshotListener::ScreenshotEvent{ ScreenshotListener::RGBA });
		}
	}

	ImGui::End();
}
