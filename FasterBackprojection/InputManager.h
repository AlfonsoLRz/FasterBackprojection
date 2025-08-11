#pragma once

#include "ApplicationState.h"
#include "Singleton.h"

class ResizeListener
{
public:
	virtual void resizeEvent(uint16_t width, uint16_t height) = 0;
};

class KeyListener
{
public:
	virtual void keyEvent(int key, bool pressed) = 0;
};

class MouseButtonListener
{
public:
	virtual void mouseButtonEvent() = 0;
};

class MouseCursorListener
{
public:
	virtual void mouseCursorEvent(float x, float y, bool pressed) = 0;
};

class ScreenshotListener
{
public:
	enum ScreenshotType { RGBA };

	struct ScreenshotEvent
	{
		ScreenshotType _type;
	};

public:
	virtual void screenshotEvent(const ScreenshotEvent& event) = 0;
};

class WindowCloseListener
{
public:
	virtual void windowCloseEvent() = 0;
};

class InputManager : public Singleton<InputManager>
{
	friend class Singleton<InputManager>;

private:
	enum Events
	{
		SCREENSHOT,
		BOOM, DOLLY, DOLLY_SPEED_UP, ORBIT_XZ, ORBIT_Y, PAN, RESET, TILT, TRUCK, ZOOM,
		ALTER_POINT, ALTER_LINE, ALTER_TRIANGLE,
		NUM_EVENTS
	};

private:
	static ApplicationState			_applicationState;
	static const glm::vec2			_defaultCursorPosition;

private:
	std::vector<glm::ivec2>			_eventKey;
	glm::vec2						_lastCursorPosition;
	bool							_leftClickPressed, _rightClickPressed;
	std::vector<GLuint>				_moves;
	float							_movementMultiplier;
	std::vector<float>				_moveSpeed;
	float							_moveSpeedUp;
	GLFWwindow* _window;

private:
	// Observer pattern
	std::vector<KeyListener*>							_keyListeners;
	std::vector<MouseButtonListener*>					_mouseButtonListeners;
	std::vector<MouseCursorListener*>					_mouseCursorListeners;
	std::vector<ResizeListener*>						_resizeListeners;
	std::vector<ScreenshotListener::ScreenshotEvent>	_screenshotEvents;
	std::vector<ScreenshotListener*>					_screenshotListeners;
	std::vector<WindowCloseListener*>					_windowCloseListeners;

private:
	InputManager();
	void buildMoveRelatedBuffers();
	bool checkPanTilt(const float xPos, const float yPos);
	void processPressedKeyEvent(int key, int mods);
	void processReleasedKeyEvent(int key, int mods);

	static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	static void mouseCursorCallback(GLFWwindow* window, double xpos, double ypos);
	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
	static void windowCloseCallback(GLFWwindow* window);

public:
	virtual ~InputManager();
	static ApplicationState* getApplicationState() { return &_applicationState; }
	static void windowRefreshCallback(GLFWwindow* window);
	void init(GLFWwindow* window);

public:
	void pushScreenshotEvent(const ScreenshotListener::ScreenshotEvent& event);
	void subscribeKeyEvent(const KeyListener* listener);
	void subscribeMouseButtonEvent(MouseButtonListener* listener);
	void subscribeMouseCursorEvent(MouseCursorListener* listener);
	void subscribeResize(ResizeListener* listener);
	void subscribeScreenshot(ScreenshotListener* listener);
	void subscribeWindowClose(WindowCloseListener* listener);
};


