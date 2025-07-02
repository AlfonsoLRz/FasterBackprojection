#pragma once

namespace GuiUtilities
{
	void leaveSpace(unsigned numSlots);
	void renderText(const glm::vec3& xyz, const std::string& title = "", char delimiter = ',');
	void renderText(const std::string& title, const std::string& content);
}
