#include "stdafx.h"
#include "GuiUtilities.h"

void GuiUtilities::leaveSpace(unsigned numSlots)
{
	for (unsigned i = 0; i < numSlots; ++i) ImGui::Spacing();
}

void GuiUtilities::renderText(const glm::vec3& xyz, const std::string& title, char delimiter)
{
	std::string txt = title + (title.empty() ? "" : ": ") + std::to_string(xyz.x) + delimiter + ' ' + std::to_string(xyz.y) + delimiter + ' ' + std::to_string(xyz.z);
	ImGui::Text(txt.c_str());
}

void GuiUtilities::renderText(const std::string& title, const std::string& content)
{
	std::string txt = title + (title.empty() ? "" : ": ") + content;
	ImGui::Text(txt.c_str());
}
