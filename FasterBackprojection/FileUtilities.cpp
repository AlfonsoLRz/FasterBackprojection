#include "stdafx.h"
#include "FileUtilities.h"

bool FileUtilities::createDirectory(const std::string& path)
{
	if (std::filesystem::exists(path))
	{
		//std::cerr << "Directory already exists: " << path << '\n';
		return false;
	}
	try
	{
		std::filesystem::create_directories(path);
	}
	catch (const std::filesystem::filesystem_error& e)
	{
		//std::cerr << "Error creating directory: " << e.what() << '\n';
		return false;
	}
	return true;
}

bool FileUtilities::writePoints(const std::string& filename, const std::vector<glm::vec3>& points)
{
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << filename << '\n';
		return false;
	}

	for (const auto& point : points)
	{
		file << point.x << " " << point.y << " " << point.z << "\n";
	}

	file.close();

	return true;
}
