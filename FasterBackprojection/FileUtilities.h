#pragma once

class FileUtilities
{
public:
	static bool createDirectory(const std::string& path);

	template<typename T>
	static bool write(const std::string& filename, const std::vector<T>& values);
	static bool writePoints(const std::string& filename, const std::vector<glm::vec3>& points);
};

template <typename T>
bool FileUtilities::write(const std::string& filename, const std::vector<T>& values)
{
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << filename << '\n';
		return false;
	}

	file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
	file.close();

	return true;
}

