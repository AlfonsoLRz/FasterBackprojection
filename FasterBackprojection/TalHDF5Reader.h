#pragma once

#include "TransientFileReader.h"

class TalHDF5Reader : public TransientFileReader
{
protected:
	static void setUp(glm::vec2& data, const std::vector<float>& rawData);
	static void setUp(glm::vec3& data, const std::vector<float>& rawData);
	static void setUp(std::vector<glm::vec3>& data, const std::vector<std::vector<std::vector<float>>>& rawData);

	// Confocal
	static void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& rawData, std::vector<size_t>& dims);
	static void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<float>>>>& rawData, std::vector<size_t>& dims);
	static void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<float>>>& rawData, std::vector<size_t>& dims);

	// Exhaustive
	static void setUp(std::vector<float>& data, 
					  const std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>>>& rawData, 
					  std::vector<size_t>& dims);

public:
	bool read(const std::string& filename, NLosData& nlosData) override;
};

inline void TalHDF5Reader::setUp(glm::vec2& data, const std::vector<float>& rawData)
{
	if (rawData.size() != 2)
		throw std::runtime_error("NLOSata: Invalid raw data size for glm::vec2.");
	data = glm::vec2(rawData[0], rawData[1]);
}

inline void TalHDF5Reader::setUp(glm::vec3& data, const std::vector<float>& rawData)
{
	if (rawData.size() != 3)
		throw std::runtime_error("NLOSata: Invalid raw data size for glm::vec3.");
	data = glm::vec3(rawData[0], rawData[1], rawData[2]);
}

inline void TalHDF5Reader::setUp(std::vector<glm::vec3>& data, const std::vector<std::vector<std::vector<float>>>& rawData)
{
	size_t width = rawData.size(), height = rawData[0].size();
	data.resize(width * height);

	#pragma omp parallel for
	for (int x = 0; x < static_cast<int>(width); ++x)
	{
		for (size_t y = 0; y < height; ++y)
		{
			data[x * width + y] = glm::vec3(rawData[x][y][0], rawData[x][y][1], rawData[x][y][2]);
		}
	}
}

inline void TalHDF5Reader::setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& rawData, std::vector<size_t>& dims)
{
	size_t	numChannels = rawData.size(), numTimeBins = rawData[0].size(), numBounces = rawData[0][0].size(),
		numRows = rawData[0][0][0].size(), numCols = rawData[0][0][0][0].size();

	assert(numChannels == 1);
	data.resize(numTimeBins * numCols * numRows, 0);

	#pragma omp parallel for
	for (size_t t = 0; t < numTimeBins; ++t)
	{
		for (size_t x = 0; x < numCols; ++x)
		{
			for (size_t y = 0; y < numRows; ++y)
			{
				for (size_t b = 0; b < numBounces; ++b)
				{
					data[y * numCols * numTimeBins + x * numTimeBins + t] += static_cast<float>(rawData[0][t][b][y][x]);
				}
			}
		}
	}

	dims = { numRows, numCols, numTimeBins };
}

inline void TalHDF5Reader::setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<float>>>>& rawData, std::vector<size_t>& dims)
{
	size_t	numTimeBins = rawData.size(), numBounces = rawData[0].size(), numRows = rawData[0][0].size(), numCols = rawData[0][0][0].size();
	data.resize(numTimeBins * numCols * numRows, 0);

	#pragma omp parallel for
	for (size_t t = 0; t < numTimeBins; ++t)
	{
		for (size_t x = 0; x < numCols; ++x)
		{
			for (size_t y = 0; y < numRows; ++y)
			{
				for (size_t b = 0; b < numBounces; ++b)
				{
					data[y * numCols * numTimeBins + x * numTimeBins + t] += static_cast<float>(rawData[t][b][y][x]);
				}
			}
		}
	}

	dims = { numRows, numCols, numTimeBins };
}

inline void TalHDF5Reader::setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<float>>>& rawData, std::vector<size_t>& dims)
{
	size_t	numTimeBins = rawData.size(), numRows = rawData[0].size(), numCols = rawData[0][0].size();
	data.resize(numTimeBins * numCols * numRows, 0);

	#pragma omp parallel for
	for (size_t t = 0; t < numTimeBins; ++t)
	{
		for (size_t x = 0; x < numCols; ++x)
		{
			for (size_t y = 0; y < numRows; ++y)
			{
				data[y * numCols * numTimeBins + x * numTimeBins + t] += static_cast<float>(rawData[t][y][x]);
			}
		}
	}

	dims = { numRows, numCols, numTimeBins };
}

inline void TalHDF5Reader::setUp(std::vector<float>& data,
                                 const std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>>>& rawData, 
								 std::vector<size_t>& dims)
{
	size_t	numChannels = rawData.size(), numTimeBins = rawData[0].size(), numBounces = rawData[0][0].size(),
			numRowsLaser = rawData[0][0][0].size(), numColsLaser = rawData[0][0][0][0].size(),
			numRowsCamera = rawData[0][0][0][0][0].size(), numColsCamera = rawData[0][0][0][0][0][0].size();

	assert(numChannels == 1);
	data.resize(numTimeBins * numColsCamera * numRowsCamera * numColsLaser * numRowsLaser, 0);

	#pragma omp parallel for
	for (int t = 0; t < static_cast<int>(numTimeBins); ++t)
	{
		for (size_t b = 0; b < numBounces; ++b)
		{
			for (size_t cX = 0; cX < numColsCamera; ++cX)
			{
				for (size_t cY = 0; cY < numRowsCamera; ++cY)
				{
					for (size_t lX = 0; lX < numColsLaser; ++lX)
					{
						for (size_t lY = 0; lY < numRowsLaser; ++lY)
						{
							data[lY * numColsLaser * numRowsCamera * numColsCamera * numTimeBins +
								lX * numRowsCamera * numColsCamera * numTimeBins +
								cY * numColsCamera * numTimeBins +
								cX * numTimeBins + t] += static_cast<float>(rawData[0][t][b][lY][lX][cY][cX]);
						}
					}
				}
			}
		}
	}

	dims = { numRowsLaser, numColsLaser, numRowsCamera, numColsCamera, numTimeBins };
}

inline bool TalHDF5Reader::read(const std::string& filename, NLosData& nlosData)
{
	//['H', 'H_format', 'delta_t', 'laser_grid_format', 'laser_grid_normals',
	//'laser_grid_xyz', 'laser_xyz', 'scene_info', 'sensor_grid_format', 'sensor_grid_normals',
	//'sensor_grid_xyz', 'sensor_xyz', 't_accounts_first_and_last_bounces', 't_start',
	//'volume_format']

	auto file = HighFive::File(filename, HighFive::File::ReadOnly);
	if (!file.exist("sensor_grid_xyz") ||
		!file.exist("sensor_grid_normals") ||
		!file.exist("sensor_xyz") ||
		!file.exist("laser_grid_normals") ||
		!file.exist("laser_grid_xyz") ||
		!file.exist("laser_xyz") ||
		!file.exist("t_start") ||
		!file.exist("delta_t") ||
		!file.exist("H_format") ||
		!file.exist("H") ||
		!file.exist("t_accounts_first_and_last_bounces"))
		return false;

	std::vector<float> data;

	auto dataset = file.getDataSet("sensor_grid_normals");
	setUp(nlosData._cameraGridNormals, dataset.read<std::vector<std::vector<std::vector<float>>>>());

	dataset = file.getDataSet("sensor_grid_xyz");
	setUp(nlosData._cameraGridPositions, dataset.read<std::vector<std::vector<std::vector<float>>>>());

	dataset = file.getDataSet("sensor_xyz");
	setUp(nlosData._cameraPosition, dataset.read<std::vector<float>>());

	nlosData._cameraGridSize = glm::abs(nlosData._cameraGridPositions.front() - nlosData._cameraGridPositions.back());

	dataset = file.getDataSet("t_start");
	nlosData._t0 = dataset.read<float>();

	dataset = file.getDataSet("delta_t");
	nlosData._deltaT = dataset.read<float>();

	dataset = file.getDataSet("laser_grid_normals");
	setUp(nlosData._laserGridNormals, dataset.read<std::vector<std::vector<std::vector<float>>>>());

	dataset = file.getDataSet("laser_grid_xyz");
	setUp(nlosData._laserGridPositions, dataset.read<std::vector<std::vector<std::vector<float>>>>());

	dataset = file.getDataSet("laser_xyz");
	setUp(nlosData._laserPosition, dataset.read<std::vector<float>>());

	nlosData._laserGridSize = glm::abs(nlosData._laserGridPositions.front() - nlosData._laserGridPositions.back());

	dataset = file.getDataSet("H_format");
	glm::uint H_format = dataset.read<glm::uint>();
	nlosData._isConfocal = H_format == 1 || H_format == 3 || H_format == 10 || H_format == 12;
	nlosData._discardFirstLastBounces = !file.getDataSet("t_accounts_first_and_last_bounces").read<bool>();
	nlosData._wallWidth = nlosData._cameraGridPositions.back().x;

	if (nlosData._isConfocal)
	{
		dataset = file.getDataSet("H");
		auto dims = dataset.getDimensions();

		if (dims.size() == 5)
			setUp(nlosData._data, dataset.read<std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>>(), nlosData._dims);
		else if (dims.size() == 4)
			setUp(nlosData._data, dataset.read<std::vector<std::vector<std::vector<std::vector<float>>>>>(), nlosData._dims);
		else if (dims.size() == 3)
			setUp(nlosData._data, dataset.read<std::vector<std::vector<std::vector<float>>>>(), nlosData._dims);
		else
			throw std::runtime_error("NLosData: Invalid dimensions for confocal data.");
	}
	else
	{
		dataset = file.getDataSet("data");
		setUp(nlosData._data, dataset.read<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>>>>(), nlosData._dims);
	}

	// Hidden geometry
	{
		glm::vec3 hiddenGeometryMin = nlosData._cameraGridPositions.front();
		glm::vec3 hiddenGeometryMax = nlosData._cameraGridPositions.back();
		if (glm::epsilonEqual(hiddenGeometryMax.x, hiddenGeometryMin.x, glm::epsilon<float>()))
			hiddenGeometryMax.x += 2.5f; 
		else if (glm::epsilonEqual(hiddenGeometryMax.y, hiddenGeometryMin.y, glm::epsilon<float>()))
			hiddenGeometryMax.y += 2.5f; 
		else if (glm::epsilonEqual(hiddenGeometryMax.z, hiddenGeometryMin.z, glm::epsilon<float>()))
			hiddenGeometryMax.z += 2.5f; 

		nlosData._hiddenGeometry = AABB(hiddenGeometryMin, hiddenGeometryMax);
	}

	return true;
}
