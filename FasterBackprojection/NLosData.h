#pragma once

#include <highfive/H5DataSet.hpp>

#include "AABB.h"
#include "transient_utils.cuh"

class NLosData
{
	friend class TransientReconstruction;
	friend class TransientParameters;
	friend class NLosDataVisualizer;
	friend class DrawEllipsoids;
	friend class Laser;

public:
	std::vector<float>		_data;
	std::vector<size_t>		_dims;

	std::vector<glm::vec3>	_cameraGridPositions;
	std::vector<glm::vec3>	_cameraGridNormals;
	glm::vec3				_cameraPosition;
	glm::vec2				_cameraGridSize;

	std::vector<glm::vec3>	_laserGridPositions;
	std::vector<glm::vec3>	_laserGridNormals;
	glm::vec3				_laserPosition;
	glm::vec2				_laserGridSize;

	AABB					_hiddenGeometry;

	glm::uint				_temporalResolution;
	float					_deltaT, _t0;

	bool					_isConfocal;

protected:
	template<typename T>
	static void cast(const std::vector<float>& suppData, size_t elementCount, T& buffer);
	static void expandData(const HighFive::DataSet& dataset, std::vector<float>& suppData);
	static void setUp(glm::vec2& data, const std::vector<double>& rawData);
	static void setUp(glm::vec3& data, const std::vector<double>& rawData);
	static void setUp(std::vector<glm::vec3>& data, const std::vector<std::vector<std::vector<double>>>& rawData);

	// Confocal
	void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& rawData);
	void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<double>>>>& rawData);
	// Exhaustive
	void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>>>& rawData);

	void loadMat(const std::string& filename);
	bool loadBinaryFile(const std::string& filename);
	bool saveBinaryFile(const std::string& filename) const;

public:
	NLosData(const std::string& filename, bool saveBinary = true, bool useBinary = true);
	virtual ~NLosData();

	void toGpu(ReconstructionInfo& recInfo, ReconstructionBuffers& recBuffers, const TransientParameters& transientParameters);

	void saveImages(const std::string& outPath);
};

template <typename T>
void NLosData::cast(const std::vector<float>& suppData, size_t elementCount, T& buffer)
{
	for (int idx = 0; idx < static_cast<int>(elementCount); ++idx)
		buffer[idx] = suppData[idx];
}

