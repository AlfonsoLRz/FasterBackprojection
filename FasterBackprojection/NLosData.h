#pragma once

#include <matio.h>
#include <highfive/highfive.hpp>
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

	float					_deltaT = .0f, _t0 = .0f, _wallWidth = .0f;

	bool					_isConfocal = true;
	bool					_discardFirstLastBounces = true;

	glm::uint				_zOffset = 0;

protected:
	static void setUp(glm::vec2& data, const std::vector<double>& rawData);
	static void setUp(glm::vec3& data, const std::vector<double>& rawData);
	static void setUp(std::vector<glm::vec3>& data, const std::vector<std::vector<std::vector<double>>>& rawData);

	// Confocal
	void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& rawData);
	void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<double>>>>& rawData);
	// Exhaustive
	void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>>>& rawData);

	void swapXYZOrder();

	bool loadNLOSFile(const HighFive::File& file);

	bool loadBinaryFile(const std::string& filename);
	bool saveBinaryFile(const std::string& filename) const;

public:
	NLosData(const std::string& filename, bool saveBinary = true, bool useBinary = true);
	virtual ~NLosData();

	void downsampleSpace(glm::uint times);
	void downsampleTime(glm::uint times);
	void toGpu(ReconstructionInfo& recInfo, ReconstructionBuffers& recBuffers, const TransientParameters& transientParameters);

	void discardDistanceToSensorAndLaser();
	void saveImages(const std::string& outPath);
};