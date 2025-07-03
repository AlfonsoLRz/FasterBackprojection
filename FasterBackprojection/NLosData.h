#pragma once

#include <highfive/H5DataSet.hpp>

#include "AABB.h"
#include "transient_utils.cuh"

using Complex = std::complex<float>;

class NLosData
{
	friend class TransientReconstruction;
	friend class TransientParameters;
	friend class NLosDataVisualizer;
	friend class DrawEllipsoids;

protected:
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
	void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& rawData);
	void setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>>>& rawData);

	bool loadBinaryFile(const std::string& filename);
	bool saveBinaryFile(const std::string& filename) const;

    void padIntensity(std::vector<Complex>& paddedIntensity, size_t padding, const std::string& mode, size_t timeDim = 0) const;

public:
	NLosData(const TransientParameters& transientParams);
	NLosData(const std::string& filename, bool saveBinary = true, bool useBinary = true);
	virtual ~NLosData();

	void filter_H_cuda(float wl_mean, float wl_sigma, const std::string& border = "edge");

	void toGpu(ReconstructionInfo& recInfo, ReconstructionBuffers& recBuffers);

	float* getTimeSlice(glm::uint t);
	void saveImages(const std::string& outPath);
};

template <typename T>
void NLosData::cast(const std::vector<float>& suppData, size_t elementCount, T& buffer)
{
	for (int idx = 0; idx < static_cast<int>(elementCount); ++idx)
		buffer[idx] = suppData[idx];
}

