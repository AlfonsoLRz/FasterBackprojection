#include "stdafx.h"
#include "NLosData.h"

#include <matio.h> 
#include <highfive/highfive.hpp>
#include <print>

#include "CudaHelper.h"
#include "progressbar.hpp"
#include "TransientImage.h"
#include "TransientParameters.h"

//

NLosData::NLosData(const std::string& filename, bool saveBinary, bool useBinary)
{
	const std::string binaryFile = filename.substr(0, filename.find_last_of('.')) + ".nlos";
	if (useBinary)
		if (loadBinaryFile(binaryFile))
			return; 

	if (filename.find(".mat") != std::string::npos)
	{
		loadMat(filename);
	}
	else
	{
		// Open the file
		auto file = HighFive::File(filename, HighFive::File::ReadOnly);
		if (!file.exist("cameraGridPositions") ||
			!file.exist("cameraGridNormals") ||
			!file.exist("cameraPosition") ||
			!file.exist("laserGridNormals") ||
			!file.exist("laserGridPositions") ||
			!file.exist("laserPosition") ||
			!file.exist("t") ||
			!file.exist("t0") ||
			!file.exist("deltaT") ||
			!file.exist("isConfocal") ||
			!file.exist("data"))
			throw std::runtime_error("NLosData: Missing required datasets in the file.");

		std::vector<float> data;

		auto dataset = file.getDataSet("cameraGridNormals");
		setUp(_cameraGridNormals, dataset.read<std::vector<std::vector<std::vector<double>>>>());

		dataset = file.getDataSet("cameraGridPositions");
		setUp(_cameraGridPositions, dataset.read<std::vector<std::vector<std::vector<double>>>>());

		dataset = file.getDataSet("cameraPosition");
		setUp(_cameraPosition, dataset.read<std::vector<double>>());

		dataset = file.getDataSet("cameraGridSize");
		setUp(_cameraGridSize, dataset.read<std::vector<double>>());

		dataset = file.getDataSet("t");
		_temporalResolution = dataset.read<glm::uint>();

		dataset = file.getDataSet("t0");
		_t0 = static_cast<float>(dataset.read<int>());

		dataset = file.getDataSet("deltaT");
		_deltaT = static_cast<float>(dataset.read<double>());

		dataset = file.getDataSet("laserGridNormals");
		setUp(_laserGridNormals, dataset.read<std::vector<std::vector<std::vector<double>>>>());

		dataset = file.getDataSet("laserGridPositions");
		setUp(_laserGridPositions, dataset.read<std::vector<std::vector<std::vector<double>>>>());

		dataset = file.getDataSet("laserPosition");
		setUp(_laserPosition, dataset.read<std::vector<double>>());

		dataset = file.getDataSet("laserGridSize");
		setUp(_laserGridSize, dataset.read<std::vector<double>>());

		// Hidden geometry
		{
			dataset = file.getDataSet("hiddenVolumePosition");
			std::vector<double> volumePosition = dataset.read<std::vector<double>>();

			std::vector<double> volumeSize;
			dataset = file.getDataSet("hiddenVolumeSize");
			if (dataset.getDimensions().empty())
				volumeSize = { dataset.read<double>(), 0.1, dataset.read<double>() };
			else
				volumeSize = dataset.read<std::vector<double>>();

			glm::vec3 hiddenGeometryMin = glm::vec3(
				static_cast<float>(volumePosition[0]) - static_cast<float>(volumeSize[0]) / 2.0f,
				static_cast<float>(volumePosition[1]) - static_cast<float>(volumeSize[1]) / 2.0f,
				static_cast<float>(volumePosition[2]) - static_cast<float>(volumeSize[2]) / 2.0f
			);
			glm::vec3 hiddenGeometryMax = glm::vec3(
				static_cast<float>(volumePosition[0]) + static_cast<float>(volumeSize[0]) / 2.0f,
				static_cast<float>(volumePosition[1]) + static_cast<float>(volumeSize[1]) / 2.0f,
				static_cast<float>(volumePosition[2]) + static_cast<float>(volumeSize[2]) / 2.0f
			);
			_hiddenGeometry = AABB(hiddenGeometryMin, hiddenGeometryMax);
		}

		dataset = file.getDataSet("isConfocal");
		_isConfocal = static_cast<bool>(dataset.read<glm::uint>());
		_discardFirstLastBounces = false;

		if (_isConfocal)
		{
			dataset = file.getDataSet("data");
			auto dims = dataset.getDimensions();

			if (dims.size() == 5)
				setUp(_data, dataset.read<std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>>());
			else if (dims.size() == 4)
				setUp(_data, dataset.read<std::vector<std::vector<std::vector<std::vector<double>>>>>());
			else
				throw std::runtime_error("NLosData: Invalid dimensions for confocal data.");
		}
		else
		{
			dataset = file.getDataSet("data");
			setUp(_data, dataset.read<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>>>>());
		}
	}

	if (saveBinary && !saveBinaryFile(binaryFile))
		std::cerr << "NLosData: Failed to save binary file: " << binaryFile << '\n';
}

NLosData::~NLosData() = default;

void NLosData::toGpu(ReconstructionInfo& recInfo, ReconstructionBuffers& recBuffers, const TransientParameters& transientParameters)
{
	CudaHelper::initializeBufferGPU(recBuffers._intensity, _data.size(), _data.data());
	if (!_cameraGridPositions.empty())
		CudaHelper::initializeBufferGPU(recBuffers._sensorTargets, _cameraGridPositions.size(), _cameraGridPositions.data());
	if (!_laserGridPositions.empty())
		CudaHelper::initializeBufferGPU(recBuffers._laserTargets, _laserGridPositions.size(), _laserGridPositions.data());
	if (!_laserGridNormals.empty())
		CudaHelper::initializeBufferGPU(recBuffers._laserTargetsNormals, _laserGridNormals.size(), _laserGridNormals.data());

	recInfo._sensorPosition = _cameraPosition;
	recInfo._numSensorTargets = static_cast<glm::uint>(_cameraGridPositions.size());

	recInfo._laserPosition = _laserPosition;
	recInfo._numLaserTargets = static_cast<glm::uint>(_laserGridPositions.size());

	recInfo._numTimeBins = _temporalResolution;
	recInfo._timeStep = _deltaT;
	recInfo._timeOffset = _t0;
	recInfo._captureSystem = _isConfocal ? CaptureSystem::Confocal : CaptureSystem::Exhaustive;
	recInfo._discardFirstLastBounces = _discardFirstLastBounces;

	recInfo._relayWallNormal = _cameraGridNormals.empty() ? glm::vec3(.0, -1.0f, .0f) : _cameraGridNormals.front();
	recInfo._relayWallMinPosition = glm::vec3(-_cameraGridSize.x / 2.0f, .0f, -_cameraGridSize.y / 2.0f);
	recInfo._relayWallSize = glm::vec3(_cameraGridSize.x, .0f, _cameraGridSize.y);

	recInfo._hiddenVolumeMin = _hiddenGeometry.minPoint();
	recInfo._hiddenVolumeMax = _hiddenGeometry.maxPoint();
	recInfo._hiddenVolumeSize = _hiddenGeometry.size();

	recInfo._voxelResolution = transientParameters._voxelResolution;
	recInfo._hiddenVolumeVoxelSize = _hiddenGeometry.size() / glm::vec3(recInfo._voxelResolution);
}

void NLosData::saveImages(const std::string& outPath)
{
	glm::uint sliceSize = static_cast<glm::uint>(_data.size()) / _dims[0];
	if (_dims.size() > 3)
		throw std::runtime_error("NLosData: Invalid dimensions for saving images.");

	// Create output directory if it does not exist
	if (!std::filesystem::exists(outPath))
		std::filesystem::create_directories(outPath);

	progressbar bar(_dims[0], true);
	for (size_t idx = 0; idx < _dims[0]; ++idx)
	{
		TransientImage transientImage(_dims[1], _dims[2]);
		transientImage.save
		(
			outPath + "transient_" + std::to_string(idx) + ".png", _data.data() + idx * sliceSize, glm::uvec2(_dims[1], _dims[2]) * 1u, 
			1, 0, true
		);
		bar.update();
	}
}

//

void NLosData::expandData(const HighFive::DataSet& dataset, std::vector<float>& suppData)
{
	if (suppData.size() < dataset.getElementCount())
		suppData.resize(dataset.getElementCount());
}

void NLosData::setUp(glm::vec2& data, const std::vector<double>& rawData)
{
	if (rawData.size() != 2)
		throw std::runtime_error("NLOSata: Invalid raw data size for glm::vec2.");
	data = glm::vec2(static_cast<float>(rawData[0]), static_cast<float>(rawData[1]));
}

void NLosData::setUp(glm::vec3& data, const std::vector<double>& rawData)
{
	if (rawData.size() != 3)
		throw std::runtime_error("NLOSata: Invalid raw data size for glm::vec3.");
	data = glm::vec3(static_cast<float>(rawData[0]), static_cast<float>(rawData[1]), static_cast<float>(rawData[2]));
}

void NLosData::setUp(std::vector<glm::vec3>& data, const std::vector<std::vector<std::vector<double>>>& rawData)
{
	size_t width = rawData[0].size(), height = rawData[0][0].size();
	data.resize(width * height);

	#pragma omp parallel for
	for (int x = 0; x < static_cast<int>(width); ++x)
	{
		for (size_t y = 0; y < height; ++y)
		{
			data[x * width + y] = glm::vec3(
				static_cast<float>(rawData[0][x][y]),
				static_cast<float>(rawData[1][x][y]),
				static_cast<float>(rawData[2][x][y]));
		}
	}
}

void NLosData::setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& rawData)
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

	_dims = { numRows, numCols, numTimeBins };
}

void NLosData::setUp(std::vector<float>& data, const std::vector<std::vector<std::vector<std::vector<double>>>>& rawData)
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

	_dims = { numRows, numCols, numTimeBins };
}

void NLosData::setUp(std::vector<float>& data,
                     const std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>>>& rawData)
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

	_dims = { numRowsLaser, numColsLaser, numRowsCamera, numColsCamera, numTimeBins };
}

void NLosData::loadMat(const std::string& filename)
{
	// Open the .mat file
	mat_t* matFile = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
	if (!matFile) {
		std::cerr << "Error opening .mat file!" << '\n';
	}

	matvar_t* matvar;
	while ((matvar = Mat_VarReadNextInfo(matFile)) != NULL) {
		std::cout << "Found variable: " << matvar->name << std::endl;
		Mat_VarFree(matvar);
	}

	if (loadLCTMat(matFile))
		return;

	throw std::runtime_error("NLosData: Failed to load LCT data from .mat file.");
}

bool NLosData::loadLCTMat(mat_t* matFile)
{
	matvar_t* rectDataVar = Mat_VarRead(matFile, "rect_data");
	if (!rectDataVar)
	{
		Mat_Close(matFile);
		return false;
	}

	glm::uint16_t* rectData = static_cast<glm::uint16_t*>(rectDataVar->data);

	size_t globalSize = 1;
	_dims.resize(rectDataVar->rank);
	for (int i = 0; i < rectDataVar->rank; ++i)
	{
		_dims[i] = rectDataVar->dims[i];
		globalSize *= rectDataVar->dims[i];
	}

	_data.resize(globalSize);
	#pragma omp parallel for
	for (size_t x = 0; x < _dims[0]; ++x)
	{
		for (size_t y = 0; y < _dims[1]; ++y)
		{
			for (size_t t = 0; t < _dims[2]; ++t)
			{
				size_t idx = y * _dims[1] * _dims[2] + x * _dims[2] + t;
				_data[idx] = static_cast<float>(rectData[t * _dims[0] * _dims[1] + x * _dims[0] + y]);
			}
		}
	}

	matvar_t* widthVar = Mat_VarRead(matFile, "width");
	if (!widthVar)
	{
		Mat_VarFree(rectDataVar);
		Mat_Close(matFile);
		return false;
	}

	double* temporalWidth = static_cast<double*>(widthVar->data);
	_temporalWidth = static_cast<float>(temporalWidth[0]);

	// Fill with fake laser and sensor grid positions and normals
	const glm::uint numRelayWallTargets = _dims[0] * _dims[1];
	_cameraGridPositions.resize(numRelayWallTargets);
	_cameraGridNormals.resize(numRelayWallTargets);
	_laserGridPositions.resize(numRelayWallTargets);
	_laserGridNormals.resize(numRelayWallTargets);
	_cameraPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	_laserPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	_cameraGridSize = glm::vec2(_dims[1], _dims[0]);
	_laserGridSize = glm::vec2(_dims[1], _dims[0]);
	_discardFirstLastBounces = true;
	_isConfocal = true;
	_temporalResolution = _dims.back();
	_deltaT = 4e-12;
	_t0 = .0f;

	for (size_t i = 0; i < _cameraGridPositions.size(); ++i)
	{
		_cameraGridPositions[i] = glm::vec3(static_cast<float>(i / _dims[0]), 0.0f, static_cast<float>(i % _dims[0]));
		_cameraGridNormals[i] = glm::vec3(0.0f, -1.0f, 0.0f);

		_laserGridPositions[i] = glm::vec3(static_cast<float>(i / _dims[0]), 0.0f, static_cast<float>(i % _dims[0]));
		_laserGridNormals[i] = glm::vec3(0.0f, -1.0f, 0.0f);
	}

	Mat_VarFree(widthVar);

	return true;
}

bool NLosData::loadBinaryFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())
		return false;

	file.read(reinterpret_cast<char*>(&_cameraPosition), sizeof(glm::vec3));
	file.read(reinterpret_cast<char*>(&_laserPosition), sizeof(glm::vec3));
	file.read(reinterpret_cast<char*>(&_temporalResolution), sizeof(glm::uint));
	file.read(reinterpret_cast<char*>(&_deltaT), sizeof(float));
	file.read(reinterpret_cast<char*>(&_t0), sizeof(float));
	file.read(reinterpret_cast<char*>(&_temporalWidth), sizeof(float));
	file.read(reinterpret_cast<char*>(&_isConfocal), sizeof(bool));
	file.read(reinterpret_cast<char*>(&_hiddenGeometry), sizeof(AABB));

	size_t numDims;
	file.read(reinterpret_cast<char*>(&numDims), sizeof(size_t));
	_dims.resize(numDims);
	file.read(reinterpret_cast<char*>(_dims.data()), numDims * sizeof(size_t));

	size_t numCameraGridPositions;
	file.read(reinterpret_cast<char*>(&numCameraGridPositions), sizeof(size_t));
	_cameraGridPositions.resize(numCameraGridPositions);
	_cameraGridNormals.resize(numCameraGridPositions);
	file.read(reinterpret_cast<char*>(_cameraGridPositions.data()), numCameraGridPositions * sizeof(glm::vec3));
	file.read(reinterpret_cast<char*>(_cameraGridNormals.data()), numCameraGridPositions * sizeof(glm::vec3));
	file.read(reinterpret_cast<char*>(&_cameraGridSize), sizeof(glm::vec2));

	size_t numLaserGridPositions;
	file.read(reinterpret_cast<char*>(&numLaserGridPositions), sizeof(size_t));
	_laserGridPositions.resize(numLaserGridPositions);
	_laserGridNormals.resize(numLaserGridPositions);
	file.read(reinterpret_cast<char*>(_laserGridPositions.data()), numLaserGridPositions * sizeof(glm::vec3));
	file.read(reinterpret_cast<char*>(_laserGridNormals.data()), numLaserGridPositions * sizeof(glm::vec3));
	file.read(reinterpret_cast<char*>(&_laserGridSize), sizeof(glm::vec2));

	size_t numValues;
	file.read(reinterpret_cast<char*>(&numValues), sizeof(size_t));
	_data.resize(numValues);
	file.read(reinterpret_cast<char*>(_data.data()), numValues * sizeof(float));

	file.close();

	return true;
}

bool NLosData::saveBinaryFile(const std::string& filename) const
{
	std::cout << filename << "\n";
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open())
	{
		std::cerr << "NLosData: Failed to open file for saving binary data: " << filename << "\n";
		return false;
	}

	file.write(reinterpret_cast<const char*>(&_cameraPosition), sizeof(glm::vec3));
	file.write(reinterpret_cast<const char*>(&_laserPosition), sizeof(glm::vec3));
	file.write(reinterpret_cast<const char*>(&_temporalResolution), sizeof(glm::uint));
	file.write(reinterpret_cast<const char*>(&_deltaT), sizeof(float));
	file.write(reinterpret_cast<const char*>(&_t0), sizeof(float));
	file.write(reinterpret_cast<const char*>(&_temporalWidth), sizeof(float));
	file.write(reinterpret_cast<const char*>(&_isConfocal), sizeof(bool));
	file.write(reinterpret_cast<const char*>(&_hiddenGeometry), sizeof(AABB));

	size_t numDims = _dims.size();
	file.write(reinterpret_cast<const char*>(&numDims), sizeof(size_t));
	file.write(reinterpret_cast<const char*>(_dims.data()), numDims * sizeof(size_t));

	size_t numCameraGridPositions = _cameraGridPositions.size();
	file.write(reinterpret_cast<const char*>(&numCameraGridPositions), sizeof(size_t));
	file.write(reinterpret_cast<const char*>(_cameraGridPositions.data()), numCameraGridPositions * sizeof(glm::vec3));
	file.write(reinterpret_cast<const char*>(_cameraGridNormals.data()), numCameraGridPositions * sizeof(glm::vec3));
	file.write(reinterpret_cast<const char*>(&_cameraGridSize), sizeof(glm::vec2));

	size_t numLaserGridPositions = _laserGridPositions.size();
	file.write(reinterpret_cast<const char*>(&numLaserGridPositions), sizeof(size_t));
	file.write(reinterpret_cast<const char*>(_laserGridPositions.data()), numLaserGridPositions * sizeof(glm::vec3));
	file.write(reinterpret_cast<const char*>(_laserGridNormals.data()), numLaserGridPositions * sizeof(glm::vec3));
	file.write(reinterpret_cast<const char*>(&_laserGridSize), sizeof(glm::vec2));

	size_t numValues = _data.size();
	file.write(reinterpret_cast<const char*>(&numValues), sizeof(size_t));
	file.write(reinterpret_cast<const char*>(_data.data()), _data.size() * sizeof(float));

	file.close();
	return true;
}
