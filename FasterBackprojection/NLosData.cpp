#include "stdafx.h"
#include "NLosData.h"

#include <matio.h> 
#include <print>

#include "ChronoUtilities.h"
#include "CudaHelper.h"
#include "MatFkReader.h"
#include "math.cuh"
#include "MatLCTReader.h"
#include "progressbar.hpp"
#include "TalHDF5Reader.h"
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
		MatLCTReader lctReader;
		if (!lctReader.read(filename, *this))
		{
			MatFkReader fkReader;
			if (!fkReader.read(filename, *this))
				throw std::runtime_error("NLosData: Failed to read the file: " + filename);
		}
	}
	else if (filename.find("hdf5") != std::string::npos)
	{
		auto file = HighFive::File(filename, HighFive::File::ReadOnly);
		if (!loadNLOSFile(file))
		{
			TalHDF5Reader talReader;
			if (!talReader.read(filename, *this))
				throw std::runtime_error("NLosData: Failed to read the file: " + filename);
		}
	}

	swapXYZOrder();

	if (saveBinary && !saveBinaryFile(binaryFile))
		std::cerr << "NLosData: Failed to save binary file: " << binaryFile << '\n';
}

NLosData::~NLosData() = default;

void NLosData::downsampleSpace(glm::uint times)
{
	ChronoUtilities::startTimer();

	std::vector<size_t> newDims(_dims);
	for (size_t& dim: newDims)
		dim /= times;
	newDims.back() = _dims.back(); // Keep the time dimension unchanged

	size_t numTimeBins = _dims.back();
	std::vector<float> downsampledData(_data.size() / times);

	if (_dims.size() == 3) // Confocal
	{
		for (size_t x = 0; x < _dims[0]; x += times)
		{
			for (size_t y = 0; y < _dims[1]; y += times)
			{
				for (size_t t = 0; t < numTimeBins; ++t)
				{
					size_t downsampledIndex = (x / times) * newDims[1] * numTimeBins + (y / times) * numTimeBins + t;
					float sum = 0.0f;
					for (size_t i = 0; i < times; ++i)
					{
						if (x + i < _dims[0] && y + i < _dims[1])
							sum += _data[(x + i) * _dims[1] * numTimeBins + (y + i) * numTimeBins + t];
					}
					downsampledData[downsampledIndex] = sum / static_cast<float>(times * times);
				}
			}
		}
	}

	_data = std::move(downsampledData);
	_dims = newDims;

	std::cout << "NLosData: Downsampled spatial dimensions by a factor of " << times << ". \n";
	std::cout << "Time taken to downsample: " << ChronoUtilities::getElapsedTime() << " milliseconds.\n";
}

void NLosData::downsampleTime(glm::uint times)
{
	ChronoUtilities::startTimer();

	size_t numSlices = _data.size() / _dims.back(), numTimeBins = _dims.back();
	size_t newTimeDimension = numTimeBins / times;
	std::vector<float> downsampledData(_data.size() / times);

	#pragma omp parallel for
	for (size_t slice = 0; slice < numSlices; ++slice)
	{
		for (size_t t = 0; t < _dims.back(); t += times)
		{
			size_t downsampledIndex = slice * newTimeDimension + t / times;
			float sum = 0.0f;
			for (size_t i = 0; i < times; ++i)
			{
				if (t + i < _dims.back())
					sum += _data[slice * _dims.back() + t + i];
			}

			downsampledData[downsampledIndex] = sum / static_cast<float>(times);
		}
	}

	_data = std::move(downsampledData);
	_dims.back() = newTimeDimension;
	_deltaT *= static_cast<float>(times);

	std::cout << "NLosData: Downsampled time dimension by a factor of " << times << ". New temporal resolution: " 
				<< _dims.back() << ", new deltaT: " << _deltaT << '\n';
	std::cout << "Time taken to downsample: " << ChronoUtilities::getElapsedTime() << " milliseconds.\n";
}

void NLosData::toGpu(ReconstructionInfo& recInfo, ReconstructionBuffers& recBuffers, const TransientParameters& transientParameters)
{
	CudaHelper::initializeBuffer(recBuffers._intensity, _data.size(), _data.data());
	if (!_cameraGridPositions.empty())
		CudaHelper::initializeBuffer(recBuffers._sensorTargets, _cameraGridPositions.size(), _cameraGridPositions.data());
	if (!_laserGridPositions.empty())
		CudaHelper::initializeBuffer(recBuffers._laserTargets, _laserGridPositions.size(), _laserGridPositions.data());
	if (!_laserGridNormals.empty())
		CudaHelper::initializeBuffer(recBuffers._laserTargetsNormals, _laserGridNormals.size(), _laserGridNormals.data());

	recInfo._sensorPosition = _cameraPosition;
	recInfo._numSensorTargets = static_cast<glm::uint>(_cameraGridPositions.size());

	recInfo._laserPosition = _laserPosition;
	recInfo._numLaserTargets = static_cast<glm::uint>(_laserGridPositions.size());

	recInfo._numTimeBins = static_cast<glm::uint>(_dims.back());
	recInfo._timeStep = _deltaT;
	recInfo._timeOffset = _t0;
	recInfo._captureSystem = _isConfocal ? CaptureSystem::Confocal : CaptureSystem::Exhaustive;
	recInfo._discardFirstLastBounces = _discardFirstLastBounces;

	//recInfo._relayWallNormal = _cameraGridNormals.empty() ? glm::vec3(.0, -1.0f, .0f) : _cameraGridNormals.front();
	//recInfo._relayWallMinPosition = glm::vec3(-_cameraGridSize.x / 2.0f, .0f, -_cameraGridSize.y / 2.0f);
	//recInfo._relayWallSize = glm::vec3(_cameraGridSize.x, .0f, _cameraGridSize.y);

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

void NLosData::swapXYZOrder()
{
	if (_cameraGridPositions.empty())
		return;

	const glm::vec3 sensorGridPositionFirst = _cameraGridPositions.front(), sensorGridPositionLast = _cameraGridPositions.back();

	if (glm::epsilonEqual(sensorGridPositionFirst.y, sensorGridPositionLast.y, glm::epsilon<float>()))
		return;

	// Swap Y and Z
	_cameraPosition = glm::vec3(_cameraPosition.x, _cameraPosition.z, _cameraPosition.y);
	_laserPosition = glm::vec3(_laserPosition.x, _laserPosition.z, _laserPosition.y);

	for (auto& pos : _cameraGridPositions)
		pos = glm::vec3(pos.x, pos.z, pos.y);

	for (auto& pos : _laserGridPositions)
		pos = glm::vec3(pos.x, pos.z, pos.y);

	for (auto& normal : _cameraGridNormals)
		normal = glm::vec3(normal.x, normal.z, normal.y);

	for (auto& normal : _laserGridNormals)
		normal = glm::vec3(normal.x, normal.z, normal.y);

	_hiddenGeometry = AABB(
		glm::vec3(_hiddenGeometry.minPoint().x, _hiddenGeometry.minPoint().z, _hiddenGeometry.minPoint().y),
		glm::vec3(_hiddenGeometry.maxPoint().x, _hiddenGeometry.maxPoint().z, _hiddenGeometry.maxPoint().y)
	);
}

bool NLosData::loadNLOSFile(const HighFive::File& file)
{
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
		return false;

	std::vector<float> data;

	auto dataset = file.getDataSet("cameraGridNormals");
	setUp(_cameraGridNormals, dataset.read<std::vector<std::vector<std::vector<double>>>>());

	dataset = file.getDataSet("cameraGridPositions");
	setUp(_cameraGridPositions, dataset.read<std::vector<std::vector<std::vector<double>>>>());

	dataset = file.getDataSet("cameraPosition");
	setUp(_cameraPosition, dataset.read<std::vector<double>>());

	dataset = file.getDataSet("cameraGridSize");
	setUp(_cameraGridSize, dataset.read<std::vector<double>>());

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
	_wallWidth = glm::abs(_cameraGridPositions.back().x);

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

	return true;
}

bool NLosData::loadBinaryFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())
		return false;

	file.read(reinterpret_cast<char*>(&_cameraPosition), sizeof(glm::vec3));
	file.read(reinterpret_cast<char*>(&_laserPosition), sizeof(glm::vec3));
	file.read(reinterpret_cast<char*>(&_deltaT), sizeof(float));
	file.read(reinterpret_cast<char*>(&_t0), sizeof(float));
	file.read(reinterpret_cast<char*>(&_wallWidth), sizeof(float));
	file.read(reinterpret_cast<char*>(&_isConfocal), sizeof(bool));
	file.read(reinterpret_cast<char*>(&_hiddenGeometry), sizeof(AABB));
	file.read(reinterpret_cast<char*>(&_zOffset), sizeof(glm::uint));
	file.read(reinterpret_cast<char*>(&_discardFirstLastBounces), sizeof(bool));

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
	file.write(reinterpret_cast<const char*>(&_deltaT), sizeof(float));
	file.write(reinterpret_cast<const char*>(&_t0), sizeof(float));
	file.write(reinterpret_cast<const char*>(&_wallWidth), sizeof(float));
	file.write(reinterpret_cast<const char*>(&_isConfocal), sizeof(bool));
	file.write(reinterpret_cast<const char*>(&_hiddenGeometry), sizeof(AABB));
	file.write(reinterpret_cast<const char*>(&_zOffset), sizeof(glm::uint));
	file.write(reinterpret_cast<const char*>(&_discardFirstLastBounces), sizeof(bool));

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
