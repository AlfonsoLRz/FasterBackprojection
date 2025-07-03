#include "stdafx.h"
#include "NLosData.h"

#include <cufft.h>
#include <highfive/highfive.hpp>
#include <print>

#include "CudaHelper.h"
#include "fourier.cuh"
#include "progressbar.hpp"
#include "TransientImage.h"
#include "TransientParameters.h"

//

NLosData::NLosData(const TransientParameters& transientParams):
	_cameraPosition(), _cameraGridSize(),
	_laserPosition(), _laserGridSize()
{
	this->_laserPosition = transientParams._laserPosition;
	this->_cameraPosition = transientParams._cameraPosition;
	this->_temporalResolution = transientParams._numTimeBins;
	this->_deltaT = transientParams._temporalResolution;
	this->_t0 = transientParams._timeOffset;
	this->_isConfocal = transientParams._captureSystem == CaptureSystem::Confocal;
}

NLosData::NLosData(const std::string& filename, bool saveBinary, bool useBinary)
{
	const std::string binaryFile = filename.substr(0, filename.find_last_of('.')) + ".nlos";
	if (useBinary)
		if (loadBinaryFile(binaryFile))
			return; // Successfully loaded binary data

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

	if (_isConfocal)
	{
		dataset = file.getDataSet("data");
		setUp(_data, dataset.read<std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>>());
	}
	else
	{
		dataset = file.getDataSet("data");
		setUp(_data, dataset.read<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>>>>());
	}

	if (saveBinary && !saveBinaryFile(binaryFile))
		std::cerr << "NLosData: Failed to save binary file: " << binaryFile << '\n';
}

NLosData::~NLosData() = default;

void NLosData::filter_H_cuda(float wl_mean, float wl_sigma, const std::string& border)
{
	using Complex = std::complex<float>;

	size_t nt = _temporalResolution;

	if (wl_sigma == 0.0f) 
	{ 
		std::print("tal.reconstruct.filter_H: wl_sigma not specified, using wl_mean / sqrt(2)");
		wl_sigma = wl_mean / std::numbers::sqrt2_v<float>;
	}

	int t_6sigma = static_cast<int>(std::round(6 * wl_sigma / _deltaT));
	if (t_6sigma % 2 == 1)
		t_6sigma += 1;

	size_t nt_pad = nt + 2 * (t_6sigma - 1);
	float t_max = _deltaT * (nt_pad - 1);
	std::vector<float> t_vals(nt_pad);

	for (size_t i = 0; i < nt_pad; ++i)
		t_vals[i] = static_cast<float>(i) * _deltaT;

	// K 
	std::vector<Complex> K(nt_pad);
	float sumGaussianEnvelope = 0.0f;
	std::vector<float> gaussianEnvelope(nt_pad);

	for (size_t i = 0; i < nt_pad; ++i) 
	{
		float val = (t_vals[i] - t_max / 2.0f) / wl_sigma;
		gaussianEnvelope[i] = std::exp(-(val * val) / 2.0f);
		sumGaussianEnvelope += gaussianEnvelope[i];
	}

	for (size_t i = 0; i < nt_pad; ++i) 
		K[i] = gaussianEnvelope[i] / sumGaussianEnvelope * std::polar(1.0f, 2.0f * glm::pi<float>() * t_vals[i] / wl_mean); 

	// This prepares K for FFT since cuFFT computes the unshifted FFT.
	std::vector<Complex> K_ifftShifted(nt_pad);
	int shift = (nt_pad + 1) / 2;
	for (size_t i = 0; i < nt_pad; ++i) 
		K_ifftShifted[i] = K[(i + shift) % nt_pad];
	K = K_ifftShifted; 

	// --- Padding H on Host ---
	size_t padding = (nt_pad - nt) / 2;
	std::vector<Complex> H_pad_host;
	padIntensity(H_pad_host, padding, border, 0);

	// Calculate total elements for padded H
	size_t totalPaddedElements = 1;
	std::vector<size_t> paddedDims = _dims;
	paddedDims[0] = nt_pad; // Time dimension
	for (size_t dim : paddedDims) {
		totalPaddedElements *= dim;
	}

	size_t totalOriginalElements = 1;
	for (size_t dim : _dims) {
		totalOriginalElements *= dim;
	}

	// Memory allocation on device
	Complex* d_H_pad, * d_K, * d_HoK;
	CudaHelper::checkError(cudaMalloc((void**)&d_H_pad, totalPaddedElements * sizeof(Complex)));
	CudaHelper::checkError(cudaMalloc((void**)&d_K, nt_pad * sizeof(Complex)));
	CudaHelper::checkError(cudaMalloc((void**)&d_HoK, totalPaddedElements * sizeof(Complex)));

	// Move buffers to device
	CudaHelper::checkError(cudaMemcpy(d_H_pad, H_pad_host.data(), totalPaddedElements * sizeof(Complex), cudaMemcpyHostToDevice));
	CudaHelper::checkError(cudaMemcpy(d_K, K.data(), nt_pad * sizeof(Complex), cudaMemcpyHostToDevice));

	// CUFFT plans
	cufftHandle plan_H_fft;
	cufftHandle plan_K_fft;
	cufftHandle plan_HoK_ifft;

	// For H_fft (multi-dimensional FFT along axis 0)
	// rank: number of dimensions for the FFT (1 for 1D FFT along time_dim)
	// n: array of sizes of each dimension (nt_pad)
	// idist, odist: distance between input/output batches (stride for non-time dims)
	// istride, ostride: distance between elements in a dimension (1 for contiguous)
	// batch: number of FFTs to perform (size of non-time dims)
	int rank = 1; // 1D FFT
	int n[] = { (int)nt_pad };
	int istride = 1, ostride = 1; // Contiguous elements along time dim

	int batch = 1; // Default for 1D, will be overwritten for multi-dim H
	for (size_t i = 1; i < _dims.size(); ++i) { // Multiply all other dimensions
		batch *= _dims[i];
	}

	istride = 1; // Stride within the 1D FFT block (elements in time dim)
	ostride = 1;

	// inembed and onembed for rank = 1: array with one element, which is the physical length of this dimension
	int inembed_H[] = { (int)nt_pad };
	int onembed_H[] = { (int)nt_pad };

	// The distance between the start of consecutive 1D FFTs
	// This is the total number of elements in one (non-time) slice
	size_t dist = 1;
	for (size_t i = 1; i < _dims.size(); ++i) 
		dist *= _dims[i]; 

	// Correct for padded dimension
	dist = totalPaddedElements / nt_pad; // Total elements per time slice (across non-time dims)
	CUFFT_CHECK((cufftPlanMany(&plan_H_fft, rank, n,
		inembed_H, istride, (int)dist,
		onembed_H, ostride, (int)dist,
		CUFFT_C2C, batch)));

	CUFFT_CHECK(cufftPlan1d(&plan_K_fft, nt_pad, CUFFT_C2C, 1)); // K is always 1D
	CUFFT_CHECK(cufftPlanMany(&plan_HoK_ifft, rank, n,
		inembed_H, istride, (int)dist, // Use same strides/distances as for H_fft
		onembed_H, ostride, (int)dist,
		CUFFT_C2C, batch));

	// --- Perform FFTs ---
	std::print("Performing FFT on H...");
	CUFFT_CHECK(cufftExecC2C(plan_H_fft, (cufftComplex*)d_H_pad, (cufftComplex*)d_H_pad, CUFFT_FORWARD));

	std::print("Performing FFT on K...");
	CUFFT_CHECK(cufftExecC2C(plan_K_fft, (cufftComplex*)d_K, (cufftComplex*)d_K, CUFFT_FORWARD));

	// --- Element-wise Multiplication (CUDA Kernel) ---
	// Reshape K_fft as K_shape = (nt_pad,) + (1,) * (H.ndim - 1)
	// This implies multiplying each slice of H_fft by the same K_fft vector.
	// The kernel will broadcast d_K across the other dimensions.
	dim3 threadsPerBlock(256); // Adjust as needed
	dim3 numBlocks((totalPaddedElements + threadsPerBlock.x - 1) / threadsPerBlock.x);

	std::print("Performing element-wise multiplication...");
	//elementwise_multiply_kernel<<<numBlocks, threadsPerBlock>>>((float2*)d_H_pad, (float2*)d_K, totalPaddedElements, nt_pad);


	// --- Perform IFFT ---
	std::print("Performing IFFT on HoK...");
	CUFFT_CHECK(cufftExecC2C(plan_HoK_ifft, (cufftComplex*)d_H_pad, (cufftComplex*)d_H_pad, CUFFT_INVERSE));

	// Normalize IFFT output (cuFFT output is scaled by N)
	// Since HoK is in-place with d_H_pad, we apply normalization to d_H_pad.
	//normalize_kernel<<<numBlocks, threadsPerBlock>>>((float2*)d_H_pad, totalPaddedElements, nt_pad);


	// --- Copy Result back to Host ---
	std::vector<Complex> HoK_padded_host(totalPaddedElements);
	CudaHelper::checkError(cudaMemcpy(HoK_padded_host.data(), d_H_pad, totalPaddedElements * sizeof(Complex), cudaMemcpyDeviceToHost));

	// --- Cleanup CUDA Resources ---
	CUFFT_CHECK(cufftDestroy(plan_H_fft));
	CUFFT_CHECK(cufftDestroy(plan_K_fft));
	CUFFT_CHECK(cufftDestroy(plan_HoK_ifft));
	CudaHelper::free(d_H_pad);
	CudaHelper::free(d_K);
	CudaHelper::free(d_HoK); // This was allocated but not used as d_H_pad was in-place

	// --- Unpadding and Final Crop (Host Side) ---
	// `HoK = HoK[padding:-padding, ...]`
	std::vector<Complex> HoK_unpadded_host(totalOriginalElements);
	size_t unpadded_time_stride = 1;
	for (size_t i = 1; i < _dims.size(); ++i) {
		unpadded_time_stride *= _dims[i];
	}

	for (size_t j = 0; j < totalOriginalElements / nt; ++j) { // Iterate over non-time slices
		for (size_t i = 0; i < nt; ++i) { // Iterate over original time dimension
			size_t original_idx_padded = (i + padding) * unpadded_time_stride + j;
			size_t final_idx = i * unpadded_time_stride + j;
			HoK_unpadded_host[final_idx] = HoK_padded_host[original_idx_padded];
		}
	}

	if (border == "erase") {
		size_t erase_padding_half = padding / 2; // Python's padding//2
		for (size_t j = 0; j < totalOriginalElements / nt; ++j) {
			// Erase beginning
			for (size_t i = 0; i < erase_padding_half; ++i) {
				size_t final_idx = i * unpadded_time_stride + j;
				HoK_unpadded_host[final_idx] = Complex(0.0f, 0.0f);
			}
			// Erase end
			for (size_t i = nt - erase_padding_half; i < nt; ++i) {
				size_t final_idx = i * unpadded_time_stride + j;
				HoK_unpadded_host[final_idx] = Complex(0.0f, 0.0f);
			}
		}
	}

	//return HoK_unpadded_host;
}

void NLosData::toGpu(ReconstructionInfo& recInfo, ReconstructionBuffers& recBuffers)
{
	CudaHelper::initializeBufferGPU(recBuffers._intensity, _data.size(), _data.data());
	CudaHelper::initializeBufferGPU(recBuffers._sensorTargets, _cameraGridPositions.size(), _cameraGridPositions.data());
	CudaHelper::initializeBufferGPU(recBuffers._laserTargets, _laserGridPositions.size(), _laserGridPositions.data());

	recInfo._sensorPosition = _cameraPosition;
	recInfo._numSensorTargets = static_cast<glm::uint>(_cameraGridPositions.size());

	recInfo._laserPosition = _laserPosition;
	recInfo._numLaserTargets = static_cast<glm::uint>(_laserGridPositions.size());

	recInfo._numTimeBins = _temporalResolution;
	recInfo._timeStep = _deltaT;
	recInfo._timeOffset = _t0;
	recInfo._captureSystem = _isConfocal ? CaptureSystem::Confocal : CaptureSystem::Exhaustive;
	recInfo._discardFirstLastBounces = 0u;

	recInfo._relayWallNormal = _cameraGridNormals.front();
	recInfo._relayWallMinPosition = glm::vec3(-_cameraGridSize.x / 2.0f, .0f, -_cameraGridSize.y / 2.0f);
	recInfo._relayWallSize = glm::vec3(_cameraGridSize.x, .0f, _cameraGridSize.y);

	recInfo._hiddenVolumeMin = _hiddenGeometry.minPoint();
	recInfo._hiddenVolumeMax = _hiddenGeometry.maxPoint();
	recInfo._hiddenVolumeSize = _hiddenGeometry.size();

	recInfo._voxelResolution = glm::uvec3(256u);
	recInfo._hiddenVolumeVoxelSize = _hiddenGeometry.size() / glm::vec3(recInfo._voxelResolution);
}

float* NLosData::getTimeSlice(glm::uint t)
{
	if (t >= _temporalResolution)
		throw std::out_of_range("NLosData: Time index out of range.");

	glm::uint sliceSize = static_cast<glm::uint>(_data.size()) / _dims[0];
	return _data.data() + t * sliceSize;
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

	#pragma omp simd
	for (size_t t = 0; t < numTimeBins; ++t)
	{
		for (size_t x = 0; x < numCols; ++x)
		{
			for (size_t y = 0; y < numRows; ++y)
			{
				for (size_t b = 0; b < numBounces; ++b)
				{
					data[t * numRows * numCols + y * numCols + x] += static_cast<float>(rawData[0][t][b][y][x]);
				}
			}
		}
	}

	_dims = { numTimeBins, numRows, numCols };
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
							data[t * numRowsCamera * numColsCamera * numRowsLaser * numColsLaser +
								 lY * numColsLaser * numRowsCamera * numColsCamera +
								 lX * numRowsCamera * numColsCamera +
								 cY * numColsCamera + cX] += static_cast<float>(rawData[0][t][b][lY][lX][cY][cX]);
						}
					}
				}
			}
		}
	}

	_dims = { numTimeBins, numRowsLaser, numColsLaser, numRowsCamera, numColsCamera };
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

void NLosData::padIntensity(std::vector<Complex>& paddedIntensity, size_t padding, const std::string& mode, size_t timeDim) const
{
	size_t nt = _dims[timeDim];
	size_t nt_pad = nt + 2 * padding;

	std::vector<size_t> paddedDims = _dims;
	paddedDims[timeDim] = nt_pad;

	size_t totalOriginalElements = 1;
	for (size_t dim : _dims) totalOriginalElements *= dim;

	size_t totalPaddedElements = 1;
	for (size_t dim : paddedDims) totalPaddedElements *= dim;

	std::vector<Complex> H_pad(totalPaddedElements);

	// Calculate stride for non-time dimensions
	size_t nonTimeStride = 1;
	for (size_t i = timeDim + 1; i < _dims.size(); ++i)
		nonTimeStride *= _dims[i];

	if (mode == "constant") 
	{
		for (size_t j = 0; j < totalOriginalElements / nt; ++j) { 
			for (size_t i = 0; i < nt; ++i) { 
				size_t original_idx = i * nonTimeStride + j;
				size_t padded_idx = (i + padding) * nonTimeStride + j; 
				H_pad[padded_idx] = _data[original_idx];
			}
		}
	}
	else if (mode == "edge") 
	{
		for (size_t j = 0; j < totalOriginalElements / nt; ++j) {
			for (size_t i = 0; i < nt; ++i) {
				size_t original_idx = i * nonTimeStride + j;
				size_t padded_idx = (i + padding) * nonTimeStride + j;
				H_pad[padded_idx] = _data[original_idx];
			}
		}

		// Fill padding
		for (size_t j = 0; j < totalOriginalElements / nt; ++j) 
		{
			// Before padding
			for (size_t i = 0; i < padding; ++i) 
			{
				size_t padded_idx = i * nonTimeStride + j;
				size_t original_edge_idx = 0 * nonTimeStride + j; 
				H_pad[padded_idx] = _data[original_edge_idx];
			}
			// After padding
			for (size_t i = nt + padding; i < nt_pad; ++i) 
			{
				size_t padded_idx = i * nonTimeStride + j;
				size_t original_edge_idx = (nt - 1) * nonTimeStride + j; 
				H_pad[padded_idx] = _data[original_edge_idx];
			}
		}
	}
	else 
	{
		std::cerr << "Unsupported padding mode: " << mode << '\n';
	}

	paddedIntensity = std::move(H_pad);
}
