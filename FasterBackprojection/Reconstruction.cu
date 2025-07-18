#include "stdafx.h"
#include "Reconstruction.h"

#include <cub/device/device_reduce.cuh>

#include "CudaHelper.h"
#include "FileUtilities.h"
#include "fourier.cuh"
#include "math.cuh"
#include "preprocessing.cuh"
#include "PostprocessingFilters.h"
#include "transient_postprocessing.cuh"

//

const PostprocessingFilters* Reconstruction::_postprocessingFilters[PostprocessingFilterType::NUM_POSTPROCESSING_FILTERS] = {
	new None(),
	new Laplacian(),
	new LoG(),
	new LoGFFT()
};

//

std::vector<float> Reconstruction::linearSpace(float minValue, float maxValue, int n)
{
	std::vector<float> v(n);
	float step = (maxValue - minValue) / static_cast<float>(n);

	for (int i = 0; i < n; ++i)
		v[i] = minValue + step * i;

	return v;
}

void Reconstruction::padIntensity(float* volumeGpu, cufftComplex*& paddedIntensity, size_t padding, const std::string& mode) const
{
	size_t timeDim = _nlosData->_dims.size() - 1;
	size_t nt = _nlosData->_dims[timeDim];
	size_t nt_pad = nt + 2 * padding;

	std::vector<size_t> paddedDims = _nlosData->_dims;
	paddedDims[timeDim] = nt_pad;

	size_t sliceSize = 1;
	for (size_t i = 0; i < _nlosData->_dims.size() - 1; ++i)
		sliceSize *= _nlosData->_dims[i];

	paddedIntensity = nullptr;
	CudaHelper::initializeBufferGPU(paddedIntensity, sliceSize * nt_pad);

	dim3 blockSize(64, 16);
	dim3 gridSize(
		(sliceSize + blockSize.x - 1) / blockSize.x,
		(nt_pad + blockSize.y - 1) / blockSize.y);

	padBuffer<<<gridSize, blockSize>>>(
		sliceSize, nt, padding, nt_pad, volumeGpu, paddedIntensity,
		mode == "zero" ? PadMode::Zero : PadMode::Edge);
}

void Reconstruction::filter_H_cuda(float* intensityGpu, float wl_mean, float wl_sigma, const std::string& border) const
{
	size_t nt = _nlosData->_temporalResolution;
	if (glm::epsilonEqual(wl_sigma, .0f, glm::epsilon<float>()))
	{
		std::cout << "tal.reconstruct.filter_H: wl_sigma not specified, using wl_mean / sqrt(2)\n";
		wl_sigma = wl_mean / sqrtf(2);
	}

	int t_6sigma = static_cast<int>(std::round(6 * wl_sigma / _nlosData->_deltaT));
	if (t_6sigma % 2 == 1)
		t_6sigma += 1;

	size_t nt_pad = nt + 2 * (t_6sigma - 1);
	float t_max = _nlosData->_deltaT * (nt_pad - 1);
	std::vector<float> t_vals(nt_pad);

	for (size_t i = 0; i < nt_pad; ++i)
		t_vals[i] = static_cast<float>(i) * _nlosData->_deltaT;

	// K 
	std::vector<cufftComplex> K(nt_pad);
	float sumGaussianEnvelope = 0.0f;
	std::vector<float> gaussianEnvelope(nt_pad);

	#pragma omp parallel for (reduction(+:sumGaussianEnvelope))
	for (size_t i = 0; i < nt_pad; ++i)
	{
		float val = (t_vals[i] - t_max * 0.5f) / wl_sigma;
		gaussianEnvelope[i] = std::exp(-(val * val) * 0.5f);
		sumGaussianEnvelope += gaussianEnvelope[i];
	}

	#pragma omp parallel for
	for (size_t i = 0; i < nt_pad; ++i)
	{
		float phase = TWO_PI * t_vals[i] / wl_mean;
		K[i].x = gaussianEnvelope[i] * std::cos(phase) / sumGaussianEnvelope;
		K[i].y = gaussianEnvelope[i] * std::sin(phase) / sumGaussianEnvelope;
	}

	// This prepares K for FFT since cuFFT computes the unshifted FFT
	std::vector<cufftComplex> K_ifftShifted(nt_pad);
	size_t shift = nt_pad / 2;

	#pragma omp parallel for
	for (size_t i = 0; i < nt_pad; ++i)
		K_ifftShifted[i] = K[(i + shift) % nt_pad];
	K = K_ifftShifted;

	// Pad H in host
	size_t padding = (nt_pad - nt) / 2;
	cufftComplex* d_H = nullptr;
	padIntensity(intensityGpu, d_H, padding, border);

	// Transfer H_pad_host and K to device
	cufftComplex* d_K = nullptr;
	CudaHelper::initializeBufferGPU(d_K, K.size(), K.data());

	//
	size_t dimProduct = 1;
	std::vector<size_t> newDims = _nlosData->_dims;
	newDims[newDims.size() - 1] = nt_pad;		// Last dimension is time, set to padded size
	for (size_t dim : newDims)
		dimProduct *= dim;

	cufftHandle planH;
	int n[1] = { static_cast<int>(nt_pad) };
	int rank = 1;
	int inStride = 1, outStride = 1;														// Stride between elements in time dimension (contiguous)
	int inDistance = static_cast<int>(nt_pad), outDistance = static_cast<int>(nt_pad);		// Distance between consecutive batches in output
	int sliceSize = static_cast<int>(dimProduct / nt_pad);

	// Create 1D FFT plan for batches
	CUFFT_CHECK(cufftPlanMany(&planH, rank, n, NULL, inStride, inDistance, NULL, outStride, outDistance, CUFFT_C2C, sliceSize));
	CUFFT_CHECK(cufftExecC2C(planH, d_H, d_H, CUFFT_FORWARD));

	//
	cufftHandle planK;
	CUFFT_CHECK(cufftPlan1d(&planK, static_cast<int>(nt_pad), CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(planK, d_K, d_K, CUFFT_FORWARD));

	// 
	dim3 block(512);
	dim3 grid((dimProduct + block.x - 1) / block.x);

	multiplyHK<<<grid, block >>>(d_H, d_K, sliceSize, nt_pad, sliceSize * nt_pad);
	CUFFT_CHECK(cufftExecC2C(planH, d_H, d_H, CUFFT_INVERSE));

	//
	//normalizeH<<<grid, block>>>(d_H, batch, nt_pad);		// I read the IFFT results, and they were too small; I think this is not needed

	readBackFromIFFT<<<grid, block>>>(d_H, intensityGpu, sliceSize, nt, nt_pad, padding, sliceSize * nt);
	CudaHelper::downloadBufferGPU(intensityGpu, _nlosData->_data.data(), _nlosData->_data.size(), 0);

	// 
	CUFFT_CHECK(cufftDestroy(planH));
	CudaHelper::free(d_H);
	CudaHelper::free(d_K);
}

void Reconstruction::compensateLaserCosDistance(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	ChronoUtilities::startTimer();

	const glm::uvec3 voxelResolution = recInfo._voxelResolution;
	const glm::uint sliceSize = voxelResolution.x * voxelResolution.y;

	glm::uint spatialSize = recInfo._numLaserTargets;
	if (recInfo._captureSystem == CaptureSystem::Exhaustive)
		spatialSize *= recInfo._numSensorTargets;

	dim3 blockSize(64, 8);
	dim3 gridSize(
		(spatialSize + blockSize.x - 1) / blockSize.x,
		(recInfo._numTimeBins + blockSize.y - 1) / blockSize.y
	);

	compensateLaserPosition<<<gridSize, blockSize>>>(
		recBuffers._intensity, 
		recInfo._numLaserTargets, recInfo._captureSystem == CaptureSystem::Confocal ? 1 : recInfo._numSensorTargets, spatialSize, recInfo._numTimeBins);
	CudaHelper::synchronize("compensateLaserCosDistance");
}

void Reconstruction::normalizeMatrix(float* v, glm::uint size)
{
	size_t tempStorageBytes = 0;
	void* tempStorage = nullptr;

	float* maxValue = nullptr;
	CudaHelper::initializeBufferGPU(maxValue, 1);

	float* minValue = nullptr;
	CudaHelper::initializeBufferGPU(minValue, 1);

	cub::DeviceReduce::Max(tempStorage, tempStorageBytes, v, maxValue, size);
	cudaMalloc(&tempStorage, tempStorageBytes);
	cub::DeviceReduce::Max(tempStorage, tempStorageBytes, v, maxValue, size);
	cub::DeviceReduce::Min(tempStorage, tempStorageBytes, v, minValue, size);

	glm::uint threadsBlock = 512, numBlocks = CudaHelper::getNumBlocks(size, threadsBlock);
	normalizeReconstruction<<<numBlocks, threadsBlock>>>(v, size, maxValue, minValue);

	CudaHelper::free(tempStorage);
	CudaHelper::free(maxValue);
	CudaHelper::free(minValue);
}

bool Reconstruction::saveReconstructedAABB(const std::string& filename, float* voxels, glm::uint numVoxels)
{
	std::vector<float> voxelsCpu(numVoxels);
	CudaHelper::downloadBufferGPU(voxels, voxelsCpu.data(), numVoxels);

	return FileUtilities::write<float>(filename, voxelsCpu);
}
