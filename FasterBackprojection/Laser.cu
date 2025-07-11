// ReSharper disable CppExpressionWithoutSideEffects
// ReSharper disable CppClangTidyClangDiagnosticShadow
#include "stdafx.h"
#include "Laser.cuh"

#include <valarray>
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>

#include "ChronoUtilities.h"
#include "GpuStructs.cuh"
#include "fourier.cuh"
#include "transient_reconstruction.cuh"

#include "CudaHelper.h"
#include "FileUtilities.h"
#include "LCT.h"
#include "PostprocessingFilters.h"
#include "RandomUtilities.h"
#include "TransientImage.h"
#include "transient_postprocessing.cuh"

//

const PostprocessingFilters* Laser::_postprocessingFilters[PostprocessingFilterType::NUM_POSTPROCESSING_FILTERS] = {
	new None(),
	new Laplacian(),
	new LoG(),
	new LoGFFT()
};

//

Laser::Laser(NLosData* nlosData) : _nlosData(nlosData)
{
}

Laser::~Laser() = default;

void Laser::reconstructShape(const TransientParameters& transientParams)
{
	ReconstructionInfo recInfo;
	ReconstructionBuffers recBuffers;

	// Fourier?
	//if (transientParams._useFourierFilter)
	//{
	//	ChronoUtilities::startTimer();
	//	filter_H_cuda(_nlosData->_deltaT * 10.0f, 0.0f);		// Sigma zero equals to non-valid, thus it is being calculated from wl_mean
	//	std::cout << "Filter H finished in " << ChronoUtilities::getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";
	//}

	// Transfer data to GPU
	_nlosData->toGpu(recInfo, recBuffers, transientParams);

	CudaHelper::checkError(cudaMemcpyToSymbol(rtRecInfo, &recInfo, sizeof(ReconstructionInfo)));
	CudaHelper::checkError(cudaMemcpyToSymbol(laserTargets, &recBuffers._laserTargets, sizeof(glm::vec3*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(sensorTargets, &recBuffers._sensorTargets, sizeof(glm::vec3*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(intensityCube, &recBuffers._intensity, sizeof(float*)));

	std::cout << "Reconstructing shape...\n";

	if (transientParams._reconstructAABB)
		reconstructShapeAABB(recInfo, recBuffers, transientParams);
	else
		reconstructShapeDepths(recInfo, transientParams);

	CudaHelper::free(recBuffers._laserTargets);
	CudaHelper::free(recBuffers._sensorTargets);
	CudaHelper::free(recBuffers._intensity);
}

std::vector<double> Laser::linearSpace(double minValue, double maxValue, int n)
{
	std::vector<double> v(n);
	double step = (maxValue - minValue) / static_cast<double>(n);

	for (int i = 0; i < n; ++i)
		v[i] = minValue + step * i;

	return v;
}

float* Laser::padIntensity(cufftComplex*& paddedIntensity, size_t padding, const std::string& mode) const
{
	size_t timeDim = _nlosData->_dims.size() - 1;
	size_t nt = _nlosData->_dims[timeDim];
	size_t nt_pad = nt + 2 * padding;

	std::vector<size_t> paddedDims = _nlosData->_dims;
	paddedDims[timeDim] = nt_pad;

	size_t sliceSize = 1;
	for (size_t i = 0; i < _nlosData->_dims.size() - 1; ++i)
		sliceSize *= _nlosData->_dims[i];

	float* d_H_orig = nullptr;
	CudaHelper::initializeBufferGPU(d_H_orig, _nlosData->_data.size(), _nlosData->_data.data());

	//paddedIntensity = nullptr;
	//CudaHelper::initializeBufferGPU(paddedIntensity, sliceSize * nt_pad);

	//dim3 blockSize(32, 16);
	//dim3 gridSize(
	//	(sliceSize + blockSize.x - 1) / blockSize.x,
	//	(nt_pad + blockSize.y - 1) / blockSize.y);

	//padBuffer<<<gridSize, blockSize>>>(
	//	sliceSize, nt, padding, nt_pad, d_H_orig, paddedIntensity, 
	//	mode == "zero" ? PadMode::Zero : PadMode::Edge);

	std::vector<cufftComplex> paddedIntensityBuf(sliceSize * nt_pad);

	#pragma omp parallel for
	for (size_t i = 0; i < sliceSize; ++i)
	{
		for (size_t t = 0; t < nt; ++t)
			paddedIntensityBuf[i * nt_pad + (t + padding)] = cufftComplex{ _nlosData->_data[i * nt + t], 0.0f };

		if (mode == "constant" || mode == "zero")
		{
			for (size_t t = 0; t < padding; ++t) {
				paddedIntensityBuf[i * nt_pad + t] = { 0.0f, 0.0f };
			}
			for (size_t t = nt + padding; t < nt_pad; ++t) {
				paddedIntensityBuf[i * nt_pad + t] = { 0.0f, 0.0f };
			}
		}
		else if (mode == "edge")
		{
			float first = _nlosData->_data[i * nt + 0];
			float last = _nlosData->_data[i * nt + nt - 1];

			for (size_t t = 0; t < padding; ++t)
				paddedIntensityBuf[i * nt_pad + t] = cufftComplex{ first, .0f };
			for (size_t t = nt + padding; t < nt_pad; ++t)
				paddedIntensityBuf[i * nt_pad + t] = cufftComplex{ last, .0f };
		}
	}

	CudaHelper::initializeBufferGPU(paddedIntensity, paddedIntensityBuf.size(), paddedIntensityBuf.data());

	return d_H_orig;
}

void Laser::filter_H_cuda(float wl_mean, float wl_sigma, const std::string& border) const
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

	//#pragma omp parallel for (reduction(+:sumGaussianEnvelope))
	for (size_t i = 0; i < nt_pad; ++i)
	{
		float val = (t_vals[i] - t_max * 0.5f) / wl_sigma;
		gaussianEnvelope[i] = std::exp(-(val * val) * 0.5f);
		sumGaussianEnvelope += gaussianEnvelope[i];
	}

	//#pragma omp parallel for
	for (size_t i = 0; i < nt_pad; ++i)
	{
		float phase = TWO_PI * t_vals[i] / wl_mean;
		K[i].x = gaussianEnvelope[i] * std::cos(phase) / sumGaussianEnvelope;
		K[i].y = gaussianEnvelope[i] * std::sin(phase) / sumGaussianEnvelope;
	}

	// This prepares K for FFT since cuFFT computes the unshifted FFT
	std::vector<cufftComplex> K_ifftShifted(nt_pad);
	size_t shift = nt_pad / 2;

	//#pragma omp parallel for
	for (size_t i = 0; i < nt_pad; ++i)
		K_ifftShifted[i] = K[(i + shift) % nt_pad];
	K = K_ifftShifted;

	// Pad H in host
	size_t padding = (nt_pad - nt) / 2;
	cufftComplex* d_H = nullptr;
	float* d_H_orig = padIntensity(d_H, padding, border);

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
	int batch = static_cast<int>(dimProduct / nt_pad);

	// Create 1D FFT plan for batches
	CUFFT_CHECK(cufftPlanMany(&planH, rank, n, NULL, inStride, inDistance, NULL, outStride, outDistance, CUFFT_C2C, batch));
	CUFFT_CHECK(cufftExecC2C(planH, d_H, d_H, CUFFT_FORWARD));

	//
	cufftHandle planK;
	CUFFT_CHECK(cufftPlan1d(&planK, static_cast<int>(nt_pad), CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(planK, d_K, d_K, CUFFT_FORWARD));

	// 
	dim3 block(256);
	dim3 grid((batch * nt_pad + block.x - 1) / block.x);

	multiplyHK<<<grid, block>>>(d_H, d_K, batch, nt_pad);
	cudaDeviceSynchronize();

	CUFFT_CHECK(cufftExecC2C(planH, d_H, d_H, CUFFT_INVERSE));

	//
	//normalizeH<<<grid, block>>>(d_H, batch, nt_pad);		// I read the IFFT results, and they were too small; I think this is not needed

	// Copy result back to host
	std::vector<cufftComplex> H_pad_host(dimProduct);
	CudaHelper::downloadBufferGPU(d_H, H_pad_host.data(), dimProduct, 0);

	//
	size_t inner = nt, outer = dimProduct / nt_pad;

	#pragma omp parallel for
	for (size_t i = 0; i < outer; ++i) {
		size_t paddedOffset = i * nt_pad + padding;
		size_t unpaddedOffset = i * nt;

		for (size_t j = 0; j < inner; ++j)
		{
			cufftComplex val = H_pad_host[paddedOffset + j];
			_nlosData->_data[unpaddedOffset + j] = val.x;
		}
	}

	//dim3 blockSize(32, 16);
	//dim3 gridSize(
	//	(dimProduct / nt_pad + blockSize.x - 1) / blockSize.x,
	//	(nt + blockSize.y - 1) / blockSize.y);
	//readBackFromIFFT<<<gridSize, blockSize>>>(d_H, d_H_orig, dimProduct / nt_pad, nt, nt_pad, padding);

	//CudaHelper::downloadBufferGPU(d_H_orig, _nlosData->_data.data(), _nlosData->_data.size(), 0);

	// 
	CUFFT_CHECK(cufftDestroy(planH));
	CudaHelper::free(d_H);
	CudaHelper::free(d_K);
}

/**
 * @brief Implements Vose's Alias Method for efficient sampling from a discrete probability distribution.
 *
 * This function pre-computes the alias and probability tables on the CPU, which can then
 * be used for O(1) sampling on the GPU.
 *
 * @param cdf A vector of probabilities for each outcome. Must sum to approximately 1.0.
 * @param aliasTable An output vector that will store the alias indices.
 * @param probTable An output vector that will store the probabilities for direct sampling.
 * @throws std::runtime_error if probabilities are invalid (e.g., negative, sum not approx 1).
 */
void Laser::buildAliasTables(const std::vector<float>& cdf, std::vector<glm::uint>& aliasTable, std::vector<float>& probTable)
{
	// Build probabilities from CDF
	const glm::uint N = static_cast<glm::uint>(cdf.size());

	std::vector<float> probabilities(N);
	probabilities[0] = cdf[0];
	for (glm::uint i = 1; i < N; ++i)
		probabilities[i] = cdf[i] - cdf[i - 1];

	aliasTable.resize(N);
	probTable.resize(N);

	// Validate probabilities and scale them
	std::vector<glm::uint> smallProbs; // Indices of bins with probability < 1.0
	std::vector<glm::uint> largeProbs; // Indices of bins with probability >= 1.0

	float sumProbabilities = 0.0f;
	for (glm::uint i = 0; i < N; ++i)
	{
		if (probabilities[i] < 0.0f)
			throw std::runtime_error("Probabilities cannot be negative.");

		probTable[i] = probabilities[i] * N; // Scale P_i by N
		sumProbabilities += probabilities[i];

		if (probTable[i] < 1.0f)
			smallProbs.push_back(i);
		else
			largeProbs.push_back(i);
	}

	// Check if sum is close to 1.0
	if (std::abs(sumProbabilities - 1.0f) > 1e-5f)
		throw std::runtime_error("Probabilities do not sum to approximately 1.0.");

	while (!smallProbs.empty() && !largeProbs.empty())
	{
		glm::uint s = smallProbs.back();
		smallProbs.pop_back();
		glm::uint l = largeProbs.back();
		largeProbs.pop_back();

		aliasTable[s] = l;
		probTable[l] += probTable[s] - 1.0f;

		if (probTable[l] < 1.0f)
			smallProbs.push_back(l);
		else
			largeProbs.push_back(l);

	}

	// Handle any remaining bins (due to floating point inaccuracies)
	while (!smallProbs.empty()) {
		probTable[smallProbs.back()] = 1.0f; // Set to 1.0 (means it always picks itself)
		smallProbs.pop_back();
	}
	while (!largeProbs.empty()) {
		probTable[largeProbs.back()] = 1.0f; // Set to 1.0
		largeProbs.pop_back();
	}
}

void Laser::normalizeMatrix(float* v, glm::uint size)
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

void Laser::reconstructShapeAABB(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers, const TransientParameters& transientParams)
{
	const glm::uvec3 voxelResolution = recInfo._voxelResolution;
	const glm::uint numVoxels = voxelResolution.x * voxelResolution.y * voxelResolution.z;
	float* volumeGpu = nullptr;
	CudaHelper::initializeZeroBufferGPU(volumeGpu, numVoxels);

	if (recInfo._captureSystem == CaptureSystem::Confocal)
		reconstructAABBConfocal(volumeGpu, recInfo);
		//reconstructAABBConfocalMIS(volumeGpu, recInfo);
	else if (recInfo._captureSystem == CaptureSystem::Exhaustive)
		reconstructAABBExhaustive(volumeGpu, recInfo);

	// Post-process the activation matrix
	_postprocessingFilters[transientParams._postprocessingFilterType]->compute(volumeGpu, voxelResolution, transientParams);
	normalizeMatrix(volumeGpu, numVoxels);

	std::cout << "Reconstruction finished in " << ChronoUtilities::getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	// Save volume & free resources
	saveReconstructedAABB("output/aabb.cube", volumeGpu, numVoxels);
	CudaHelper::free(volumeGpu);
}

void Laser::reconstructShapeDepths(const ReconstructionInfo& recInfo, const TransientParameters& transientParams)
{
	std::cout << recInfo.getFocusDepth() << std::endl;
	std::vector<double> reconstructionDepths = linearSpace(
		recInfo.getFocusDepth() - 0.1f, recInfo.getFocusDepth() + 0.1f, static_cast<glm::uint>(transientParams._numReconstructionDepths)
	);
	assert(!reconstructionDepths.empty());

	if (recInfo._captureSystem == CaptureSystem::Confocal)
		reconstructDepthConfocal(recInfo, reconstructionDepths);
	else if (recInfo._captureSystem == CaptureSystem::Exhaustive)
		reconstructDepthExhaustive(recInfo, reconstructionDepths);
}

void Laser::reconstructDepthConfocal(const ReconstructionInfo& recInfo, std::vector<double>& reconstructionDepths)
{
	// Pointer to reconstruction depths in GPU memory
	double* depthsGPU = nullptr;
	CudaHelper::initializeBufferGPU(depthsGPU, reconstructionDepths.size(), reconstructionDepths.data());

	// Let's initialize a matrix of size (sx, sy, depths)
	const glm::uint numDepths = static_cast<glm::uint>(reconstructionDepths.size());
	const glm::uint numPixels = recInfo._numSensorTargets * numDepths;
	float* activationGpu = nullptr;
	CudaHelper::initializeZeroBufferGPU(activationGpu, numPixels);

	ChronoUtilities::startTimer();

	// Determine size of thread groups and threads within them
	dim3 blockSize(BLOCK_X_CONFOCAL, BLOCK_Y_CONFOCAL);
	dim3 gridSize(
		(recInfo._numLaserTargets + blockSize.x - 1) / blockSize.x,
		(recInfo._numLaserTargets + blockSize.y - 1) / blockSize.y,			
		(static_cast<glm::uint>(reconstructionDepths.size()) + blockSize.z - 1) / blockSize.z);

	backprojectConfocal<<<gridSize, blockSize>>>(activationGpu, depthsGPU, numDepths);
	normalizeMatrix(activationGpu, numPixels);
	CudaHelper::synchronize("backprojectConfocal");

	long long elapsedTime = ChronoUtilities::getElapsedTime(ChronoUtilities::MILLISECONDS);
	std::cout << "Reconstruction finished in " << elapsedTime << " milliseconds.\n";
	std::cout << "Time per depth: " << elapsedTime / numDepths << "\n";

	// Prepare buckets and info for storing data in the local system
	const std::string outputFolder = "output/";
	const uint16_t cameraTargetRes = static_cast<uint16_t>(glm::sqrt(recInfo._numSensorTargets));
	std::cout << "Camera target resolution: " << cameraTargetRes << '\n';

	std::vector<float> reconstruction(numPixels);
	CudaHelper::downloadBufferGPU(activationGpu, reconstruction.data(), numPixels, 0);

	#pragma omp parallel for
	for (int idx = 0; idx < static_cast<int>(numDepths); ++idx)
	{
		TransientImage transientImage(cameraTargetRes, cameraTargetRes);
		transientImage.save(
			outputFolder + "transient_" + std::to_string(idx) + ".png", reconstruction.data(), 
			glm::uvec2(cameraTargetRes * (cameraTargetRes < 256u ? 256u / cameraTargetRes : 1u)), 
			numDepths, idx
		);
	}

	CudaHelper::free(activationGpu);
	CudaHelper::free(depthsGPU);
}

void Laser::reconstructDepthExhaustive(const ReconstructionInfo& recInfo, std::vector<double>& reconstructionDepths)
{
	ChronoUtilities::startTimer();

	// Pointer to reconstruction depths in GPU memory
	double* depthsGPU = nullptr;
	CudaHelper::initializeBufferGPU(depthsGPU, reconstructionDepths.size(), reconstructionDepths.data());

	// Let's initialize a matrix of size (sx, sy, depths)
	const glm::uint numDepths = static_cast<glm::uint>(reconstructionDepths.size());
	const glm::uint numPixels = recInfo._numSensorTargets * numDepths;
	float* activationGpu = nullptr;
	CudaHelper::initializeZeroBufferGPU(activationGpu, numPixels);

	// Determine size of thread groups and threads within them
	dim3 blockSize(BLOCK_X_EXHAUSTIVE, BLOCK_Y_EXHAUSTIVE);
	dim3 gridSize(
		(recInfo._numLaserTargets * recInfo._numSensorTargets + blockSize.x - 1) / blockSize.x,
		(recInfo._numLaserTargets + blockSize.y - 1) / blockSize.y,
		(static_cast<glm::uint>(reconstructionDepths.size()) + blockSize.z - 1) / blockSize.z);

	backprojectExhaustive<<<gridSize, blockSize>>>(activationGpu, depthsGPU, numDepths);
	normalizeMatrix(activationGpu, numPixels);
	CudaHelper::synchronize("backprojectExhaustive");

	long long elapsedTime = ChronoUtilities::getElapsedTime(ChronoUtilities::MILLISECONDS);
	std::cout << "Reconstruction finished in " << elapsedTime << " milliseconds.\n";
	std::cout << "Time per depth: " << elapsedTime / numDepths << "\n";

	// Prepare buckets and info for storing data in the local system
	const std::string outputFolder = "output/";
	const uint16_t cameraTargetRes = static_cast<uint16_t>(glm::sqrt(recInfo._numSensorTargets));
	std::cout << "Camera target resolution: " << cameraTargetRes << '\n';

	std::vector<float> reconstruction(numPixels);
	CudaHelper::downloadBufferGPU(activationGpu, reconstruction.data(), numPixels, 0);

	#pragma omp parallel for
	for (int idx = 0; idx < static_cast<int>(numDepths); ++idx)
	{
		TransientImage transientImage(cameraTargetRes, cameraTargetRes);
		transientImage.save(
			outputFolder + "transient_" + std::to_string(idx) + ".png", reconstruction.data(), glm::uvec2(cameraTargetRes) * 4u,
			numDepths, idx
		);
	}

	CudaHelper::free(activationGpu);
	CudaHelper::free(depthsGPU);
}

void Laser::reconstructAABBConfocal(float* volume, const ReconstructionInfo& recInfo)
{
	ChronoUtilities::startTimer();

	const glm::uvec3 voxelResolution = recInfo._voxelResolution;
	const glm::uint sliceSize = voxelResolution.x * voxelResolution.y;

	dim3 blockSize(BLOCK_X_CONFOCAL, BLOCK_Y_CONFOCAL); 
	dim3 gridSize(
		(sliceSize + blockSize.x - 1) / blockSize.x,    
		(recInfo._numLaserTargets + blockSize.y - 1) / blockSize.y, 
		(voxelResolution.z + blockSize.z - 1) / blockSize.z  
	);

	backprojectConfocalVoxel<<<gridSize, blockSize>>>(volume, sliceSize);
	CudaHelper::synchronize("backprojectConfocalVoxel");
}

void Laser::reconstructAABBExhaustive(float* volume, const ReconstructionInfo& recInfo)
{
	ChronoUtilities::startTimer();

	const glm::uvec3 voxelResolution = recInfo._voxelResolution;
	const glm::uint sliceSize = voxelResolution.x * voxelResolution.y;

	dim3 blockSize(BLOCK_X_CONFOCAL, BLOCK_Y_CONFOCAL);
	dim3 gridSize(
		(recInfo._numLaserTargets * recInfo._numSensorTargets + blockSize.x - 1) / blockSize.x,
		(sliceSize + blockSize.y - 1) / blockSize.y,
		(voxelResolution.z + blockSize.z - 1) / blockSize.z
	);

	backprojectExhaustiveVoxel<<<gridSize, blockSize>>>(volume, sliceSize);
	CudaHelper::synchronize("backprojectConfocalVoxel");
}

bool Laser::saveReconstructedAABB(const std::string& filename, float* voxels, glm::uint numVoxels)
{
	std::vector<float> voxelsCpu(numVoxels);
	CudaHelper::downloadBufferGPU(voxels, voxelsCpu.data(), numVoxels);

	return FileUtilities::write<float>(filename, voxelsCpu);
}

void Laser::reconstructAABBConfocalMIS(float* volume, const ReconstructionInfo& recInfo) const
{
	const glm::uvec3 voxelResolution = recInfo._voxelResolution;
	const glm::uint numVoxels = voxelResolution.x * voxelResolution.y * voxelResolution.z;
	const glm::uint numTimeBins = recInfo._numTimeBins;
	const glm::uint sliceSize = voxelResolution.x * voxelResolution.y;

	glm::uint totalElements = 1;
	for (const auto& dim : _nlosData->_dims)
		totalElements *= static_cast<glm::uint>(dim);

	float* spatioTemporalSum = nullptr, * spatialSumGpu = nullptr;
	CudaHelper::initializeBufferGPU(spatioTemporalSum, totalElements);
	CudaHelper::initializeBufferGPU(spatialSumGpu, totalElements / numTimeBins);

	float* noiseGpu = nullptr;
	std::vector<float> noiseHost(numVoxels);
	for (glm::uint i = 0; i < numVoxels; ++i)
		noiseHost[i] = RandomUtilities::getUniformRandom();
	CudaHelper::initializeBufferGPU(noiseGpu, numVoxels, noiseHost.data());

	// CUB's inclusive sum
	{
		size_t storageBytes = 0;
		cub::DeviceScan::InclusiveSum(nullptr, storageBytes, volume, spatioTemporalSum, static_cast<int>(totalElements));

		void* tempStorage = nullptr;
		cudaMalloc(&tempStorage, storageBytes);

		// Perform the inclusive scan
		cub::DeviceScan::InclusiveSum(tempStorage, storageBytes, volume, spatioTemporalSum, static_cast<int>(totalElements));

		CudaHelper::free(tempStorage);
	}

	// Spatio-temporal prefix sum
	{
		glm::uint blockSize = 512, blocks = CudaHelper::getNumBlocks(totalElements / numTimeBins, blockSize);
		spatioTemporalPrefixSum<<<blocks, blockSize>>>(spatioTemporalSum, spatialSumGpu, totalElements / numTimeBins, numTimeBins);
	}

	// Normalize spatio-temporal sum
	{
		glm::uint threadsBlock = 512, numBlocks = CudaHelper::getNumBlocks(totalElements / numTimeBins, threadsBlock);
		normalizeSpatioTemporalPrefixSum<<<numBlocks, threadsBlock>>>(spatioTemporalSum, numTimeBins, totalElements);
	}

	// Normalize the spatial sum
	normalizeMatrix(spatialSumGpu, totalElements / numTimeBins);

	glm::uint* aliasTableGpu = nullptr;
	float* probTableGpu = nullptr;
	{
		std::vector<float> spatialCDF(totalElements / numTimeBins);
		CudaHelper::downloadBufferGPU(spatialSumGpu, spatialCDF.data(), totalElements / numTimeBins);

		std::vector<glm::uint> aliasTable;
		std::vector<float> probTable;
		buildAliasTables(spatialCDF, aliasTable, probTable);

		CudaHelper::initializeBufferGPU(aliasTableGpu, aliasTable.size(), aliasTable.data());
		CudaHelper::initializeBufferGPU(probTableGpu, probTable.size(), probTable.data());
	}

	glm::uint numSamples = recInfo._numLaserTargets / 4;
	dim3 blockSize(BLOCK_X_CONFOCAL, BLOCK_Y_CONFOCAL);
	dim3 gridSize(
		(sliceSize + blockSize.x - 1) / blockSize.x,
		(numSamples + blockSize.y - 1) / blockSize.y,
		(voxelResolution.z + blockSize.z - 1) / blockSize.z
	);

	CudaHelper::checkError(cudaMemcpyToSymbol(spatialSum, &spatialSumGpu, sizeof(float*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(aliasTable, &aliasTableGpu, sizeof(glm::uint*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(probTable, &probTableGpu, sizeof(float*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(noiseBuffer, &noiseGpu, sizeof(float*)));

	backprojectConfocalVoxelMIS<<<gridSize, blockSize>>>(spatialSumGpu, volume, noiseGpu, aliasTableGpu, probTableGpu, sliceSize, numSamples, numVoxels);
	CudaHelper::synchronize("backprojectConfocalVoxelMIS");

	CudaHelper::free(spatioTemporalSum);
	CudaHelper::free(spatialSumGpu);
	CudaHelper::free(noiseGpu);
	CudaHelper::free(aliasTableGpu);
	CudaHelper::free(probTableGpu);
}
