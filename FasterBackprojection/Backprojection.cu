#include "stdafx.h"
#include "Backprojection.h"

#include <cub/device/device_scan.cuh>

#include "CudaHelper.h"
#include "RandomUtilities.h"
#include "TransientImage.h"
#include "transient_reconstruction.cuh"

//

void Backprojection::reconstructDepths(
	NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
	const TransientParameters& transientParams, const std::vector<float>& depths)
{
	std::vector<float> reconstructionDepths = linearSpace(
		recInfo.getFocusDepth() - 0.1f, recInfo.getFocusDepth() + 0.1f, transientParams._numReconstructionDepths
	);
	assert(!reconstructionDepths.empty());

	if (recInfo._captureSystem == CaptureSystem::Confocal)
		reconstructDepthConfocal(recInfo, reconstructionDepths);
	else if (recInfo._captureSystem == CaptureSystem::Exhaustive)
		reconstructDepthExhaustive(recInfo, reconstructionDepths);
}

void Backprojection::reconstructVolume(
	NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
	const TransientParameters& transientParams)
{
	_nlosData = nlosData;

	_perf.setAlgorithmName("Backprojection");
	_perf.tic();

	const glm::uvec3 voxelResolution = recInfo._voxelResolution;
	const glm::uint numVoxels = voxelResolution.x * voxelResolution.y * voxelResolution.z;
	float* volumeGpu = nullptr;
	CudaHelper::initializeZeroBufferGPU(volumeGpu, numVoxels);

	if (transientParams._compensateLaserCosDistance)
		compensateLaserCosDistance(recInfo, recBuffers);

	if (transientParams._useFourierFilter)
		filter_H_cuda(recBuffers._intensity, recInfo._timeStep * 10.0f, .0f);

	if (recInfo._captureSystem == CaptureSystem::Confocal)
		//reconstructVolumeConfocal(volumeGpu, recInfo);
		reconstructAABBConfocalMIS(volumeGpu, recInfo, recBuffers);
	else if (recInfo._captureSystem == CaptureSystem::Exhaustive)
		reconstructVolumeExhaustive(volumeGpu, recInfo);

	// Post-process the activation matrix
	_postprocessingFilters[transientParams._postprocessingFilterType]->compute(volumeGpu, voxelResolution, transientParams);
	normalizeMatrix(volumeGpu, numVoxels);

	_perf.toc();
	_perf.summarize();

	// Save volume & free resources
	if (transientParams._saveReconstructedBoundingBox)
		saveReconstructedAABB(transientParams._outputFolder + transientParams._outputAABBName, volumeGpu, numVoxels);

	// Save image if requested
	if (transientParams._saveMaxImage)
		Backprojection::saveMaxImage(
			transientParams._outputFolder + transientParams._outputMaxImageName,
			volumeGpu,
			voxelResolution,
			true);

	CudaHelper::free(volumeGpu);
}

//

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
void Backprojection::buildAliasTables(const std::vector<float>& cdf, std::vector<glm::uint>& aliasTable, std::vector<float>& probTable)
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

void Backprojection::reconstructDepthConfocal(const ReconstructionInfo& recInfo, std::vector<float>& reconstructionDepths)
{
	// Pointer to reconstruction depths in GPU memory
	float* depthsGPU = nullptr;
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

void Backprojection::reconstructDepthExhaustive(const ReconstructionInfo& recInfo, std::vector<float>& reconstructionDepths)
{
	ChronoUtilities::startTimer();

	// Pointer to reconstruction depths in GPU memory
	float* depthsGPU = nullptr;
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

	backprojectExhaustive<<<gridSize, blockSize >>>(activationGpu, depthsGPU, numDepths);
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

void Backprojection::reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo)
{
	_perf.tic("Backprojection");

	const glm::uvec3 voxelResolution = recInfo._voxelResolution;
	const glm::uint sliceSize = voxelResolution.x * voxelResolution.y;

	dim3 blockSize(BLOCK_X_CONFOCAL, BLOCK_Y_CONFOCAL);
	dim3 gridSize(
		(sliceSize + blockSize.x - 1) / blockSize.x,
		(recInfo._numLaserTargets + blockSize.y - 1) / blockSize.y,
		(voxelResolution.z + blockSize.z - 1) / blockSize.z
	);

	backprojectConfocalVoxel<<<gridSize, blockSize>>>(volume, sliceSize);

	_perf.toc();
}

void Backprojection::reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo)
{
	_perf.tic("Backprojection");

	const glm::uvec3 volumeResolution = glm::uvec3(_nlosData->_dims[0], _nlosData->_dims[1], _nlosData->_dims[2]);
	const glm::uint sliceSize = volumeResolution.x * volumeResolution.y;

	dim3 blockSize(BLOCK_X_CONFOCAL, BLOCK_Y_CONFOCAL);
	dim3 gridSize(
		(recInfo._numLaserTargets * recInfo._numSensorTargets + blockSize.x - 1) / blockSize.x,
		(sliceSize + blockSize.y - 1) / blockSize.y,
		(volumeResolution.z + blockSize.z - 1) / blockSize.z
	);

	backprojectExhaustiveVoxel<<<gridSize, blockSize>>>(volume, sliceSize);

	_perf.toc();
}

void Backprojection::reconstructAABBConfocalMIS(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
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
	std::vector<float> noiseHost(numVoxels * 2);
#pragma omp parallel for
	for (glm::uint i = 0; i < numVoxels; ++i)
		noiseHost[i] = RandomUtilities::getUniformRandom();
	CudaHelper::initializeBufferGPU(noiseGpu, numVoxels, noiseHost.data());

	// CUB's inclusive sum
	{
		size_t storageBytes = 0;
		cub::DeviceScan::InclusiveSum(nullptr, storageBytes, recBuffers._intensity, spatioTemporalSum, static_cast<int>(totalElements));

		void* tempStorage = nullptr;
		cudaMalloc(&tempStorage, storageBytes);

		// Perform the inclusive scan
		cub::DeviceScan::InclusiveSum(tempStorage, storageBytes, recBuffers._intensity, spatioTemporalSum, static_cast<int>(totalElements));

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

	backprojectConfocalVoxelMIS<<<gridSize, blockSize>>>(
		spatialSumGpu, volume, noiseGpu, aliasTableGpu, probTableGpu, sliceSize, numSamples, numVoxels);
	CudaHelper::synchronize("backprojectConfocalVoxelMIS");

	CudaHelper::free(spatioTemporalSum);
	CudaHelper::free(spatialSumGpu);
	CudaHelper::free(noiseGpu);
	CudaHelper::free(aliasTableGpu);
	CudaHelper::free(probTableGpu);
}
