// ReSharper disable CppExpressionWithoutSideEffects
#include "stdafx.h"
#include "Laser.cuh"

#include <cub/device/device_reduce.cuh>
#include <cufft.h>

#include "ChronoUtilities.h"
#include "GpuStructs.cuh"
#include "fourier.cuh"
#include "transient_reconstruction.cuh"

#include "CudaHelper.h"
#include "FileUtilities.h"
#include "RandomUtilities.h"
#include "TransientImage.h"

//

Laser::Laser() 
{
}

Laser::~Laser()
{
}

void Laser::reconstructShape(ReconstructionInfo& recInfo, ReconstructionBuffers& recBuffers, bool reconstructAABB)
{
	CudaHelper::checkError(cudaMemcpyToSymbol(rtRecInfo, &recInfo, sizeof(ReconstructionInfo)));
	CudaHelper::checkError(cudaMemcpyToSymbol(laserTargets, &recBuffers._laserTargets, sizeof(glm::vec3*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(sensorTargets, &recBuffers._sensorTargets, sizeof(glm::vec3*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(intensityCube, &recBuffers._intensity, sizeof(float*)));

	std::cout << "Reconstructing shape...\n";

	// Once the cube is generated, we can filter it
	//if (transientParameters._useFourierFilter)
	//	fourierFilter(transientParameters);

	if (reconstructAABB)
		reconstructShapeAABB(recInfo);
	else
		reconstructShapeDepths(recInfo);

	CudaHelper::free(recBuffers._laserTargets);
	CudaHelper::free(recBuffers._sensorTargets);
	CudaHelper::free(recBuffers._intensity);
}

std::vector<double> Laser::fftFrequencies(int n, float d)
{
	std::vector<double> frequencies(n);
	double inv_nd = 1.0f / (n * d);
	int half = (n + 1) / 2;

	for (int i = 0; i < half; ++i)
		frequencies[i] = i * inv_nd;
	for (int i = half; i < n; ++i)
		frequencies[i] = static_cast<double>(i - n) * inv_nd;

	return frequencies;
}

std::vector<double> Laser::linearSpace(double minValue, double maxValue, int n)
{
	std::vector<double> v(n);
	float step = (maxValue - minValue) / n;

	for (int i = 0; i < n; ++i)
		v[i] = minValue + step * i;

	return v;
}

void Laser::fourierFilter(const TransientParameters& transientParameters)
{
	// Filter with Fourier transform
	int numTimeBins = transientParameters._numTimeBins;
	int numPaddedTimeBins = numTimeBins;
	double deltaT = transientParameters._temporalResolution;
	int numSigmaBins = static_cast<int>(transientParameters._wavelengthMean / deltaT * 6.0f);
	if (numSigmaBins % 2 == 1)
		++numSigmaBins;

	numPaddedTimeBins += numSigmaBins * 2;

	double tMax = deltaT * (static_cast<double>(numPaddedTimeBins) - 1);
	std::vector<double> frequencies = fftShift(fftFrequencies(numPaddedTimeBins, deltaT));
	std::vector<double> t = linearSpace(.0f, tMax, numPaddedTimeBins);

	double meanIndex = (static_cast<float>(numPaddedTimeBins) * deltaT) / transientParameters._wavelengthMean;
	double sigmaIdx = (static_cast<float>(numPaddedTimeBins) * deltaT) / (transientParameters._wavelengthSigma * 6.0f);
	int frequencyMinIndex = numPaddedTimeBins / 2 + static_cast<int>(glm::floor(meanIndex - 3.0f * sigmaIdx));
	int frequencyMaxIndex = numPaddedTimeBins / 2 + static_cast<int>(glm::ceil(meanIndex + 3.0f * sigmaIdx));

	double sumGaussianEnvelope = 0.0f;
	std::vector<float> gaussianEnvelope(numPaddedTimeBins);

	for (int i = 0; i < numPaddedTimeBins; ++i)
	{
		double normalizedTime = (t[i] - tMax / 2.0f) / transientParameters._wavelengthSigma;
		gaussianEnvelope[i] = glm::exp(-0.5f * normalizedTime * normalizedTime);
		sumGaussianEnvelope += gaussianEnvelope[i];
	}

	std::vector<cuComplex> K(numPaddedTimeBins);
	for (int i = 0; i < numPaddedTimeBins; ++i)
	{
		double phase = 2.0 * glm::pi<double>() * t[i] / transientParameters._wavelengthMean;
		double normalizedEnvelope = gaussianEnvelope[i] / sumGaussianEnvelope;
		K[i] = make_cuComplex(normalizedEnvelope * glm::cos(phase), normalizedEnvelope * glm::sin(phase));
	}

	K = ifftshift<cuComplex>(K);

	std::cout << "Frequency min index: " << frequencyMinIndex << '\n';

	glm::uint padding = numPaddedTimeBins - numTimeBins;
	assert(padding % 2 == 0);
	padding /= 2;

	//float debugValue;
	//CudaHelper::downloadBufferGPU(intensity, &debugValue, 1);
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

void Laser::reconstructShapeAABB(const ReconstructionInfo& nlosInfo)
{
	if (nlosInfo._captureSystem == CaptureSystem::Confocal)
		reconstructAABBConfocal(nlosInfo);
	else if (nlosInfo._captureSystem == CaptureSystem::Exhaustive)
		reconstructAABBExhaustive(nlosInfo);
}

void Laser::reconstructShapeDepths(const ReconstructionInfo& nlosInfo)
{
	if (nlosInfo._captureSystem == CaptureSystem::Confocal)
		reconstructDepthConfocal(nlosInfo);
	else if (nlosInfo._captureSystem == CaptureSystem::Exhaustive)
		reconstructDepthExhaustive(nlosInfo);
}

void Laser::reconstructDepthConfocal(const ReconstructionInfo& nlosInfo)
{
	ChronoUtilities::startTimer();

	//std::vector<double> reconstructionDepths = transientParameters._reconstructionDepths;
	std::vector<double> reconstructionDepths = linearSpace(0.25, 0.75, 200);
	assert(!reconstructionDepths.empty());

	// Pointer to reconstruction depths in GPU memory
	double* depthsGPU = nullptr;
	CudaHelper::initializeBufferGPU(depthsGPU, reconstructionDepths.size(), reconstructionDepths.data());

	// Let's initialize a matrix of size (sx, sy, depths)
	const glm::uint numDepths = static_cast<glm::uint>(reconstructionDepths.size());
	const glm::uint numPixels = nlosInfo._numSensorTargets * numDepths;
	float* activationGpu = nullptr;
	CudaHelper::initializeZeroBufferGPU(activationGpu, numPixels);

	// Determine size of thread groups and threads within them
	dim3 blockSize(BLOCK_C, BLOCK_D);
	dim3 gridSize(
		(nlosInfo._numLaserTargets + blockSize.x - 1) / blockSize.x,
		(nlosInfo._numLaserTargets + blockSize.y - 1) / blockSize.y,			
		(reconstructionDepths.size() + blockSize.z - 1) / blockSize.z);

	backprojectConfocal<<<gridSize, blockSize>>>(activationGpu, depthsGPU, numDepths);
	normalizeMatrix(activationGpu, numPixels);
	CudaHelper::synchronize("backprojectConfocal");

	std::cout << "Reconstruction finished in " << ChronoUtilities::getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	// Prepare buckets and info for storing data in the local system
	const std::string outputFolder = "output/";
	const uint16_t cameraTargetRes = static_cast<uint16_t>(glm::sqrt(nlosInfo._numSensorTargets));
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

void Laser::reconstructDepthExhaustive(const ReconstructionInfo& nlosInfo)
{
	ChronoUtilities::startTimer();

	//std::vector<double> reconstructionDepths = transientParameters._reconstructionDepths;
	std::vector<double> reconstructionDepths = linearSpace(0.9, 1.1, 200);
	assert(!reconstructionDepths.empty());

	// Pointer to reconstruction depths in GPU memory
	double* depthsGPU = nullptr;
	CudaHelper::initializeBufferGPU(depthsGPU, reconstructionDepths.size(), reconstructionDepths.data());

	// Let's initialize a matrix of size (sx, sy, depths)
	const glm::uint numDepths = static_cast<glm::uint>(reconstructionDepths.size());
	const glm::uint numPixels = nlosInfo._numSensorTargets * numDepths;
	float* activationGpu = nullptr;
	CudaHelper::initializeZeroBufferGPU(activationGpu, numPixels);

	// Determine size of thread groups and threads within them
	constexpr glm::uint NUM_SPLITS = 16;

	dim3 blockSize(BLOCK_C, BLOCK_D);
	//dim3 gridSize(
	//	(nlosInfo._numLaserTargets / NUM_SPLITS  + nlosInfo._numSensorTargets / NUM_SPLITS + blockSize.x - 1) / blockSize.x,
	//	(nlosInfo._numLaserTargets + blockSize.y - 1) / blockSize.y,
	//	(reconstructionDepths.size() + blockSize.z - 1) / blockSize.z);

	//for (glm::uint laserSplitIdx = 0; laserSplitIdx < NUM_SPLITS; ++laserSplitIdx)
	//{
	//	for (glm::uint sensorSplitIdx = 0; sensorSplitIdx < NUM_SPLITS; ++sensorSplitIdx)
	//	{
	//		backprojectExhaustive<<<gridSize, blockSize>>>(activationGpu, depthsGPU, numDepths);
	//	}

	//	std::cout << "Laser split: " << laserSplitIdx << "/" << NUM_SPLITS << '\n';
	//	std::cout.flush();
	//}

	dim3 gridSize(
		(nlosInfo._numLaserTargets + blockSize.x - 1) / blockSize.x,
		(nlosInfo._numSensorTargets + blockSize.y - 1) / blockSize.y,
		(reconstructionDepths.size() + blockSize.z - 1) / blockSize.z);

	for (glm::uint c = 0; c < nlosInfo._numLaserTargets; ++c)
		backproject<<<gridSize, blockSize>>>(activationGpu, depthsGPU, numDepths, c);

	normalizeMatrix(activationGpu, numPixels);
	CudaHelper::synchronize("backprojectExhaustive");

	std::cout << "Reconstruction finished in " << ChronoUtilities::getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	// Prepare buckets and info for storing data in the local system
	const std::string outputFolder = "output/";
	const uint16_t cameraTargetRes = static_cast<uint16_t>(glm::sqrt(nlosInfo._numSensorTargets));
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

void Laser::reconstructAABBConfocal(const ReconstructionInfo& nlosInfo)
{
	const glm::uvec3 voxelResolution = nlosInfo._voxelResolution;
	const glm::uint numVoxels = voxelResolution.x * voxelResolution.y * voxelResolution.z;
	const glm::uint sliceSize = voxelResolution.x * voxelResolution.y;
	float* activationGPU = nullptr;
	CudaHelper::initializeZeroBufferGPU(activationGPU, numVoxels);

	//// Determine size of thread groups and threads within them
	dim3 blockSize(8, 8, 8);
	dim3 gridSize(
		(nlosInfo._numLaserTargets + blockSize.x - 1) / blockSize.x,
		(sliceSize + blockSize.y - 1) / blockSize.y);

	for (glm::uint z = 0; z < voxelResolution.z; ++z)
	{
		backprojectVoxel<<<gridSize, blockSize>>>(activationGPU, sliceSize, z);
		CudaHelper::synchronize("backprojectVoxel");

		std::cout << "Voxel slice: " << z << "/" << voxelResolution.z << '\n';
	}

	normalizeMatrix(activationGPU, numVoxels);

	// Prepare buckets and info for storing data in the local system
	const std::string outputFolder = "output/";
	std::vector<float> voxels(numVoxels);
	CudaHelper::downloadBufferGPU(activationGPU, voxels.data(), numVoxels);
	FileUtilities::write<float>(outputFolder + "aabb.cube", voxels);

	CudaHelper::free(activationGPU);
}

void Laser::reconstructAABBExhaustive(const ReconstructionInfo& nlosInfo)
{
}