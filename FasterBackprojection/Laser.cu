// ReSharper disable CppExpressionWithoutSideEffects
#include "stdafx.h"
#include "Laser.cuh"

#include <valarray>
#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>

#include "ChronoUtilities.h"
#include "GpuStructs.cuh"
#include "fourier.cuh"
#include "transient_processing.cuh"
#include "transient_reconstruction.cuh"

#include "CudaHelper.h"
#include "FileUtilities.h"
#include "RandomUtilities.h"
#include "TransientImage.h"

//

Laser::Laser(NLosData* nlosData) : _nlosData(nlosData)
{
}

Laser::~Laser()
{
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

	for (size_t i = 0; i < nt_pad; ++i)
	{
		float val = (t_vals[i] - t_max * 0.5f) / wl_sigma;
		gaussianEnvelope[i] = std::exp(-(val * val) * 0.5f);
		sumGaussianEnvelope += gaussianEnvelope[i];
	}

	for (size_t i = 0; i < nt_pad; ++i) 
	{
		float phase = TWO_PI * t_vals[i] / wl_mean;
		K[i].x = gaussianEnvelope[i] * std::cos(phase) / sumGaussianEnvelope;
		K[i].y = gaussianEnvelope[i] * std::sin(phase) / sumGaussianEnvelope;
	}

	// This prepares K for FFT since cuFFT computes the unshifted FFT
	std::vector<cufftComplex> K_ifftShifted(nt_pad);
	size_t shift = nt_pad / 2;  
	for (size_t i = 0; i < nt_pad; ++i) 
		K_ifftShifted[i] = K[(i + shift) % nt_pad];
	K = K_ifftShifted;

	// Pad H in host
	size_t padding = (nt_pad - nt) / 2;
	std::vector<cufftComplex> H_pad_host;
	padIntensity(H_pad_host, padding, border);

	// Transfer H_pad_host and K to device
	cufftComplex* d_H = nullptr, *d_K = nullptr;
	CudaHelper::initializeBufferGPU(d_H, H_pad_host.size(), H_pad_host.data());
	CudaHelper::initializeBufferGPU(d_K, K.size(), K.data());

	//
	size_t dimProduct = 1;
	std::vector<size_t> newDims = _nlosData->_dims;
	newDims[newDims.size() - 1] = nt_pad; // Last dimension is time, set to padded size
	for (size_t dim : newDims)
		dimProduct *= dim;

	cufftHandle planH;
	int n[1] = { static_cast<int>(nt_pad) };
	int rank = 1;
	int istride = 1, ostride = 1;												// stride between elements in time dimension (contiguous)
	int idist = static_cast<int>(nt_pad), odist = static_cast<int>(nt_pad);		// distance between consecutive batches in output
	int batch = static_cast<int>(dimProduct / nt_pad);

	// Create 1D FFT plan for batches
	CUFFT_CHECK(cufftPlanMany(&planH, rank, n, NULL, istride, idist, NULL, ostride, odist, CUFFT_C2C, batch));
	CUFFT_CHECK(cufftExecC2C(planH, d_H, d_H, CUFFT_FORWARD));

	//
	cufftHandle planK;
	CUFFT_CHECK(cufftPlan1d(&planK, (int)nt_pad, CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(planK, d_K, d_K, CUFFT_FORWARD));

	// 
	dim3 block(256);
	dim3 grid((batch * nt_pad + block.x - 1) / block.x);

	multiplyHK<<<grid, block>>>(d_H, d_K, batch, nt_pad);
	cudaDeviceSynchronize();

	CUFFT_CHECK(cufftExecC2C(planH, d_H, d_H, CUFFT_INVERSE));

	//
	normalizeH<<<grid, block>>>(d_H, batch, nt_pad);

	// Copy result back to host
	CudaHelper::downloadBufferGPU(d_H, H_pad_host.data(), dimProduct, 0);

	//
	float phaseCorrection = atan2(H_pad_host[0].y, H_pad_host[0].x);
	size_t inner = nt, outer = dimProduct / nt_pad;  

	for (size_t i = 0; i < outer; ++i) {
		size_t paddedOffset = i * nt_pad + padding;
		size_t unpaddedOffset = i * nt;

		for (size_t j = 0; j < inner; ++j)
		{
			cufftComplex val = H_pad_host[paddedOffset + j];
			_nlosData->_data[unpaddedOffset + j] = val.x * cos(phaseCorrection) - val.y * sin(phaseCorrection);
		}
	}

	// 
	CUFFT_CHECK(cufftDestroy(planH));
	CudaHelper::free(d_H);
	CudaHelper::free(d_K);
}

void Laser::reconstructShape(ReconstructionInfo& recInfo, ReconstructionBuffers& recBuffers, bool reconstructAABB)
{
	CudaHelper::checkError(cudaMemcpyToSymbol(rtRecInfo, &recInfo, sizeof(ReconstructionInfo)));
	CudaHelper::checkError(cudaMemcpyToSymbol(laserTargets, &recBuffers._laserTargets, sizeof(glm::vec3*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(sensorTargets, &recBuffers._sensorTargets, sizeof(glm::vec3*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(intensityCube, &recBuffers._intensity, sizeof(float*)));

	std::cout << "Reconstructing shape...\n";

	if (reconstructAABB)
		reconstructShapeAABB(recInfo, recBuffers);
	else
		reconstructShapeDepths(recInfo);

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

void Laser::padIntensity(std::vector<cufftComplex>& paddedIntensity, size_t padding, const std::string& mode) const
{
	size_t timeDim = _nlosData->_dims.size() - 1;
	size_t nt = _nlosData->_dims[timeDim];
	size_t nt_pad = nt + 2 * padding;

	std::vector<size_t> paddedDims = _nlosData->_dims;
	paddedDims[timeDim] = nt_pad;

	size_t sliceSize = 1;
	for (size_t i = 0; i < _nlosData->_dims.size() - 1; ++i)
		sliceSize *= _nlosData->_dims[i];

	paddedIntensity.resize(sliceSize * nt_pad);

	#pragma omp parallel for
	for (size_t i = 0; i < sliceSize; ++i)
	{
		for (size_t t = 0; t < nt; ++t)
			paddedIntensity[i * nt_pad + (t + padding)] = cufftComplex{ _nlosData->_data[i * nt + t], 0.0f };

		if (mode == "constant" || mode == "zero")
		{
			for (size_t t = 0; t < padding; ++t) {
				paddedIntensity[i * nt_pad + t] = { 0.0f, 0.0f };
			}
			for (size_t t = nt + padding; t < nt_pad; ++t) {
				paddedIntensity[i * nt_pad + t] = { 0.0f, 0.0f };
			}
		}
		else if (mode == "edge")
		{
			float first = _nlosData->_data[i * nt + 0];
			float last = _nlosData->_data[i * nt + nt - 1];

			for (size_t t = 0; t < padding; ++t)
				paddedIntensity[i * nt_pad + t] = cufftComplex{ first, .0f };
			for (size_t t = nt + padding; t < nt_pad; ++t)
				paddedIntensity[i * nt_pad + t] = cufftComplex{ last, .0f };
		}
		else
		{
			std::cerr << "Unsupported padding mode: " << mode << '\n';
			return;
		}
	}
}

void Laser::fftLoG(float*& inputVoxels, const glm::uvec3& resolution, float sigma)
{
	// Calculate sizes
	glm::uint size = resolution.x * resolution.y * resolution.z;
	glm::uint complexDimensionSize = (resolution.x / 2 + 1) * resolution.y * resolution.z;

	// Allocate memory - note different sizes!
	cufftComplex* fourierReconstruction, * kernel;
	CudaHelper::initializeZeroBufferGPU(fourierReconstruction, complexDimensionSize);
	CudaHelper::initializeZeroBufferGPU(kernel, complexDimensionSize);

	// Create FFT plans
	cufftHandle forward_plan, inverse_plan;
	cufftPlan3d(&forward_plan, resolution.x, resolution.y, resolution.z, CUFFT_R2C);
	cufftPlan3d(&inverse_plan, resolution.x, resolution.y, resolution.z, CUFFT_C2R);

	// Forward FFT
	CUFFT_CHECK(cufftExecR2C(forward_plan, (cufftReal*)inputVoxels, fourierReconstruction));

	// Build LoG kernel - must respect complex_size layout!
	dim3 block(8, 8, 8);
	dim3 grid(
		((resolution.x / 2 + 1) + block.x - 1) / block.x,
		(resolution.y + block.y - 1) / block.y,
		(resolution.z + block.z - 1) / block.z
	);
	buildLoGKernel3D<<<grid, block>>>(kernel, resolution.x, resolution.y, resolution.z, sigma);

	// Multiply in frequency domain
	glm::uint threads = 256;
	glm::uint blocks = CudaHelper::getNumBlocks(complexDimensionSize, threads);
	multiplyKernel<<<blocks, threads>>>(fourierReconstruction, kernel, complexDimensionSize);

	// Inverse FFT
	CUFFT_CHECK(cufftExecC2R(inverse_plan, fourierReconstruction, (cufftReal*)inputVoxels));

	// Normalize
	normalizeIFFT<<<blocks, threads>>>(inputVoxels, size);

	// Cleanup
	cufftDestroy(forward_plan);
	cufftDestroy(inverse_plan);
	CudaHelper::free(fourierReconstruction);
	CudaHelper::free(kernel);
}

void Laser::laplacianFilter(float*& inputVoxels, const glm::uvec3& resolution, glm::uint filterSize)
{
	float* laplacianGPU = nullptr;
	CudaHelper::initializeZeroBufferGPU(laplacianGPU, static_cast<size_t>(resolution.x) * resolution.y * resolution.z);

	dim3 blockSize = dim3(8, 8, 8);
	dim3 gridSize = dim3(
		(resolution.x + blockSize.x - 1) / blockSize.x,
		(resolution.y + blockSize.y - 1) / blockSize.y,
		(resolution.z + blockSize.z - 1) / blockSize.z
	);
	laplace<<<gridSize, blockSize>>>(inputVoxels, laplacianGPU, resolution, static_cast<float>(filterSize));
	std::swap(laplacianGPU, inputVoxels);
	CudaHelper::free(laplacianGPU);
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

void Laser::reconstructShapeAABB(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	if (recInfo._captureSystem == CaptureSystem::Confocal)
		//reconstructAABBConfocal(recInfo);
		reconstructAABBConfocalMIS(recInfo, recBuffers);
	else if (recInfo._captureSystem == CaptureSystem::Exhaustive)
		reconstructAABBExhaustive(recInfo);
}

void Laser::reconstructShapeDepths(const ReconstructionInfo& recInfo)
{
	//std::vector<double> reconstructionDepths = transientParameters._reconstructionDepths;
	std::vector<double> reconstructionDepths = linearSpace(recInfo.getFocusDepth() - 0.1f, recInfo.getFocusDepth() + 0.1f, 200);
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

void Laser::reconstructAABBConfocal(const ReconstructionInfo& recInfo)
{
	const glm::uvec3 voxelResolution = recInfo._voxelResolution;
	const glm::uint numVoxels = voxelResolution.x * voxelResolution.y * voxelResolution.z;
	const glm::uint sliceSize = voxelResolution.x * voxelResolution.y;
	float* activationGPU = nullptr;
	CudaHelper::initializeZeroBufferGPU(activationGPU, numVoxels);

	ChronoUtilities::startTimer();

	dim3 blockSize(16, 16); 
	dim3 gridSize(
		(sliceSize + blockSize.x - 1) / blockSize.x,    
		(recInfo._numLaserTargets + blockSize.y - 1) / blockSize.y, 
		(voxelResolution.z + blockSize.z - 1) / blockSize.z  
	);

	backprojectConfocalVoxel<<<gridSize, blockSize>>>(activationGPU, sliceSize);
	normalizeMatrix(activationGPU, numVoxels);
	CudaHelper::synchronize("backprojectConfocalVoxel");

	//fftLoG(activationGPU, voxelResolution, 1.0f);
	laplacianFilter(activationGPU, voxelResolution, 5);

	std::cout << "Reconstruction finished in " << ChronoUtilities::getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	// Prepare buckets and info for storing data in the local system
	const std::string outputFolder = "output/";
	std::vector<float> voxels(numVoxels);
	CudaHelper::downloadBufferGPU(activationGPU, voxels.data(), numVoxels);
	FileUtilities::write<float>(outputFolder + "aabb.cube", voxels);

	CudaHelper::free(activationGPU);
}

void Laser::reconstructAABBExhaustive(const ReconstructionInfo& recInfo)
{
}

void Laser::reconstructAABBConfocalMIS(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	ChronoUtilities::startTimer();

	const glm::uvec3 voxelResolution = recInfo._voxelResolution;
	const glm::uint numVoxels = voxelResolution.x * voxelResolution.y * voxelResolution.z;
	const glm::uint numTimeBins = recInfo._numTimeBins;
	const glm::uint sliceSize = voxelResolution.x * voxelResolution.y;

	glm::uint totalElements = 1;
	for (const auto& dim : _nlosData->_dims)
		totalElements *= static_cast<glm::uint>(dim);

	float* activationGPU = nullptr;
	CudaHelper::initializeZeroBufferGPU(activationGPU, numVoxels);

	float* spatioTemporalSum = nullptr, * spatialSum = nullptr;
	CudaHelper::initializeBufferGPU(spatioTemporalSum, totalElements);
	CudaHelper::initializeBufferGPU(spatialSum, totalElements / numTimeBins);

	float* noise = nullptr;
	std::vector<float> noiseHost(numVoxels);
	for (glm::uint i = 0; i < numVoxels; ++i)
		noiseHost[i] = RandomUtilities::getUniformRandom();
	CudaHelper::initializeBufferGPU(noise, numVoxels, noiseHost.data());

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
		spatioTemporalPrefixSum<<<blocks, blockSize>>>(spatioTemporalSum, spatialSum, totalElements / numTimeBins, numTimeBins);
	}

	// Normalize spatio-temporal sum
	{
		glm::uint threadsBlock = 512, numBlocks = CudaHelper::getNumBlocks(totalElements / numTimeBins, threadsBlock);
		normalizeSpatioTemporalPrefixSum<<<numBlocks, threadsBlock>>>(spatioTemporalSum, numTimeBins, totalElements);
	}

	// Normalize the spatial sum
	normalizeMatrix(spatialSum, totalElements / numTimeBins);

	//std::vector<float> spatialSumHost(totalElements / numTimeBins);
	//CudaHelper::downloadBufferGPU(spatialSum, spatialSumHost.data(), totalElements / numTimeBins);

	//std::vector<float> timePrefixSumHost(numTimeBins * 2);
	//CudaHelper::downloadBufferGPU(spatioTemporalSum, timePrefixSumHost.data(), numTimeBins * 2, numTimeBins);

	glm::uint numSamples = recInfo._numLaserTargets / 4;
	dim3 blockSize(BLOCK_X_CONFOCAL, 4);
	dim3 gridSize(
		(sliceSize + blockSize.x - 1) / blockSize.x,
		(numSamples + blockSize.y - 1) / blockSize.y,
		(voxelResolution.z + blockSize.z - 1) / blockSize.z
	);

	backprojectConfocalVoxelMIS<<<gridSize, blockSize>>>(spatialSum, activationGPU, noise, sliceSize, numSamples, numVoxels);
	normalizeMatrix(activationGPU, numVoxels);
	CudaHelper::synchronize("backprojectConfocalVoxelMIS");

	laplacianFilter(activationGPU, voxelResolution, 5);

	std::cout << "Reconstruction finished in " << ChronoUtilities::getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	// Prepare buckets and info for storing data in the local system
	const std::string outputFolder = "output/";
	std::vector<float> voxels(numVoxels);
	CudaHelper::downloadBufferGPU(activationGPU, voxels.data(), numVoxels);
	FileUtilities::write<float>(outputFolder + "aabb.cube", voxels);

	CudaHelper::free(activationGPU);
	CudaHelper::free(spatioTemporalSum);
	CudaHelper::free(spatialSum);
	CudaHelper::free(noise);
}