#include "stdafx.h"
#include "FK.h"

#include "CudaHelper.h"
#include "fk.cuh"
#include "fourier.cuh"
#include "transient_postprocessing.cuh"

//

void FK::reconstructDepths(NLosData* nlosData, const ReconstructionInfo& recInfo,
                           const ReconstructionBuffers& recBuffers, const TransientParameters& transientParams,
                           const std::vector<float>& depths)
{
}

void FK::reconstructVolume(
	NLosData* nlosData, 
	const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers, const TransientParameters& transientParams)
{
	_nlosData = nlosData;

	_perf.setAlgorithmName("fk-migration");
	_perf.tic();

	compensateLaserCosDistance(transientParams, recInfo, recBuffers);

	if (recInfo._captureSystem == CaptureSystem::Confocal)
		reconstructVolumeConfocal(nullptr, recInfo, recBuffers);
	else
		throw std::runtime_error("Unsupported capture system for LCT reconstruction.");

	const glm::uvec3 volumeResolution = glm::uvec3(nlosData->_dims[0], nlosData->_dims[1], nlosData->_dims[2]);
	float* volumeGpu = recBuffers._intensity;

	// Post-process the activation matrix
	_perf.tic("Post-processing");
	_postprocessingFilters[transientParams._postprocessingFilterType]->compute(volumeGpu, volumeResolution, transientParams);
	_perf.toc();

	normalizeMatrix(volumeGpu, volumeResolution.x * volumeResolution.y * volumeResolution.z);

	_perf.toc();
	_perf.summarize();

	// Save volume & free resources
	if (transientParams._saveReconstructedBoundingBox)
		saveReconstructedAABB(
			transientParams._outputFolder + transientParams._outputAABBName, volumeGpu, 
			volumeResolution.x * volumeResolution.y * volumeResolution.z);

	if (transientParams._saveMaxImage)
		FK::saveMaxImage(
			transientParams._outputFolder + transientParams._outputMaxImageName,
			volumeGpu,
			glm::uvec3(nlosData->_dims[0], nlosData->_dims[1], nlosData->_dims[2]));
}

//

void FK::reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	const glm::uvec3 volumeResolution = glm::uvec3(_nlosData->_dims[0], _nlosData->_dims[1], _nlosData->_dims[2]);
	const glm::uvec3 fftVolumeResolution = glm::vec3(volumeResolution) * glm::vec3(2);

	_perf.tic("Resource allocation");

	cudaStream_t stream1, stream2;
	CudaHelper::createStreams({ &stream1, &stream2 });

	cufftHandle planH;
	cufftComplex* fft = nullptr, *fftAux = nullptr;
	CudaHelper::initializeZeroBufferAsync(
		fft, static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z,
		stream1
	);
	CudaHelper::initializeBufferAsync(
		fftAux, static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z,
		static_cast<cufftComplex*>(nullptr), stream2
	);
	float* intensityGpu = recBuffers._intensity;

	_perf.toc();

	dim3 blockSize(16, 8, 8);
	dim3 gridSize(
		(volumeResolution.z + blockSize.x - 1) / blockSize.x,
		(volumeResolution.y + blockSize.y - 1) / blockSize.y,
		(volumeResolution.x + blockSize.z - 1) / blockSize.z
	);

	dim3 blockSizeFFT(16, 8, 8);
	dim3 gridSizeFFT(
		(fftVolumeResolution.z + blockSizeFFT.x - 1) / blockSizeFFT.x,
		(fftVolumeResolution.y + blockSizeFFT.y - 1) / blockSizeFFT.y,
		(fftVolumeResolution.x + blockSizeFFT.z - 1) / blockSizeFFT.z
	);

	// Perform forward FFT on the intensity data
	CudaHelper::waitFor({ &stream1 });
	{
		_perf.tic("Pad Intensity FFT");

		float divisor = 1.0f / static_cast<float>(volumeResolution.z);
		fk::padIntensityFFT_FK<<<gridSize, blockSize>>>(intensityGpu, fft, volumeResolution, fftVolumeResolution, divisor);

		_perf.toc();
	}

	{
		_perf.tic("FFT");

		int rank = 3;
		int n[3] = { static_cast<int>(fftVolumeResolution[0]),
					 static_cast<int>(fftVolumeResolution[1]),
					 static_cast<int>(fftVolumeResolution[2]) };

		CUFFT_CHECK(cufftPlanMany(&planH, rank, n,
			NULL, 1, 0,
			NULL, 1, 0,
			CUFFT_C2C, 1));
		CUFFT_CHECK(cufftExecC2C(planH, fft, fft, CUFFT_FORWARD));

		_perf.toc();
	}

	// Stolt interpolation
	CudaHelper::waitFor({ &stream2 });
	{
		_perf.tic("Stolt");

		float width = _nlosData->_wallWidth, range = recInfo._timeStep * static_cast<float>(recInfo._numTimeBins);
		float sqrtConst = static_cast<float>(volumeResolution.x) * range / (static_cast<float>(volumeResolution.z) * width * 4.0f);

		dim3 paddedBlockSizeFFT(8, 8, 8);
		dim3 paddedGridSizeFFT(
			(fftVolumeResolution.z / 2 + paddedBlockSizeFFT.x - 1) / paddedBlockSizeFFT.x,
			(fftVolumeResolution.y + paddedBlockSizeFFT.y - 1) / paddedBlockSizeFFT.y,
			(fftVolumeResolution.x + paddedBlockSizeFFT.z - 1) / paddedBlockSizeFFT.z
		);

		fk::stoltKernel<<<paddedGridSizeFFT, paddedBlockSizeFFT>>>(
			fft, fftAux,
			fftVolumeResolution, fftVolumeResolution / 2u, 
			sqrtConst * sqrtConst);

		_perf.toc();
	}

	// Inverse FFT
	{
		_perf.tic("Inverse FFT");

		CUFFT_CHECK(cufftExecC2C(planH, fftAux, fftAux, CUFFT_INVERSE));

		_perf.toc();
	}

	// IFFT requires normalization, but it also produces very small values, so we avoid this and produce valid results by normalizing later
	//size_t fftSize = static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z;
	//normalizeIFFT<<<CudaHelper::getNumBlocks(fftSize, 512), 512>>>(fftAux, fftSize, 1.0f / static_cast<float>(fftSize));

	// Inverse padding
	{
		_perf.tic("Unpad intensity FFT");

		fk::unpadIntensityFFT_FK<<<gridSize, blockSize>>>(intensityGpu, fftAux, volumeResolution, fftVolumeResolution);

		_perf.toc();
	}

	spdlog::info("Allocated memory: {} MB", CudaHelper::getAllocatedMemory() / static_cast<size_t>(1024 * 1024));

	CudaHelper::freeAsync(fft, stream1);
	CudaHelper::freeAsync(fftAux, stream2);
	CUFFT_CHECK(cufftDestroy(planH));
	CudaHelper::waitFor({ &stream1, &stream2 });
	CudaHelper::destroyStreams({ &stream1, &stream2 });
}

void FK::reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo)
{
}

