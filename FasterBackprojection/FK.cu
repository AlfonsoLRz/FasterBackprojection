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

	if (transientParams._compensateLaserCosDistance)
		compensateLaserCosDistance(recInfo, recBuffers);

	if (recInfo._captureSystem == CaptureSystem::Confocal)
		reconstructVolumeConfocal(nullptr, recInfo, recBuffers);
	else
		throw std::runtime_error("Unsupported capture system for LCT reconstruction.");

	std::cout << "Reconstruction finished in " << getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	if (transientParams._saveMaxImage)
		FK::saveMaxImage(
			transientParams._outputFolder + transientParams._outputMaxImageName,
			recBuffers._intensity,
			glm::uvec3(nlosData->_dims[0], nlosData->_dims[1], nlosData->_dims[2]));
}

//

void FK::reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers) const
{
	const glm::uvec3 volumeResolution = glm::uvec3(_nlosData->_dims[0], _nlosData->_dims[1], _nlosData->_dims[2]);
	const glm::uvec3 fftVolumeResolution = volumeResolution * 2u;
	float* intensityGpu = recBuffers._intensity;

	cufftComplex* fft = nullptr, *fftAux = nullptr;
	CudaHelper::initializeZeroBufferGPU(fft, static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z);
	CudaHelper::initializeBufferGPU(fftAux, static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z);

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

	ChronoUtilities::startTimer();

	// Perform forward FFT on the intensity data
	const float divisor = 1.0f / static_cast<float>(volumeResolution.z);
	padIntensityFFT_FK<<<gridSize, blockSize>>>(intensityGpu, fft, volumeResolution, fftVolumeResolution, divisor);
	CudaHelper::synchronize("padIntensityFFT");

	std::cout << "FFT padding took " << getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	// Calculate sqrt term
	float width = _nlosData->_temporalWidth, range = recInfo._timeStep * static_cast<float>(recInfo._numTimeBins);
	float sqrtConst = (static_cast<float>(volumeResolution.x) * range) / (static_cast<float>(volumeResolution.z) * width * 4.0f);

	//
	cufftHandle planH;
	int rank = 3; 
	int n[3] = { static_cast<int>(fftVolumeResolution[0]),
				 static_cast<int>(fftVolumeResolution[1]),
				 static_cast<int>(fftVolumeResolution[2]) };

	ChronoUtilities::startTimer();

	CUFFT_CHECK(cufftPlanMany(&planH, rank, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(planH, fft, fft, CUFFT_FORWARD));
	CudaHelper::synchronize("cufftExecC2C");

	std::cout << "FFT execution took " << getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	// Shift FFT
	//ChronoUtilities::startTimer();

	//shiftFFT<<<gridSizeFFT, blockSizeFFT>>>(fft, fftAux, fftVolumeResolution, fftVolumeResolution / 2u);
	//CudaHelper::synchronize("shiftFFT");

	//std::cout << "FFT shifting took " << getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	// Cleanup
	ChronoUtilities::startTimer();

	float maxValue = sqrtf(divisor * divisor * 2.0f + 1.0f);
	stoltKernel<<<gridSizeFFT, blockSizeFFT>>>(fft, fftAux, volumeResolution, fftVolumeResolution, fftVolumeResolution / 2u, sqrtConst * sqrtConst, maxValue);
	CudaHelper::synchronize("stoltKernel");

	std::cout << "Stolt kernel computation took " << getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	//
	//shiftFFT<<<gridSizeFFT, blockSizeFFT>>>(fftAux, fft, fftVolumeResolution, fftVolumeResolution / 2u);
	//CudaHelper::synchronize("unshiftFFT");

	// Inverse FFT
	CUFFT_CHECK(cufftExecC2C(planH, fftAux, fftAux, CUFFT_INVERSE));

	// IFFT requires normalization, but it also produces very small values, so we avoid this and produce valid results by normalizing later
	//size_t fftSize = static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z;
	//normalizeIFFT<<<CudaHelper::getNumBlocks(fftSize, 512), 512>>>(fftAux, fftSize, 1.0f / static_cast<float>(fftSize));

	// Inverse padding
	unpadIntensityFFT_FK<<<gridSize, blockSize>>>(intensityGpu, fftAux, volumeResolution, fftVolumeResolution);
	CudaHelper::synchronize("unpadIntensityFFT");

	CudaHelper::free(fft);
	CudaHelper::free(fftAux);
	CUFFT_CHECK(cufftDestroy(planH));
}

void FK::reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo)
{
}

