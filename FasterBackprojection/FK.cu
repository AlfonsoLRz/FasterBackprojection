#include "stdafx.h"
#include "FK.h"

#include "CudaHelper.h"

#include "fk.cuh"
#include "fourier.cuh"

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

	if (recInfo._captureSystem == CaptureSystem::Confocal)
		reconstructVolumeConfocal(nullptr, recInfo, recBuffers);
	else
		throw std::runtime_error("Unsupported capture system for LCT reconstruction.");

	std::cout << "Reconstruction finished in " << getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";
}

//

void FK::reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	const glm::uvec3 volumeResolution = glm::uvec3(_nlosData->_dims[0], _nlosData->_dims[1], _nlosData->_dims[2]);
	const glm::uvec3 fftVolumeResolution = volumeResolution * 2u;
	float* intensityGpu = recBuffers._intensity;

	cufftComplex* fft = nullptr;
	CudaHelper::initializeZeroBufferGPU(fft, static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z);

	float* sqrtTerm = nullptr;
	CudaHelper::initializeBufferGPU(sqrtTerm, static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z);

	dim3 blockSize(8, 8, 8);
	dim3 gridSize(
		(volumeResolution.x + blockSize.x - 1) / blockSize.x,
		(volumeResolution.y + blockSize.y - 1) / blockSize.y,
		(volumeResolution.z + blockSize.z - 1) / blockSize.z
	);

	dim3 blockSizeFFT(8, 8, 8);
	dim3 gridSizeFFT(
		(fftVolumeResolution.x + blockSizeFFT.x - 1) / blockSizeFFT.x,
		(fftVolumeResolution.y + blockSizeFFT.y - 1) / blockSizeFFT.y,
		(fftVolumeResolution.z + blockSizeFFT.z - 1) / blockSizeFFT.z
	);

	ChronoUtilities::startTimer();

	// Perform forward FFT on the intensity data
	padIntensityFFT<<<gridSize, blockSize>>>(intensityGpu, fft, volumeResolution, fftVolumeResolution);
	CudaHelper::synchronize("padIntensityFFT");

	// Calculate sqrt term
	float width = _nlosData->_temporalWidth, range = recInfo._timeStep * recInfo._numTimeBins;
	float divisor = (float(volumeResolution.x) * range) / (float(volumeResolution.z) * width * 4);
	calculateSqrtTerm<<<gridSizeFFT, blockSizeFFT>>>(sqrtTerm, volumeResolution, fftVolumeResolution, divisor * divisor);
	CudaHelper::synchronize("calculateSqrtTerm");

	
	size_t newDimProduct = static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z;

	//
	cufftHandle planH;
	int rank = 3;  // 2D FFT
	int n[3] = { static_cast<int>(fftVolumeResolution[0]),
				 static_cast<int>(fftVolumeResolution[1]),
				 static_cast<int>(fftVolumeResolution[2]) };

	CUFFT_CHECK(cufftPlanMany(&planH, rank, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(planH, fft, fft, CUFFT_FORWARD));

	// Multiply by inverse PSF
	//multiplyPSF << <CudaHelper::getNumBlocks(newDimProduct, 512), 512 >> > (d_H, inversePSF, newDimProduct);
	//CudaHelper::synchronize("multiplyHK");

	// Inverse FFT
	CUFFT_CHECK(cufftExecC2C(planH, fft, fft, CUFFT_INVERSE));
	CUFFT_CHECK(cufftDestroy(planH));

	// IFFT requires normalization, but it also produces very small values, so we avoid this and produce valid results by normalizing later
	//normalizeIFFT<<<CudaHelper::getNumBlocks(newDimProduct, 512), 512>>>(d_H, newDimProduct, 1.0f / newDimProduct);

	// Inverse padding
	unpadIntensityFFT<<<gridSize, blockSize>>>(intensityGpu, fft, volumeResolution, fftVolumeResolution);
	//CudaHelper::synchronize("unpadIntensityFFT");
}

void FK::reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo)
{
}

