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

void FK::bindInterpolationTexture(cufftComplex* data, const glm::uvec3& dataResolution) const
{
	//cudaResourceDesc resDesc = {};
	//resDesc.resType = cudaResourceTypeArray;

	//cudaArray* cuArray;
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	//cudaExtent extent = make_cudaExtent(dataResolution.x, dataResolution.y, dataResolution.z);
	//cudaMalloc3DArray(&cuArray, &channelDesc, extent);

	//// Copy data to 3D array (x, y, z order)
	//cudaMemcpy3DParms copyParams = { nullptr };
	//copyParams.srcPtr = make_cudaPitchedPtr(data,
	//	dataResolution.x * sizeof(float),
	//	dataResolution.x, dataResolution.y);
	//copyParams.dstArray = cuArray;
	//copyParams.extent = extent;
	//copyParams.kind = cudaMemcpyDeviceToDevice;
	//cudaMemcpy3D(&copyParams);

	//// Bind texture
	//texInterpData.normalized = false;  // Using explicit coordinates
	//texInterpData.filterMode = cudaFilterModeLinear;
	//cudaBindTextureToArray(texInterpData, cuArray, channelDesc);
}

void FK::reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	const glm::uvec3 volumeResolution = glm::uvec3(_nlosData->_dims[0], _nlosData->_dims[1], _nlosData->_dims[2]);
	const glm::uvec3 fftVolumeResolution = volumeResolution * 2u;
	float* intensityGpu = recBuffers._intensity;

	cufftComplex* fft = nullptr, *fftAux = nullptr;
	CudaHelper::initializeZeroBufferGPU(fft, static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z);
	CudaHelper::initializeZeroBufferGPU(fftAux, static_cast<size_t>(fftVolumeResolution.x) * fftVolumeResolution.y * fftVolumeResolution.z);

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
	float width = _nlosData->_temporalWidth, range = recInfo._timeStep * static_cast<float>(recInfo._numTimeBins);
	float divisor = (static_cast<float>(volumeResolution.x) * range) / (static_cast<float>(volumeResolution.z) * width * 4);

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

	// Shift FFT
	shiftFFT<<<gridSizeFFT, blockSizeFFT>>>(fft, fftAux, fftVolumeResolution, fftVolumeResolution / 2u);
	CudaHelper::synchronize("shiftFFT");

	// Stolt trick
	bindInterpolationTexture(fftAux, fftVolumeResolution);

	// Cleanup
	stoltKernel<<<gridSizeFFT, blockSizeFFT>>>(fftAux, volumeResolution, fftVolumeResolution, divisor);

	// Inverse FFT
	CUFFT_CHECK(cufftExecC2C(planH, fft, fft, CUFFT_INVERSE));
	CUFFT_CHECK(cufftDestroy(planH));

	// IFFT requires normalization, but it also produces very small values, so we avoid this and produce valid results by normalizing later
	//normalizeIFFT<<<CudaHelper::getNumBlocks(newDimProduct, 512), 512>>>(d_H, newDimProduct, 1.0f / newDimProduct);

	//
	unshiftFFT<<<gridSizeFFT, blockSizeFFT>>>(fft, fftAux, fftVolumeResolution, fftVolumeResolution / 2u);
	CudaHelper::synchronize("unshiftFFT");

	// Inverse padding
	unpadIntensityFFT<<<gridSize, blockSize>>>(intensityGpu, fft, volumeResolution, fftVolumeResolution);
	CudaHelper::synchronize("unpadIntensityFFT");

	CudaHelper::free(fft);
	CudaHelper::free(fftAux);
	CudaHelper::free(sqrtTerm);
	//cudaUnbindTexture(tex_f_data);
	//cudaFreeArray(cuArray);
}

void FK::reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo)
{
}

