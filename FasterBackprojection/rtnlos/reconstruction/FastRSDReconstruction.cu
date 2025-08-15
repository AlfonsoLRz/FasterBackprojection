#include "../stdafx.h"
#include "FastRSDReconstruction.h"

#include <cccl/cub/device/device_reduce.cuh>

#include "rsd_cuda_kernels.h"
#include "../CudaHelper.h"
#include "../../fourier.cuh"
#include "../../transient_postprocessing.cuh"
#include "../../CudaPerf.h"
#include "../../ViewportSurface.h"

//

FastRSDReconstruction::FastRSDReconstruction()
	  : _rsd(nullptr)
	  , _imageConvolution(nullptr), _cubeImages(nullptr), _dWeights(nullptr)
	  , _fftPlan2D(0)
	  , _blockSize1D(0), _gridSize1D(0)
{
}

FastRSDReconstruction::~FastRSDReconstruction()
{
	FastReconstructionAlgorithm::destroyResources();
	FastRSDReconstruction::destroyResources();
}

void FastRSDReconstruction::destroyResources()
{
	FastReconstructionAlgorithm::destroyResources();

	CudaHelper::reset(_rsd);
	CudaHelper::reset(_imageConvolution);
	CudaHelper::reset(_cubeImages);
	CudaHelper::reset(_dWeights);

	if (_fftPlan2D != 0)
	{
		CUFFT_CHECK(cufftDestroy(_fftPlan2D));
		_fftPlan2D = 0;
	}
}

void FastRSDReconstruction::precalculate()
{
	FastReconstructionAlgorithm::precalculate();

	// Launch dimensions for the kernels
	_blockSize1D = 512;
	_gridSize1D = CudaHelper::getNumBlocks(_sliceSize, _blockSize1D);

	_blockSize2D_freq = dim3(64, 8);
	_gridSize2D_freq = dim3(
		(_sliceSize + _blockSize2D_freq.x - 1) / _blockSize2D_freq.x,
		(_numFrequencies + _blockSize2D_freq.y - 1) / _blockSize2D_freq.y
	);

	_blockSize2D_depth = dim3(64, 8);
	_gridSize2D_depth = dim3(
		(_sliceSize + _blockSize2D_depth.x - 1) / _blockSize2D_depth.x,
		(_numDepths + _blockSize2D_depth.y - 1) / _blockSize2D_depth.y
	);

	_blockSize2D_pix = dim3(16, 16);
	_gridSize2D_pix = dim3(
		(_imageWidth + _blockSize2D_pix.x - 1) / _blockSize2D_pix.x,
		(_imageHeight + _blockSize2D_pix.y - 1) / _blockSize2D_pix.y
	);

	// RSD
	CudaHelper::initializeBuffer(_rsd, sliceNumPixels() * _numDepths * _numFrequencies);
	CUFFT_CHECK(cufftPlan2d(&_fftPlan2D, _imageHeight, _imageWidth, CUFFT_C2C));

	//
	glm::uint depthIdx;
	float depth;

	for (depth = _info._minDistance, depthIdx = 0; depth < _info._maxDistance; depth += _info._deltaDistance, depthIdx++) 
	{
		float time = (depth + _info._distanceOffset) / LIGHT_SPEED;

		for (int waveIdx = 0; waveIdx < _numFrequencies; waveIdx++) 
		{
			float lambda = _lambdas[waveIdx];
			float omega = _omegas[waveIdx];

			cufftComplex* kernelPtr = _rsd + depthIdx * _frequencyCubeSize + waveIdx * _sliceSize;
			RSDKernelConvolution(kernelPtr, _fftPlan2D, lambda, omega, depth, time, _cudaStreams[waveIdx]);
		}

		synchronizeStreams(_numFrequencies);
	}
	assert(i == _numDepths);

	// Now allocate all the storage that the reconstructImage() function will need
	CudaHelper::initializeBuffer(_dWeights, _weights.size(), _weights.data());
	CudaHelper::initializeBuffer(_imageConvolution, sliceNumPixels() * _numDepths);

	// 4 cubes, so we can store 3 reconstructions for dda, plus a temporary one for computation
	CudaHelper::initializeBuffer(_cubeImages, sliceNumPixels() * _numDepths * 4);
}

// call after each time full set of images has been added
void FastRSDReconstruction::reconstructImage(ViewportSurface* viewportSurface)
{
	assert(_precalculated);

	_currentCount++;

	for (int frequencyIdx = 0; frequencyIdx < _numFrequencies; ++frequencyIdx)
	{
		CUFFT_CHECK(cufftSetStream(_fftPlan2D, _cudaStreams[frequencyIdx]));
		CUFFT_CHECK(cufftExecC2C(_fftPlan2D,
			_imageData + frequencyIdx * _sliceSize,
			_imageData + frequencyIdx * _sliceSize,
			CUFFT_FORWARD));
	}
	synchronizeStreams(_numFrequencies);

	//
	int depthIndex;
	float depth;

	// Temporal output wavefront at the depth plane
	cudaMemset(_imageConvolution, 0, _sliceSize * _numDepths * sizeof(cufftComplex));
	glm::uint idx = _currentCount % 3;

	for (depth = _info._minDistance, depthIndex = 0; depth < _info._maxDistance; depth += _info._deltaDistance, ++depthIndex)
	{
		multiplySpectrumManyAndScale<<<_gridSize2D_freq, _blockSize2D_freq, 0, _cudaStreams[depthIndex]>>>(
			_imageData,
			_rsd + depthIndex * _frequencyCubeSize,
			_imageConvolution + depthIndex * _sliceSize,
			_dWeights,
			_numFrequencies, _sliceSize);

		// IFFT after integration
		CUFFT_CHECK(cufftSetStream(_fftPlan2D, _cudaStreams[depthIndex]));
		CUFFT_CHECK(cufftExecC2C(
			_fftPlan2D, _imageConvolution + depthIndex * _sliceSize, _imageConvolution + depthIndex * _sliceSize, CUFFT_INVERSE
		));

		// Store this slice
		abs<<<_gridSize1D, _blockSize1D, 0, _cudaStreams[depthIndex]>>>(
				_imageConvolution + depthIndex * _sliceSize,
				_cubeImages + idx * cubeNumPixels() + depthIndex * sliceNumPixels(), 
				_sliceSize);
	}
	synchronizeStreams(depthIndex);

	if (_currentCount > 0) 
	{
		// We have the next frame, so we can process the previous frame.
		glm::uint previousIdx = (_currentCount - 1) % 3;
		if (_useDDA) 
		{
			// perform depth dependent averaging across the 3 stored frames
			DDA<<<_gridSize2D_depth, _blockSize2D_depth>>>(
				_cubeImages, previousIdx, _ddaWeights,
				_sliceSize, _sliceSize * _numDepths, _numDepths);

			maxZ<<<_gridSize2D_pix, _blockSize2D_pix>>>(
				_cubeImages + 3 * _sliceSize * _numDepths, _imageResult,
				_imageWidth, _imageHeight, _numDepths, _sliceSize, glm::uvec2(_imageWidth / 2, _imageHeight / 2));
		}
		else 
		{
			// Depth dependent averaging is turned off, just pick the max from the current (previous) single frame.
			maxZ<<<_gridSize2D_pix, _blockSize2D_pix>>>(
				_cubeImages + previousIdx * _sliceSize * _numDepths, _imageResult,
				_imageWidth, _imageHeight, _numDepths, _sliceSize, glm::uvec2(_imageWidth / 2, _imageHeight / 2));
		}

		// Normalize in [0, 1]
		{
			cub::DeviceReduce::Max(_tempStorage, _tempStorageBytes, _imageResult, _maxValue, _sliceSize);
			cub::DeviceReduce::Min(_tempStorage, _tempStorageBytes, _imageResult, _minValue, _sliceSize);
			normalizeReconstruction<<<_gridSize1D, _blockSize1D>>>(_imageResult, _sliceSize, _maxValue, _minValue);
		}

		// Bandpass filter if interval is not [0, 1]
		if (_bandpassInterval.x > 0.0f && _bandpassInterval.y < 1.0f)
		{
			float lf = _bandpassInterval.x, hf = _bandpassInterval.y;
			float scale = 1.0f / (hf - lf);

			bandpassFilter<<<_gridSize1D, _blockSize1D>>>(_imageResult, _sliceSize, lf, scale);
		}

		// Write result into cudaSurface
		float4* texture;
		do
		{
			texture = viewportSurface->acquireDrawSurface();
		} while (!texture);

		writeImage<<<_gridSize1D, _blockSize1D>>>(_imageResult, _sliceSize, texture);
		viewportSurface->present();
	}
}

void FastRSDReconstruction::RSDKernelConvolution(
	cufftComplex* dKernel, cufftHandle fftPlan, 
	const float lambda, const float omega, 
	const float depth, const float t, cudaStream_t cudaStream) const
{
	float apertureX = _apertureDst[0];								// Physical aperture x, unit meter
	glm::uint dim_x = _imageWidth, dim_y = _imageHeight;
	float dx = apertureX / static_cast<float>(dim_x - 1);			// Spatial sampling density

	// Apply RSD convolution kernel equation
	float z_hat = depth / dx;
	float mulSquare = lambda * z_hat / (static_cast<float>(dim_x) * dx);
	rsdKernel<<<_gridSize2D_pix, _blockSize2D_pix, 0, cudaStream>>>(dKernel, z_hat * z_hat, mulSquare, dim_x, dim_y);

	// Perform fft on gpu
	CUFFT_CHECK(cufftSetStream(fftPlan, cudaStream));
	CUFFT_CHECK(cufftExecC2C(fftPlan, dKernel, dKernel, CUFFT_FORWARD));

	// Convolve the two by pointwise multiplication of the spectrums
	cufftComplex harmonic = { cos(omega * t), sin(omega * t) };
	multiplySpectrumExpHarmonic<<<_gridSize1D, _blockSize1D, 0, cudaStream>>>(dKernel, harmonic, _sliceSize);
}