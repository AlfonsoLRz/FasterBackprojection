#include "../stdafx.h"
#include "FastReconstructionAlgorithm.h"

#include <cccl/cub/device/device_reduce.cuh>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "../CudaHelper.h"

namespace rtnlos
{
	FastReconstructionAlgorithm::FastReconstructionAlgorithm()
		: _info()
		  , _useDDA(true)
		  , _precalculated(false)
		  , _currentCount(0)
		  , _bandpassInterval(0.0f, 1.0f)
		  , _numFrequencies(0), _numDepths(0), _imageHeight(0), _imageWidth(0), _sliceSize(0), _frequencyCubeSize(0)
	      , _apertureFullSize{ 0.0f, 0.0f }, _apertureDst{ 0.0f, 0.0f }, _samplingSpace(0.0f)
		  , _imageData(nullptr), _imageResult(nullptr), _ddaWeights(nullptr)
		  , _maxValue(nullptr), _minValue(nullptr), _tempStorage(nullptr), _tempStorageBytes(0)
	{
	}

	FastReconstructionAlgorithm::~FastReconstructionAlgorithm()
	{
		FastReconstructionAlgorithm::destroyResources();
	}

	void FastReconstructionAlgorithm::initialize(const rtnlos::DatasetInfo& info)
	{
		_info = info;

		// For now, aperture_dst is going to be the same as ApertureFullSize
		_apertureDst[0] = info._apertureDstWidth;
		_apertureDst[1] = info._apertureDstHeight;
		_numDepths = static_cast<int>(std::round((info._maxDistance - info._minDistance) / info._deltaDistance)) + 1;
	}

	void FastReconstructionAlgorithm::destroyResources()
	{
		CudaHelper::free(_imageData);
		CudaHelper::free(_imageResult);
		CudaHelper::free(_ddaWeights);
		CudaHelper::free(_maxValue);
		CudaHelper::free(_minValue);
		CudaHelper::free(_tempStorage);

		destroyStreams();
	}

	void FastReconstructionAlgorithm::precalculate()
	{
		// Now allocate all the storage that the ReconstructImage() function will need
		CudaHelper::initializeBuffer(_imageResult, sliceNumPixels());
		CudaHelper::initializeBuffer(_imageData, sliceNumPixels() * _numFrequencies);
		precalculateDDAWeights();

		// Allocate intermediate storage used for ffts during reconstruction
		CudaHelper::initializeBuffer(_maxValue, 1);
		CudaHelper::initializeBuffer(_minValue, 1);

		// Temporary storage for max/min
		cub::DeviceReduce::Max(_tempStorage, _tempStorageBytes, _imageResult, _maxValue, _sliceSize);
		CudaHelper::initializeBuffer(_tempStorage, _tempStorageBytes);

		// Prepare streams for parallel processing
		allocateStreams(glm::max(_numFrequencies, _numDepths));

		// Flag that we've done it.
		_precalculated = true;
		_currentCount = -1; // We haven't reconstructed any yet
	}

	void FastReconstructionAlgorithm::setFFTData(const cufftComplex* data)
	{
		assert(_numFrequencies);
		assert(_imgWidth != 0 && _imgHeight != 0);

		cudaMemcpyAsync(_imageData, data, static_cast<size_t>(_numFrequencies) * sliceNumPixels() * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	}

	void FastReconstructionAlgorithm::enableDepthDependentAveraging(bool useDDA)
	{
		_useDDA = useDDA;
	}

	void FastReconstructionAlgorithm::setNumFrequencies(int n)
	{
		assert(!_numFrequencies);
		assert(!_precalculated);
		_numFrequencies = n;
	}

	void FastReconstructionAlgorithm::setWeights(const float* weights)
	{
		assert(_numFrequencies);
		assert(!_precalculated);
		_weights.clear();
		_weights.insert(_weights.end(), weights, weights + _numFrequencies);
	}

	void FastReconstructionAlgorithm::setLambdas(const float* lambdas)
	{
		assert(_numFrequencies);
		assert(!_precalculated);
		_lambdas.clear();
		_lambdas.insert(_lambdas.end(), lambdas, lambdas + _numFrequencies);
	}

	void FastReconstructionAlgorithm::setOmegas(const float* omegas)
	{
		assert(_numFrequencies);
		assert(!_precalculated);
		_omegas.clear();
		_omegas.insert(_omegas.end(), omegas, omegas + _numFrequencies);
	}

	void FastReconstructionAlgorithm::setSamplingSpace(const float sampling_space)
	{
		assert(!_precalculated);
		_samplingSpace = sampling_space;
	}

	void FastReconstructionAlgorithm::setApertureFullSize(const float* apt)
	{
		assert(!_precalculated);
		memcpy(_apertureFullSize, apt, sizeof(float) * 2);
	}

	void FastReconstructionAlgorithm::setImageDimensions(int width, int height)
	{
		assert(_numFrequencies);
		assert(_apertureFullSize[0] > 0.0 && _apertureFullSize[1] > 0.0);
		assert(!_precalculated);

		// Calculate virtual aperature
		// Calculate the sampling spacing, need for re-calculated virtual aperture size
		float dx = _apertureFullSize[0] / static_cast<float>(height); // Aperture sampling spacing

		_imageWidth = width;
		_imageHeight = height;

		_sliceSize = _imageHeight * _imageWidth;
		_frequencyCubeSize = _numFrequencies * _sliceSize;

		// Update the virtual aperture size
		_apertureDst[0] = static_cast<float>(_imageHeight) * dx;
		_apertureDst[1] = static_cast<float>(_imageWidth) * dx;
	}

	void FastReconstructionAlgorithm::setBandpassInterval(float min, float max)
	{
		_bandpassInterval = glm::vec2(min, max);
	}

	void FastReconstructionAlgorithm::dumpInfo() const
	{
		// Print values for testing
		spdlog::info(
			"Name: {}\n"
			"Number of components: {}\n"
			"Weight size: {}\n"
			"Lambda size: {}\n"
			"Omega size: {}\n"
			"Aperture full size: [{}, {}]\n"
			"Sampling spacing: {:.4f}\n"
			"Image dimensions: [{}, {}]",
			_info._name, _numFrequencies, _weights.size(), _lambdas.size(), _omegas.size(),
			_apertureFullSize[0], _apertureFullSize[1], _samplingSpace,
			_imageWidth, _imageHeight
		);

	}

	void FastReconstructionAlgorithm::writeImageResult(const std::string& filename) const
	{
		cv::Mat image(cv::Size(_imageWidth, _imageHeight), CV_32FC1);

		CudaHelper::checkError(cudaMemcpy(image.data, _imageResult, sliceNumPixels() * sizeof(float), cudaMemcpyDeviceToHost));

		image.convertTo(image, CV_8UC1, 255.0, 0);
		cv::imwrite("logs/" + std::to_string(_currentCount) + ".png", image);
	}

	void FastReconstructionAlgorithm::allocateStreams(glm::uint numStreams)
	{
		_cudaStreams.resize(numStreams);
		for (glm::uint j = 0; j < numStreams; j++)
		{
			cudaStream_t stream;
			CudaHelper::checkError(cudaStreamCreate(&stream));
			_cudaStreams[j] = stream;
		}
	}

	void FastReconstructionAlgorithm::destroyStreams()
	{
		for (auto& stream : _cudaStreams)
			CudaHelper::checkError(cudaStreamDestroy(stream));
		_cudaStreams.clear();
	}

	void FastReconstructionAlgorithm::synchronizeStreams(glm::uint numStreams) const
	{
		for (glm::uint j = 0; j < numStreams; j++)
		{
			CudaHelper::checkError(cudaStreamSynchronize(_cudaStreams[j]));
		}
	}

	void FastReconstructionAlgorithm::precalculateDDAWeights()
	{
		std::vector<float> weights(_numDepths * 3);

		// TODO: remove assumption that min_depth = 1, and max_depth = 3
		float minDepth = 1.0f;
		float maxDepth = 3.0f;
		float depthRange = maxDepth - minDepth;

#pragma omp parallel for
		for (glm::uint i = 0; i < _numDepths; ++i)
		{
			float cen = (_numDepths - 1) / (depthRange * i + _numDepths - 1);
			float lr = (1.f - cen) / 2.f; // The 3 values are a partition of unity

			weights[_numDepths + i] = cen;
			weights[i] = weights[_numDepths * 2 + i] = lr;
		}

		CudaHelper::initializeBuffer(_ddaWeights, _numDepths * 3, weights.data());
	}
}
