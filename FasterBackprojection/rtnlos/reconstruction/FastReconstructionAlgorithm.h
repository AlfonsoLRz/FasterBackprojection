#pragma once

#include <cufft.h>

#include "data/SceneParameters.h"
#include "../ViewportSurface.h"

namespace rtnlos
{
	class FastReconstructionAlgorithm 
	{
	protected:
		rtnlos::DatasetInfo			_info;
		bool						_useDDA;			// Depth dependent averaging
		bool						_precalculated;
		int							_currentCount;
		glm::vec2					_bandpassInterval;

		// Inferred data
		glm::uint					_numFrequencies, _numDepths, _imageHeight, _imageWidth;
		glm::uint					_sliceSize, _frequencyCubeSize;
		float						_apertureFullSize[2];
		float						_apertureDst[2];
		float						_samplingSpace;
		std::vector<float>			_weights, _lambdas, _omegas;

		// Device pointers and Cuda resources
		cufftComplex*				_spadData;
		float*						_imageResult;
		float*						_ddaWeights;
		float*						_maxValue, * _minValue;

		void*						_tempStorage;
		size_t						_tempStorageBytes;

		std::vector<cudaStream_t>	_cudaStreams;

	public:
		FastReconstructionAlgorithm();
		virtual ~FastReconstructionAlgorithm();

		virtual void initialize(const rtnlos::DatasetInfo& info);
		virtual void destroyResources();

		// Call this before reconstructing images
		// Allocates all necessary GPU data, precalculates RSD. Basically sets everything up
		virtual void precalculate();

		virtual void setFFTData(const cufftComplex* data);

		virtual void reconstructImage(ViewportSurface* viewportSurface) = 0;

		// Call all of these before calling Precalculate()
		void enableDepthDependentAveraging(bool useDDA);
		void setNumFrequencies(int n);
		void setWeights(const float* weights);
		void setLambdas(const float* lambdas);
		void setOmegas(const float* omegas);
		void setSamplingSpace(const float sampling_space);
		void setApertureFullSize(const float* apt);
		void setImageDimensions(int width, int height);
		void setBandpassInterval(float min, float max);

		// Query
		void dumpInfo() const;

		void writeImageResult(const std::string& filename) const;

	protected:
		void allocateStreams(glm::uint numStreams);
		void destroyStreams();
		void synchronizeStreams(glm::uint numStreams) const;

		void precalculateDDAWeights();

		glm::uint sliceNumPixels() const { return _sliceSize; }
		glm::uint cubeNumPixels() const { return _sliceSize * _numDepths; }
    };
}
