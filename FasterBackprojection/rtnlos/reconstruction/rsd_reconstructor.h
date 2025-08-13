#pragma once

#include <vector>
#include <memory>
#include <opencv4/opencv2/core.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv4/opencv2/core/mat.hpp>

using namespace std;

// Contains info about a data set
struct DatasetInfo
{
	string name;
	float apt_dst_width;
	float apt_dst_height;
	float d_min;
	float d_max;
	float d_d;
	float d_offset;
};

class RSDReconstructor
{
	DatasetInfo					_info;
	bool						_useDDA;			// Depth dependent averaging

	glm::uint					_numFrequencies, _numDepths, _imgHeight, _imgWidth;

	float						_apertureFullSize[2];
	float						_apertureDst[2];
	float						_samplingSpace;
	int							_diffTopLeft, _diffLowerRight;
	vector<float>				_weights, _lambdas, _omegas;
	glm::uint					_sliceSize, _frequencyCubeSize;
	bool						_precalculated;
	glm::uint					_currentCount;

	// Device pointers
	cufftComplex* 				_imgData;
	float*						_img2D;
	cufftComplex* 				_rsd;
	cufftComplex*				_uTotalFFTs;
	cufftComplex*				_uOut;
	cufftComplex*				_uSum;
	float*						_cubeImages;
	float*						_ddaWeights;
	float*						_dWeights;
	float*						_maxValue, * _minValue;

	void*						_tempStorage;		
	size_t						_tempStorageBytes;

	cufftHandle					_fftPlan2D;
	std::vector<cudaStream_t>	_cudaStreams;

	glm::uint					_blockSize1D, _gridSize1D;
	dim3						_blockSize2D_freq, _blockSize2D_depth, _blockSize2D_pix;
	dim3						_gridSize2D_freq, _gridSize2D_depth, _gridSize2D_pix;

public:
	RSDReconstructor();
	~RSDReconstructor();

	void Initialize(const DatasetInfo& info);

	// Call all of these before calling PrecalculateRSD()
	void EnableDepthDependentAveraging(bool useDDA);
	void SetNumFrequencies(int n);
	void SetWeights(const float* weights);
	void SetLambdas(const float* lambdas);
	void SetOmegas(const float* omegas);
	void SetSamplingSpace(const float sampling_space);
	void SetApertureFullSize(const float* apt);
	void SetImageDimensions(int width, int height);

	// Call this once after calling the above methods
	void PrecalculateRSD(); // Allocates all necessary GPU data, precalculates RSD. Basically sets everything up.

	// Enables/disables calculation of the image cube
	void EnableCubeGeneration(bool enable);

	// Add the image ffts directly (bypassing the AddImage calls)
	void SetFFTData(const cufftComplex* data) const;
	
	// Call after each time full set of images has been added
	void ReconstructImage(cv::Mat& img_out);

	// Query
	void DumpInfo() const;

private:
	glm::uint SliceNumPixels() const { return _sliceSize; }
	glm::uint CubeNumPixels() const { return _sliceSize * _numDepths; }

	void RSDKernelConvolution(cufftComplex* dKernel, cufftHandle fftPlan, const float lambda, const float omega, const float depth, const float t) const;

	float*			CubeAt(shared_ptr<float> p, int cube_num) const; 
	cufftComplex*	ImageAt(cufftComplex* p, int depth) const;		// Assumes complex

	void		FFTShift(cv::Mat& out);
	static void	CircShift(cv::Mat& out, const cv::Point& delta);
	void		PrecalculateDDAWeights();
};