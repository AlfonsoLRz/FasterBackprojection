#pragma once

#include <vector>
#include <memory>
#include <opencv4/opencv2/core.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv4/opencv2/core/mat.hpp>

using namespace std;

// Forward declaration of helper functions
void throw_if(bool test, const char* msg);
void cuda_throw_if(cudaError_t res, const char* msg);

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
	DatasetInfo				_info;
	bool					_useDDA; // use depth dependent averaging
	int						_numComponents;
	float					_apertureFullSize[2];
	float					_appertureDst[2];
	float					_samplingSpace;
	int						_spadIndex;
	bool					_isSimulated;
	int						_numDepths;
	int						_rawImgHeight;
	int						_rawImageWidth;
	int						_imgHeight;
	int						_imgWidth;
	int						_diffTb;
	int						_diffLr;
	vector<float>			_weights;
	vector<float>			_lambdas;
	vector<float>			_omegas;
	int						_waveSize;
	int						_depthSize;
	bool					_precalculated;
	float					_downsamplingRatio;
	int						_currentCount;

	// Device pointers
	size_t					_allocatedBytes;
	double					_precalculateTime;
	shared_ptr<float>		_imgData;
	shared_ptr<float>		_img2D;
	shared_ptr<float>		_rsd;
	shared_ptr<float>		_uTotalFFTs;
	shared_ptr<float>		_uOut;
	shared_ptr<float>		_uSum;
	shared_ptr<float>		_cubeImages;
	shared_ptr<float>		_ddaWeights;
	cufftHandle				_fftPlan;

public:
	RSDReconstructor();
	~RSDReconstructor();

	void Initialize(const DatasetInfo info);

	// Call all of these before calling PrecalculateRSD()
	void EnableDepthDependentAveraging(bool useDDA);
	void SetNumComponents(int n);
	void SetWeights(const float* weights);
	void SetLambdas(const float* lambdas);
	void SetOmegas(const float* omegas);
	void SetSamplingSpace(const float sampling_space);
	void SetSPADIndex(const int spadIndex);
	void SetApertureFullsize(const float* apt);
	void SetIsSimulated(const bool sim) { _isSimulated = sim; }

	// call this once before PrecalculateRSD to set the histogram input size
	void SetImageDimensions(int raw_width, int raw_height, int resampled_width, int resampled_height);

	// Call this once after calling the above methods
	void PrecalculateRSD(); // Allocates all necessary GPU data, precalculates RSD. Basically sets everything up.

	// Enables/disables calculation of the image cube
	void EnableCubeGeneration(bool enable);

	// Add the image ffts directly (bypassing the AddImage calls)
	void SetFFTData(const float* data, const int width, const int height);
	
	// Call this any time after SetImageDimensions has been called (call once for each wavefront)
	void AddImage(int idx, cv::Mat& image_re, cv::Mat& image_im);
	
	// Call after each time full set of images has been added
	void ReconstructImage(cv::Mat& img_out);

	// query
	void DumpInfo() const;
	void DumpPrecalcStats() const;
	int ImageWidth() const { return _rawImageWidth; }
	int ImageHeight() const { return _rawImgHeight; }

	static std::unique_ptr<std::vector<float>>	DbgInspect(float* devptr, int sz);
	std::unique_ptr<std::vector<float>>			GetImageData();
	std::unique_ptr<std::vector<float>>			GetCubeData();

private:
	int		SliceNumPixels() const { return _imgHeight * _imgWidth; }
	int		CubeNumPixels() const { return SliceNumPixels() * _numDepths; }

	void	RSDKernelConvolution(const float lambda, const float depth, const float omega, const float t, cufftHandle fft_plan, float* d_ker);

	shared_ptr<float> CudaAllocFloatArr(int sz);
	shared_ptr<float> CudaAllocImg(int depths, int waves, bool complex = true);

	float*		CubeAt(shared_ptr<float> p, int cube_num); 
	float*		ImgAt(shared_ptr<float> p, int wave_num);				// Assumes complex
	float*		ImgAt(shared_ptr<float> p, int depth, int wave_num);	// Assumes complex

	void		FFTShift(cv::Mat& out);
	void		CircShift(cv::Mat& out, const cv::Point& delta);
	void		PrecalculateDDAWeights();
};