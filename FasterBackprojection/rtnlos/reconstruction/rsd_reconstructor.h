#pragma once

#include <vector>
#include <memory>
#include <opencv4/opencv2/core.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv4/opencv2/core/mat.hpp>

using namespace std;

// forward declaration of helper functions
void throw_if(bool test, const char* msg);
void cuda_throw_if(cudaError_t res, const char* msg);

// contains info about a data set
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
public:
	RSDReconstructor();
	~RSDReconstructor();

	void Initialize(const DatasetInfo info);

	// call all of these before calling PrecalculateRSD()
	void EnableDepthDependentAveraging(bool useDDA);
	void SetNumComponents(const int n);
	void SetWeights(const float* weights);
	void SetLambdas(const float* lambdas);
	void SetOmegas(const float* omegas);
	void SetSamplingSpace(const float sampling_space);
	void SetSPADIndex(const int spad_index);
	void SetAperatureFullsize(const float* apt);
	void SetIsSimulated(const bool sim) { m_is_simulated = sim; }

	// call this once before PrecalculateRSD to set the histogram input size
	void SetImageDimensions(int raw_width, int raw_height, int resampled_width, int resampled_height);

	// call this once after calling the above methods
	void PrecalculateRSD(); // allocates all necessary GPU data, precalculates RSD. Basically sets everything up.

	// enables/disables calculation of the image cube
	void EnableCubeGeneration(bool enable);

	// add the image ffts directly (bypassing the AddImage calls)
	void SetFFTData(const float* data, const int width, const int height);
	
	// call this any time after SetImageDimensions has been called (call once for each wavefront)
	void AddImage(int idx, cv::Mat& image_re, cv::Mat& image_im);
	
	// call after each time full set of images has been added
	void ReconstructImage(cv::Mat& img_out);

	// query
	void DumpInfo() const;
	void DumpPrecalcStats() const;
	int ImageWidth() const { return m_raw_img_width; }
	int ImageHeight() const { return m_raw_img_height; }

	static std::unique_ptr<std::vector<float>> DbgInspect(float* devptr, int sz);
	std::unique_ptr<std::vector<float>> GetImageData();
	std::unique_ptr<std::vector<float>> GetCubeData();

private:
	int SliceNumPixels() const { return m_img_height * m_img_width; }
	int CubeNumPixels() const { return SliceNumPixels() * m_num_depths; }
	void RSDKernelConvolution(const float lambda, const float depth, const float omega, const float t, cufftHandle fft_plan, float* d_ker);

	shared_ptr<float> CudaAllocFloatArr(int sz);
	shared_ptr<float> CudaAllocImg(int depths, int waves, bool complex = true);
	float* CubeAt(shared_ptr<float> p, int cube_num); 
	float* ImgAt(shared_ptr<float> p, int wave_num); // assumes complex
	float* ImgAt(shared_ptr<float> p, int depth, int wave_num); // assumes complex
	void FFTShift(cv::Mat& out);
	void CircShift(cv::Mat& out, const cv::Point& delta);
	void PrecalculateDDAWeights();

	DatasetInfo m_info;
	bool m_use_dda; // use depth dependent averaging
	int m_num_components;
	float m_aperature_fullsize[2];
	float m_aperature_dst[2];
	float m_sampling_space;
	int m_spad_index;
	bool m_is_simulated;
	int m_num_depths;
	int m_raw_img_height;
	int m_raw_img_width;
	int m_img_height;
	int m_img_width;
	int m_diff_tb;
	int m_diff_lr;
	vector<float> m_weights;
	vector<float> m_lambdas;
	vector<float> m_omegas;
	int m_wave_size;
	int m_depth_size;
	bool m_precalculated;
	float m_downsampling_ratio;
	int m_cur_cnt;

	// device pointers
	size_t m_allocated_bytes;
	double m_precalculate_time;
	shared_ptr<float> m_image_data;
	shared_ptr<float> m_img_2d;
	shared_ptr<float> m_rsd;
	shared_ptr<float> m_u_total_ffts;
	shared_ptr<float> m_u_out;
	shared_ptr<float> m_u_sum;
	shared_ptr<float> m_cube_images;
	shared_ptr<float> m_dda_weights;
	cufftHandle m_fft_plan;
};