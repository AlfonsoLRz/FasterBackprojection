#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cassert>
#include <iomanip>
#include "rsd_reconstructor.h"
#include "rsd_cuda_kernels.h"

#include <cufft.h>
#include <opencv4/opencv2/imgproc.hpp>

using namespace std;

extern float* g_histo;

void throw_if(bool test, const char* msg)
{
	if (test) throw runtime_error(msg);
}

void cuda_throw_if(cudaError_t res, const char* msg)
{
	if (res != cudaSuccess) {
		stringstream s;
		std::string serr = cudaGetErrorString(res);
		s << msg << " (cuda error=(" << res << ") " << serr <<  ")";
		throw runtime_error(s.str());
	}
}

void cufft_throw_if(cufftResult res, const char* msg)
{
	if (res != CUFFT_SUCCESS) {
		stringstream s;
		s << msg << " (cufft error=" << res << ")";
		throw runtime_error(s.str());
	}
}

RSDReconstructor::RSDReconstructor()
	: m_num_components(0)
	, m_aperature_fullsize{ 0.0f, 0.0f }
	, m_aperature_dst{ 0.0f, 0.0f }
	, m_is_simulated(false)
	, m_sampling_space(0.0f)
	, m_spad_index(0)
	, m_img_width(0)
	, m_img_height(0)
	, m_raw_img_width(0)
	, m_raw_img_height(0)
	, m_diff_tb(0)
	, m_diff_lr(0)
	, m_precalculated(false)
	, m_num_depths(0)
	, m_fft_plan(0)
	, m_downsampling_ratio(1.0)
	, m_allocated_bytes(0)
	, m_precalculate_time(0.0)
	, m_use_dda(true)
{
}

RSDReconstructor::~RSDReconstructor()
{
	if (m_fft_plan)
		cufft_throw_if(cufftDestroy(m_fft_plan), "Error destroying cufft plan");
}

void RSDReconstructor::Initialize(const DatasetInfo info)
{
	m_info = info;
	// for now, aperature_dst is going to be the same as ApertureFullSize
	m_aperature_dst[0] = info.apt_dst_width;
	m_aperature_dst[1] = info.apt_dst_height;
	m_num_depths = (int)std::round((info.d_max - info.d_min) / info.d_d) + 1;
}

void RSDReconstructor::EnableDepthDependentAveraging(bool useDDA)
{
	m_use_dda = useDDA;
}

void RSDReconstructor::SetNumComponents(const int num)
{
	throw_if(m_num_components, "SetNumComponents() has already been called");
	throw_if(m_precalculated, "PrecalculateRSD() has already been called");
	m_num_components = num;
}

void RSDReconstructor::SetWeights(const float* weights)
{
	throw_if(!m_num_components, "call SetNumComponents() first");
	throw_if(m_precalculated, "PrecalculateRSD() has already been called");
	m_weights.clear();
	m_weights.insert(m_weights.end(), weights, weights + m_num_components);
}

void RSDReconstructor::SetLambdas(const float* lambdas)
{
	throw_if(!m_num_components, "call SetNumComponents() first");
	throw_if(m_precalculated, "PrecalculateRSD() has already been called");
	m_lambdas.clear();
	m_lambdas.insert(m_lambdas.end(), lambdas, lambdas + m_num_components);
}

void RSDReconstructor::SetOmegas(const float* omegas)
{
	throw_if(!m_num_components, "call SetNumComponents() first");
	throw_if(m_precalculated, "PrecalculateRSD() has already been called");
	m_omegas.clear();
	m_omegas.insert(m_omegas.end(), omegas, omegas + m_num_components);
}

void RSDReconstructor::SetSamplingSpace(const float sampling_space)
{
	throw_if(m_precalculated, "PrecalculateRSD() has already been called");
	m_sampling_space = sampling_space;
}

void RSDReconstructor::SetSPADIndex(const int spad_index)
{
	throw_if(m_precalculated, "PrecalculateRSD() has already been called");
	m_spad_index = spad_index;
}

void RSDReconstructor::SetAperatureFullsize(const float* apt)
{
	throw_if(m_precalculated, "PrecalculateRSD() has already been called");
	memcpy(m_aperature_fullsize, apt, sizeof(float) * 2);
}

void RSDReconstructor::SetImageDimensions(int raw_width, int raw_height, int resampled_width, int resampled_height)
{
	throw_if(!m_num_components, "call SetNumComponents() first");
	throw_if(m_aperature_fullsize[0] == 0.0 || m_aperature_fullsize[1] == 0.0, "call SetAperatureFullsize() first");
	throw_if(m_precalculated, "PrecalculateRSD() has already been called");
	m_raw_img_width = raw_width;
	m_raw_img_height = raw_height;

	// calculate virtual aperature
	// Calculate the sampling spacing, need for re-calcualted virtual aperture size
	float dx = (m_aperature_fullsize[0] / resampled_height); // aperutre sampling spacing
	// Calcualte the difference between aperture
	float diff[2] { m_info.apt_dst_width - m_aperature_fullsize[0], m_info.apt_dst_height - m_aperature_fullsize[1] };

	// Protection if virtual wavefront is smaller
	//throw_if(diff[0] < 0 || diff[1] < 0, "virtual aperature has to be larget");

	// Zero padding
	diff[0] = diff[0] / dx; // transfer into pixel block
	diff[1] = diff[1] / dx; // transfer into pixel block

	m_diff_tb = (int)round(diff[0] / 2);
	m_diff_lr = (int)round(diff[1] / 2);

	//m_img_width = resampled_width + 2 * m_diff_lr;
	//m_img_height = resampled_height + 2 * m_diff_tb;
	m_img_width = m_raw_img_width;
	m_img_height = m_raw_img_height;

	// Update the virtual aperture size
	m_aperature_dst[0] = m_img_height * dx;
	m_aperature_dst[1] = m_img_width * dx;

	// allocate storage on gpu for images data
	m_image_data = CudaAllocImg(1, m_num_components);
}

void RSDReconstructor::SetFFTData(const float* data, const int width, const int height)
{
	throw_if(!m_num_components, "call SetNumComponents() first");
	throw_if(m_img_width == 0 || m_img_height == 0, "call SetImageDimensions() first");

	cudaMemcpy(m_image_data.get(), data, m_num_components * 2 * SliceNumPixels() * sizeof(float), cudaMemcpyHostToDevice);
}

void RSDReconstructor::AddImage(int idx, cv::Mat& image_re, cv::Mat& image_im)
{
	throw_if(!m_num_components, "call SetNumComponents() first");
	throw_if(image_re.size() != image_im.size(), "re and im image sizes don't match");

	cv::Mat u(image_re.size(), CV_32FC2);
	cv::Mat tmp_channel[2] = { image_re, image_im };
	merge(tmp_channel, 2, u);

	cv::resize(u, u, cv::Size(), m_downsampling_ratio, m_downsampling_ratio);

	if (m_raw_img_width == 0) {
		SetImageDimensions(image_re.size().width, image_re.size().height, u.size().width, u.size().height);
	}

	cv::copyMakeBorder(u, u, m_diff_tb, m_diff_tb, m_diff_lr, m_diff_lr, 0);

	throw_if(u.size().width != m_img_width, "image width doesn't match");
	throw_if(u.size().height != m_img_height, "image height doesn't match");

	cudaMemcpy(m_image_data.get() + idx * 2 * SliceNumPixels(), (float*)u.data, 2 * SliceNumPixels() * sizeof(float), cudaMemcpyHostToDevice);
}



void RSDReconstructor::DumpInfo() const
{
	// Print values for testing
	cout << "name:                 " << m_info.name << endl;
	cout << "number of components: " << m_num_components << endl;
	cout << "weight:               " << m_weights.size() << endl;
	cout << "lambda_loop:          " << m_lambdas.size() << endl;
	cout << "omega_space:          " << m_omegas.size() << endl;
	cout << "aperturefull size:    [" << m_aperature_fullsize[0] << ", " << m_aperature_fullsize[1] << "]" << endl;
	cout << "sampling spacing:     " << m_sampling_space << endl;
	cout << "SPAD index:           " << m_spad_index << endl;
	cout << "is simu:              " << m_is_simulated << endl;
}

void RSDReconstructor::DumpPrecalcStats() const
{
	// Print values for testing
	size_t sz = 0;
	if (m_fft_plan != 0)
		cufft_throw_if(cufftGetSize(m_fft_plan, &sz), "Unable to get fft plan work size");

	cout << "GPU Buffer Bytes Allocated: " << setw(14) << m_allocated_bytes << " bytes" << endl;
	cout << "GPU FFT Work Space Bytes:   " << setw(14) << sz << " bytes" << endl;
	cout << "Precalcuating RSD Took:     " << setw(14) << fixed << setprecision(3) << m_precalculate_time << " ms" << endl;
}

// allocates all necessary GPU data for RSD Array and for Reconstruction, and
// precalculates RSD. Basically sets everything up.
void RSDReconstructor::PrecalculateRSD()
{
	// check that all necessary values have been set
	throw_if(!m_num_components, "call SetNumComponents() first");
	throw_if(m_weights.size() != m_num_components, "call SetWeights() first");
	throw_if(m_lambdas.size() != m_num_components, "call SetLambdas() first");
	throw_if(m_omegas.size() != m_num_components, "call SetOmegas() first");
	throw_if(m_aperature_fullsize[0] == 0.0 || m_aperature_fullsize[1] == 0.0, "call SetAperature() first");
	throw_if(m_img_width == 0 || m_img_height == 0, "call SetImageDimensions() first");
	throw_if(m_precalculated, "PrecalculateRSD() has already been called");

	const float c_speed = 299792458.0f;
	
	// useful values to have around
	m_wave_size = m_img_height * m_img_width * 2;
	m_depth_size = m_num_components * m_wave_size;

	int i = 0;
	m_rsd = CudaAllocImg(m_num_depths, m_num_components);

	cufftHandle fft_plan;
	cufft_throw_if(cufftPlan2d(&fft_plan, m_img_height, m_img_width, CUFFT_C2C), "Error creating cufft plan");

	int depth_idx;
	float depth;
	std::cout << " m_info.d_min: " << m_info.d_min << std::endl;
	std::cout << " m_info.d_max: " << m_info.d_max << std::endl;
	std::cout << " m_info.d_d: " << m_info.d_d << std::endl;
	for (depth = m_info.d_min, depth_idx = 0; depth < m_info.d_max; depth += m_info.d_d, depth_idx++) {
		float t_tmp = (depth + m_info.d_offset) / c_speed;

		std::cout << "  i=" << i << " depth_idx=" << depth_idx << " depth=" << depth << std::endl;
		for (int wave_num = 0; wave_num < m_num_components; wave_num++) {
			float lambda = m_lambdas[wave_num];
			float omega = m_omegas[wave_num];

			float* d_ker = ImgAt(m_rsd, depth_idx, wave_num);
			RSDKernelConvolution(lambda, depth, omega, t_tmp, fft_plan, d_ker);
		}
		i++;
	}
	assert(i == m_num_depths);

	cufft_throw_if(cufftDestroy(fft_plan), "Error destroying cufft plan");

	// now allocate all the storage that the ReconstructImage() function will need
	m_img_2d = CudaAllocImg(1, 1, false);

	// allocate intermediate storage used for ffts during reconstruction
	m_u_total_ffts = CudaAllocImg(1, m_num_components);
	m_u_out = CudaAllocImg(1, m_num_components);
	m_u_sum = CudaAllocImg(1, 1);

	// allocate cufft plan for use during reconstruction
	cufftPlan2d(&m_fft_plan, m_img_height, m_img_width, CUFFT_C2C);

	EnableCubeGeneration(true);
	PrecalculateDDAWeights();

	// record how long this took, and flag that we've done it.
	m_precalculated = true;
	m_cur_cnt = -1; // we haven't reconstructed any yet.	
}

void RSDReconstructor::PrecalculateDDAWeights()
{
	std::vector<float> w(m_num_depths * 3);
	// todo: remove assumption that min_depth = 1, and max_depth = 3
	float min_depth = 1.0f;
	float max_depth = 3.0f;

	float depth_range = max_depth - min_depth;

	for (int i = 0; i < m_num_depths; i++) {
		float cen = (m_num_depths - 1) / (depth_range * i + m_num_depths - 1);
		float lr = (1.f - cen) / 2.f; // the 3 values are a partition of unity

		w[m_num_depths + i] = cen;
		w[i] = w[m_num_depths * 2 + i] = lr;
	}
	m_dda_weights = CudaAllocFloatArr(m_num_depths * 3);
	cuda_throw_if(cudaMemcpy(m_dda_weights.get(), w.data(), m_num_depths * 3 * sizeof(float), cudaMemcpyHostToDevice), "Error copying dda weights to gpu");
}

void RSDReconstructor::EnableCubeGeneration(bool enable)
{
	m_cube_images = CudaAllocImg(m_num_depths, 4, false); // 4 cubes, so we can store 3 reconstructions for dda, plus a temporary one for computation
}

// call after each time full set of images has been added
void RSDReconstructor::ReconstructImage(cv::Mat& img_out)
{
	throw_if(!m_precalculated, "Call PrecalculateRSD() first.");

	m_cur_cnt++;

	// compute the ffts of all the images
	for (int wave_num = 0; wave_num < m_num_components; wave_num++) {
		cufftResult res = cufftExecC2C(m_fft_plan,
			reinterpret_cast<cufftComplex*>(ImgAt(m_image_data, wave_num)),
			reinterpret_cast<cufftComplex*>(ImgAt(m_u_total_ffts, wave_num)),
			CUFFT_FORWARD);
	}

	int i;
	float depth;
	for (depth = m_info.d_min, i = 0; depth < m_info.d_max; depth += m_info.d_d, i++)
	{
		// Temporal output wavefront at the depth plane
		//auto ptr_zero_check = DbgInspect(m_u_sum.get(), 40);
		cudaMemset(m_u_sum.get(), 0x00, m_wave_size * sizeof(float));
		//ptr_zero_check = DbgInspect(m_u_sum.get(), 40);

		// all convolutions for this depth
		int num_threads = 512;
		int num_blocks = (m_img_width * m_img_height + num_threads - 1) / num_threads;
		MulSpectrumMany<<<dim3(num_blocks, m_num_components), num_threads>>> (
			m_u_total_ffts.get(),
			ImgAt(m_rsd, i, 0),
			m_u_out.get(),
			SliceNumPixels());

		// summing
		// for-loop summing ws faster than parallel reduction kernel (~0.655ms vs ~1.0-1.3ms)
		for (int wave_num = 0; wave_num < m_num_components; wave_num++) {
			AddScale << <m_img_height, m_img_width >> > (
				m_u_sum.get(),
				ImgAt(m_u_out, wave_num),
				m_weights[wave_num]);
		}
		// optional parallel reduction version instead (slower...)
		//AddScaleMany<<<dim3(img_width, img_height), 128>>>(u_sum(0,0), u_out(0,0), d_weights.get(), u_num);

		// idft after integration
		cufftResult res = cufftExecC2C(m_fft_plan, reinterpret_cast<cufftComplex*>(m_u_sum.get()),
			reinterpret_cast<cufftComplex*>(m_u_sum.get()),
			CUFFT_INVERSE);

		// store this slice
		int idx = m_cur_cnt % 3;
		Abs<<<m_img_height, m_img_width>>>(m_u_sum.get(), 
			m_cube_images.get()  +  // base addr of 3 cubes
				idx * CubeNumPixels() + // offset to the proper cube
				i * SliceNumPixels()); // offset to the proper slice (depth)
	}

	cv::Mat img_2d(cv::Size(m_img_width, m_img_height), CV_32FC1);
	if (m_cur_cnt > 0) {
		// we have the next frame, so we can process the previous frame.
		int idx_prev = (m_cur_cnt - 1) % 3;
		if (m_use_dda) {
			// perform depth dependent averaging across the 3 stored frames
			DDA<<<dim3(m_img_height, m_num_depths, 1), m_img_width>>> (m_cube_images.get(), idx_prev, m_dda_weights.get());
			MaxZ<<<m_img_height, m_img_width >> > (CubeAt(m_cube_images, 3), m_num_depths, m_img_2d.get());
		}
		else {
			// depth dependent averaging is turned off, just pick the max from the current (previous) single frame.
			MaxZ<<<m_img_height, m_img_width >> > (CubeAt(m_cube_images, idx_prev), m_num_depths, m_img_2d.get());
		}

		// copy back to host
		cuda_throw_if(cudaMemcpy(img_2d.data, m_img_2d.get(), SliceNumPixels() * sizeof(float), cudaMemcpyDeviceToHost), "Error copying final image from device to host");
	}
	
	// shift picture back to center, then flip it
	FFTShift(img_2d);
	cv::flip(img_2d, img_2d, 1);

	// Up sampling
	float up = 1.0f / m_downsampling_ratio;
	cv::resize(img_2d, img_out, cv::Size(), up, up); // wavefront spatial upsampling for larger image
}

void RSDReconstructor::RSDKernelConvolution(const float lambda, const float depth, const float omega, const float t, cufftHandle fft_plan, float* d_ker)
{
	// get physical aprture size
	float apt_x;
	int dim_x, dim_y; // Matrix dimension

	apt_x = m_aperature_dst[0]; // physical aperture x, unit meter
	//apt_y = apt.at<float>(0, 1); // physical aperture y, unit meter

	dim_x = m_img_height; // input wavefront matrix size
	dim_y = m_img_width;  // input wavefront matrix size 

	// spatial sampling density
	float dx = apt_x / (dim_x - 1);
	//float dy = apt_y / (dim_y - 1);

	// if(dx != dy){cout << "check dimension" << endl; exit(0);};

	// Apply RSD convolution kernel equation
	float z_hat = depth / dx;
	float mul_square = lambda * z_hat / (dim_x * dx);

	RSD_Kernel<<<dim_y, dim_x>>>(d_ker, z_hat * z_hat, mul_square);

	// perform fft on gpu
	cufftResult res = cufftExecC2C(fft_plan, reinterpret_cast<cufftComplex*>(d_ker),
		reinterpret_cast<cufftComplex*>(d_ker),
		CUFFT_FORWARD);

	// convolve the two by pointwise multiplication of the spectrums
	MulSpectrumExpHarmonic<<<dim_y, dim_x>>> (d_ker, omega, t);
}

shared_ptr<float> RSDReconstructor::CudaAllocFloatArr(int n)
{
	float* dd;
	size_t sz = n * sizeof(float);
	cuda_throw_if(cudaMalloc(&dd, sz), "cudaMalloc failed");
	shared_ptr<float> ret;
	ret.reset(dd, cudaFree);
	m_allocated_bytes += sz;
	return ret;
}

shared_ptr<float> RSDReconstructor::CudaAllocImg(int depths, int waves, bool complex)
{
	int n = m_img_height * m_img_width * depths * waves * (complex ? 2 : 1);
	return CudaAllocFloatArr(n);
}

float* RSDReconstructor::CubeAt(shared_ptr<float> p, int cube_num)
{
	assert(p);
	return &(p.get()[cube_num * m_img_height * m_img_width * m_num_depths]);
}

float* RSDReconstructor::ImgAt(shared_ptr<float> p, int wave_num)
{
	assert(p);
	return &(p.get()[wave_num * m_wave_size]);
}

float* RSDReconstructor::ImgAt(shared_ptr<float> p, int depth, int wave_num)
{
	assert(p);
	return &(p.get()[depth * m_depth_size + wave_num * m_wave_size]);
}

std::unique_ptr<std::vector<float>> RSDReconstructor::GetImageData()
{
	return DbgInspect(m_image_data.get(), m_img_width * m_img_height * 2 * m_num_components);
}

std::unique_ptr<std::vector<float>> RSDReconstructor::GetCubeData()
{
	int cur_idx = m_cur_cnt % 3;
	auto ret = DbgInspect(m_cube_images.get() + cur_idx * m_img_width * m_img_height * m_num_depths,
		m_img_width * m_img_height * m_num_depths);
	
	// there's a faster way to do this, but this is only for debug logging...
	int sz[] = { m_img_height, m_img_width };
	//#pragma omp parallel
	for (int i = 0; i < m_num_depths; i++) {
		float* p = ret->data() + i * m_img_height * m_img_width;
		cv::Mat sl(2, sz, CV_32FC1, p);
		FFTShift(sl);
		cv::flip(sl, sl, 1);
	}
	return ret;
}

unique_ptr<std::vector<float>> RSDReconstructor::DbgInspect(float* devptr, int sz)
{
	unique_ptr<std::vector<float>> ret;
	ret.reset(new std::vector<float>(sz));
	cuda_throw_if(cudaMemcpy(ret->data(), devptr, sz * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
	return ret;
}


/**
	mimic fftshift

	@param:
		Mat &out: input Mat (pointer)
	@return:
		Mat &out
	example:
		fftshift(A)
*/
void RSDReconstructor::FFTShift(cv::Mat& out)
{
	cv::Size sz = out.size();
	cv::Point pt(0, 0);
	pt.x = (int)floor(sz.width / 2.0);
	pt.y = (int)floor(sz.height / 2.0);
	CircShift(out, pt);
}

/**
	mimic circshift

	@param:

	@return:

	example:

*/
void RSDReconstructor::CircShift(cv::Mat& out, const cv::Point& delta)
{
	cv::Size sz = out.size();

	assert(sz.height > 0 && sz.width > 0);

	if ((sz.height == 1 && sz.width == 1) || (delta.x == 0 && delta.y == 0))
		return;

	int x = delta.x;
	int y = delta.y;
	if (x > 0) x = x % sz.width;
	if (y > 0) y = y % sz.height;
	if (x < 0) x = x % sz.width + sz.width;
	if (y < 0) y = y % sz.height + sz.height;


	vector<cv::Mat> planes;
	split(out, planes);

	for (size_t i = 0; i < planes.size(); i++)
	{

		cv::Mat tmp0, tmp1, tmp2, tmp3;
		cv::Mat q0(planes[i], cv::Rect(0, 0, sz.width, sz.height - y));
		cv::Mat q1(planes[i], cv::Rect(0, sz.height - y, sz.width, y));
		q0.copyTo(tmp0);
		q1.copyTo(tmp1);
		tmp0.copyTo(planes[i](cv::Rect(0, y, sz.width, sz.height - y)));
		tmp1.copyTo(planes[i](cv::Rect(0, 0, sz.width, y)));

		cv::Mat q2(planes[i], cv::Rect(0, 0, sz.width - x, sz.height));
		cv::Mat q3(planes[i], cv::Rect(sz.width - x, 0, x, sz.height));
		q2.copyTo(tmp2);
		q3.copyTo(tmp3);
		tmp2.copyTo(planes[i](cv::Rect(x, 0, sz.width - x, sz.height)));
		tmp3.copyTo(planes[i](cv::Rect(0, 0, x, sz.height)));
	}

	merge(planes, out);
}

float CompareFloatArray(float* p1, float* p2, size_t sz)
{
	if (sz == 0)
		return 0.f;
	float maxdiff = std::abs(p1[0] - p2[0]);
	for (int i = 1; i < sz; i++) {
		float diff = std::abs(p1[i] - p2[i]);
		if (diff > 1E-2)
			std::cout << i << ": " << p1[i] << p2[i] << diff << std::endl;
		maxdiff = std::max(diff, maxdiff);
	}
	return maxdiff;
}