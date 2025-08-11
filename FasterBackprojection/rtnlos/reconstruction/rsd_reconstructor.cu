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
	: _numComponents(0)
	, _apertureFullSize{ 0.0f, 0.0f }
	, _appertureDst{ 0.0f, 0.0f }
	, _isSimulated(false)
	, _samplingSpace(0.0f)
	, _spadIndex(0)
	, _imgWidth(0)
	, _imgHeight(0)
	, _rawImageWidth(0)
	, _rawImgHeight(0)
	, _diffTb(0)
	, _diffLr(0)
	, _precalculated(false)
	, _numDepths(0)
	, _fftPlan(0)
	, _downsamplingRatio(1.0)
	, _allocatedBytes(0)
	, _precalculateTime(0.0)
	, _useDDA(true)
{
}

RSDReconstructor::~RSDReconstructor()
{
	if (_fftPlan)
		cufft_throw_if(cufftDestroy(_fftPlan), "Error destroying cufft plan");
}

void RSDReconstructor::Initialize(const DatasetInfo info)
{
	_info = info;
	// for now, aperature_dst is going to be the same as ApertureFullSize
	_appertureDst[0] = info.apt_dst_width;
	_appertureDst[1] = info.apt_dst_height;
	_numDepths = (int)std::round((info.d_max - info.d_min) / info.d_d) + 1;
}

void RSDReconstructor::EnableDepthDependentAveraging(bool useDDA)
{
	_useDDA = useDDA;
}

void RSDReconstructor::SetNumComponents(int n)
{
	throw_if(_numComponents, "SetNumComponents() has already been called");
	throw_if(_precalculated, "PrecalculateRSD() has already been called");
	_numComponents = n;
}

void RSDReconstructor::SetWeights(const float* weights)
{
	throw_if(!_numComponents, "call SetNumComponents() first");
	throw_if(_precalculated, "PrecalculateRSD() has already been called");
	_weights.clear();
	_weights.insert(_weights.end(), weights, weights + _numComponents);
}

void RSDReconstructor::SetLambdas(const float* lambdas)
{
	throw_if(!_numComponents, "call SetNumComponents() first");
	throw_if(_precalculated, "PrecalculateRSD() has already been called");
	_lambdas.clear();
	_lambdas.insert(_lambdas.end(), lambdas, lambdas + _numComponents);
}

void RSDReconstructor::SetOmegas(const float* omegas)
{
	throw_if(!_numComponents, "call SetNumComponents() first");
	throw_if(_precalculated, "PrecalculateRSD() has already been called");
	_omegas.clear();
	_omegas.insert(_omegas.end(), omegas, omegas + _numComponents);
}

void RSDReconstructor::SetSamplingSpace(const float sampling_space)
{
	throw_if(_precalculated, "PrecalculateRSD() has already been called");
	_samplingSpace = sampling_space;
}

void RSDReconstructor::SetSPADIndex(const int spadIndex)
{
	throw_if(_precalculated, "PrecalculateRSD() has already been called");
	_spadIndex = spadIndex;
}

void RSDReconstructor::SetApertureFullsize(const float* apt)
{
	throw_if(_precalculated, "PrecalculateRSD() has already been called");
	memcpy(_apertureFullSize, apt, sizeof(float) * 2);
}

void RSDReconstructor::SetImageDimensions(int raw_width, int raw_height, int resampled_width, int resampled_height)
{
	throw_if(!_numComponents, "call SetNumComponents() first");
	throw_if(_apertureFullSize[0] == 0.0 || _apertureFullSize[1] == 0.0, "call SetApertureFullsize() first");
	throw_if(_precalculated, "PrecalculateRSD() has already been called");
	_rawImageWidth = raw_width;
	_rawImgHeight = raw_height;

	// calculate virtual aperature
	// Calculate the sampling spacing, need for re-calcualted virtual aperture size
	float dx = (_apertureFullSize[0] / resampled_height); // aperutre sampling spacing
	// Calcualte the difference between aperture
	float diff[2] { _info.apt_dst_width - _apertureFullSize[0], _info.apt_dst_height - _apertureFullSize[1] };

	// Protection if virtual wavefront is smaller
	//throw_if(diff[0] < 0 || diff[1] < 0, "virtual aperature has to be larget");

	// Zero padding
	diff[0] = diff[0] / dx; // transfer into pixel block
	diff[1] = diff[1] / dx; // transfer into pixel block

	_diffTb = (int)round(diff[0] / 2);
	_diffLr = (int)round(diff[1] / 2);

	//_imgWidth = resampled_width + 2 * _diffLr;
	//_imgHeight = resampled_height + 2 * _diffTb;
	_imgWidth = _rawImageWidth;
	_imgHeight = _rawImgHeight;

	// Update the virtual aperture size
	_appertureDst[0] = _imgHeight * dx;
	_appertureDst[1] = _imgWidth * dx;

	// allocate storage on gpu for images data
	_imgData = CudaAllocImg(1, _numComponents);
}

void RSDReconstructor::SetFFTData(const float* data, const int width, const int height)
{
	throw_if(!_numComponents, "call SetNumComponents() first");
	throw_if(_imgWidth == 0 || _imgHeight == 0, "call SetImageDimensions() first");

	cudaMemcpy(_imgData.get(), data, _numComponents * 2 * SliceNumPixels() * sizeof(float), cudaMemcpyHostToDevice);
}

void RSDReconstructor::AddImage(int idx, cv::Mat& image_re, cv::Mat& image_im)
{
	throw_if(!_numComponents, "call SetNumComponents() first");
	throw_if(image_re.size() != image_im.size(), "re and im image sizes don't match");

	cv::Mat u(image_re.size(), CV_32FC2);
	cv::Mat tmp_channel[2] = { image_re, image_im };
	merge(tmp_channel, 2, u);

	cv::resize(u, u, cv::Size(), _downsamplingRatio, _downsamplingRatio);

	if (_rawImageWidth == 0) {
		SetImageDimensions(image_re.size().width, image_re.size().height, u.size().width, u.size().height);
	}

	cv::copyMakeBorder(u, u, _diffTb, _diffTb, _diffLr, _diffLr, 0);

	throw_if(u.size().width != _imgWidth, "image width doesn't match");
	throw_if(u.size().height != _imgHeight, "image height doesn't match");

	cudaMemcpy(_imgData.get() + idx * 2 * SliceNumPixels(), (float*)u.data, 2 * SliceNumPixels() * sizeof(float), cudaMemcpyHostToDevice);
}



void RSDReconstructor::DumpInfo() const
{
	// Print values for testing
	cout << "name:                 " << _info.name << endl;
	cout << "number of components: " << _numComponents << endl;
	cout << "weight:               " << _weights.size() << endl;
	cout << "lambda_loop:          " << _lambdas.size() << endl;
	cout << "omega_space:          " << _omegas.size() << endl;
	cout << "aperturefull size:    [" << _apertureFullSize[0] << ", " << _apertureFullSize[1] << "]" << endl;
	cout << "sampling spacing:     " << _samplingSpace << endl;
	cout << "SPAD index:           " << _spadIndex << endl;
	cout << "is simu:              " << _isSimulated << endl;
}

void RSDReconstructor::DumpPrecalcStats() const
{
	// Print values for testing
	size_t sz = 0;
	if (_fftPlan != 0)
		cufft_throw_if(cufftGetSize(_fftPlan, &sz), "Unable to get fft plan work size");

	cout << "GPU Buffer Bytes Allocated: " << setw(14) << _allocatedBytes << " bytes" << endl;
	cout << "GPU FFT Work Space Bytes:   " << setw(14) << sz << " bytes" << endl;
	cout << "Precalcuating RSD Took:     " << setw(14) << fixed << setprecision(3) << _precalculateTime << " ms" << endl;
}

// allocates all necessary GPU data for RSD Array and for Reconstruction, and
// precalculates RSD. Basically sets everything up.
void RSDReconstructor::PrecalculateRSD()
{
	// check that all necessary values have been set
	throw_if(!_numComponents, "call SetNumComponents() first");
	throw_if(_weights.size() != _numComponents, "call SetWeights() first");
	throw_if(_lambdas.size() != _numComponents, "call SetLambdas() first");
	throw_if(_omegas.size() != _numComponents, "call SetOmegas() first");
	throw_if(_apertureFullSize[0] == 0.0 || _apertureFullSize[1] == 0.0, "call SetAperature() first");
	throw_if(_imgWidth == 0 || _imgHeight == 0, "call SetImageDimensions() first");
	throw_if(_precalculated, "PrecalculateRSD() has already been called");

	const float c_speed = 299792458.0f;
	
	// useful values to have around
	_waveSize = _imgHeight * _imgWidth * 2;
	_depthSize = _numComponents * _waveSize;

	int i = 0;
	_rsd = CudaAllocImg(_numDepths, _numComponents);

	cufftHandle fft_plan;
	cufft_throw_if(cufftPlan2d(&fft_plan, _imgHeight, _imgWidth, CUFFT_C2C), "Error creating cufft plan");

	int depth_idx;
	float depth;
	std::cout << " _info.d_min: " << _info.d_min << std::endl;
	std::cout << " _info.d_max: " << _info.d_max << std::endl;
	std::cout << " _info.d_d: " << _info.d_d << std::endl;
	for (depth = _info.d_min, depth_idx = 0; depth < _info.d_max; depth += _info.d_d, depth_idx++) {
		float t_tmp = (depth + _info.d_offset) / c_speed;

		std::cout << "  i=" << i << " depth_idx=" << depth_idx << " depth=" << depth << std::endl;
		for (int wave_num = 0; wave_num < _numComponents; wave_num++) {
			float lambda = _lambdas[wave_num];
			float omega = _omegas[wave_num];

			float* d_ker = ImgAt(_rsd, depth_idx, wave_num);
			RSDKernelConvolution(lambda, depth, omega, t_tmp, fft_plan, d_ker);
		}
		i++;
	}
	assert(i == _numDepths);

	cufft_throw_if(cufftDestroy(fft_plan), "Error destroying cufft plan");

	// now allocate all the storage that the ReconstructImage() function will need
	_img2D = CudaAllocImg(1, 1, false);

	// allocate intermediate storage used for ffts during reconstruction
	_uTotalFFTs = CudaAllocImg(1, _numComponents);
	_uOut = CudaAllocImg(1, _numComponents);
	_uSum = CudaAllocImg(1, 1);

	// allocate cufft plan for use during reconstruction
	cufftPlan2d(&_fftPlan, _imgHeight, _imgWidth, CUFFT_C2C);

	EnableCubeGeneration(true);
	PrecalculateDDAWeights();

	// record how long this took, and flag that we've done it.
	_precalculated = true;
	_currentCount = -1; // we haven't reconstructed any yet.	
}

void RSDReconstructor::PrecalculateDDAWeights()
{
	std::vector<float> w(_numDepths * 3);
	// todo: remove assumption that min_depth = 1, and max_depth = 3
	float min_depth = 1.0f;
	float max_depth = 3.0f;

	float depth_range = max_depth - min_depth;

	for (int i = 0; i < _numDepths; i++) {
		float cen = (_numDepths - 1) / (depth_range * i + _numDepths - 1);
		float lr = (1.f - cen) / 2.f; // the 3 values are a partition of unity

		w[_numDepths + i] = cen;
		w[i] = w[_numDepths * 2 + i] = lr;
	}
	_ddaWeights = CudaAllocFloatArr(_numDepths * 3);
	cuda_throw_if(cudaMemcpy(_ddaWeights.get(), w.data(), _numDepths * 3 * sizeof(float), cudaMemcpyHostToDevice), "Error copying dda weights to gpu");
}

void RSDReconstructor::EnableCubeGeneration(bool enable)
{
	_cubeImages = CudaAllocImg(_numDepths, 4, false); // 4 cubes, so we can store 3 reconstructions for dda, plus a temporary one for computation
}

// call after each time full set of images has been added
void RSDReconstructor::ReconstructImage(cv::Mat& img_out)
{
	throw_if(!_precalculated, "Call PrecalculateRSD() first.");

	_currentCount++;

	// compute the ffts of all the images
	for (int wave_num = 0; wave_num < _numComponents; wave_num++) {
		cufftResult res = cufftExecC2C(_fftPlan,
			reinterpret_cast<cufftComplex*>(ImgAt(_imgData, wave_num)),
			reinterpret_cast<cufftComplex*>(ImgAt(_uTotalFFTs, wave_num)),
			CUFFT_FORWARD);
	}

	int i;
	float depth;
	for (depth = _info.d_min, i = 0; depth < _info.d_max; depth += _info.d_d, i++)
	{
		// Temporal output wavefront at the depth plane
		//auto ptr_zero_check = DbgInspect(_uSum.get(), 40);
		cudaMemset(_uSum.get(), 0x00, _waveSize * sizeof(float));
		//ptr_zero_check = DbgInspect(_uSum.get(), 40);

		// all convolutions for this depth
		int num_threads = 512;
		int num_blocks = (_imgWidth * _imgHeight + num_threads - 1) / num_threads;
		MulSpectrumMany<<<dim3(num_blocks, _numComponents), num_threads>>> (
			_uTotalFFTs.get(),
			ImgAt(_rsd, i, 0),
			_uOut.get(),
			SliceNumPixels());

		// summing
		// for-loop summing ws faster than parallel reduction kernel (~0.655ms vs ~1.0-1.3ms)
		for (int wave_num = 0; wave_num < _numComponents; wave_num++) {
			AddScale << <_imgHeight, _imgWidth >> > (
				_uSum.get(),
				ImgAt(_uOut, wave_num),
				_weights[wave_num]);
		}
		// optional parallel reduction version instead (slower...)
		//AddScaleMany<<<dim3(img_width, img_height), 128>>>(u_sum(0,0), u_out(0,0), d_weights.get(), u_num);

		// idft after integration
		cufftResult res = cufftExecC2C(_fftPlan, reinterpret_cast<cufftComplex*>(_uSum.get()),
			reinterpret_cast<cufftComplex*>(_uSum.get()),
			CUFFT_INVERSE);

		// store this slice
		int idx = _currentCount % 3;
		Abs<<<_imgHeight, _imgWidth>>>(_uSum.get(), 
			_cubeImages.get()  +  // base addr of 3 cubes
				idx * CubeNumPixels() + // offset to the proper cube
				i * SliceNumPixels()); // offset to the proper slice (depth)
	}

	cv::Mat img_2d(cv::Size(_imgWidth, _imgHeight), CV_32FC1);
	if (_currentCount > 0) {
		// we have the next frame, so we can process the previous frame.
		int idx_prev = (_currentCount - 1) % 3;
		if (_useDDA) {
			// perform depth dependent averaging across the 3 stored frames
			DDA<<<dim3(_imgHeight, _numDepths, 1), _imgWidth>>> (_cubeImages.get(), idx_prev, _ddaWeights.get());
			MaxZ<<<_imgHeight, _imgWidth >> > (CubeAt(_cubeImages, 3), _numDepths, _img2D.get());
		}
		else {
			// depth dependent averaging is turned off, just pick the max from the current (previous) single frame.
			MaxZ<<<_imgHeight, _imgWidth >> > (CubeAt(_cubeImages, idx_prev), _numDepths, _img2D.get());
		}

		// copy back to host
		cuda_throw_if(cudaMemcpy(img_2d.data, _img2D.get(), SliceNumPixels() * sizeof(float), cudaMemcpyDeviceToHost), "Error copying final image from device to host");
	}
	
	// shift picture back to center, then flip it
	FFTShift(img_2d);
	cv::flip(img_2d, img_2d, 1);

	// Up sampling
	float up = 1.0f / _downsamplingRatio;
	cv::resize(img_2d, img_out, cv::Size(), up, up); // wavefront spatial upsampling for larger image
}

void RSDReconstructor::RSDKernelConvolution(const float lambda, const float depth, const float omega, const float t, cufftHandle fft_plan, float* d_ker)
{
	// get physical aprture size
	float apt_x;
	int dim_x, dim_y; // Matrix dimension

	apt_x = _appertureDst[0]; // physical aperture x, unit meter
	//apt_y = apt.at<float>(0, 1); // physical aperture y, unit meter

	dim_x = _imgHeight; // input wavefront matrix size
	dim_y = _imgWidth;  // input wavefront matrix size 

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
	_allocatedBytes += sz;
	return ret;
}

shared_ptr<float> RSDReconstructor::CudaAllocImg(int depths, int waves, bool complex)
{
	int n = _imgHeight * _imgWidth * depths * waves * (complex ? 2 : 1);
	return CudaAllocFloatArr(n);
}

float* RSDReconstructor::CubeAt(shared_ptr<float> p, int cube_num)
{
	assert(p);
	return &(p.get()[cube_num * _imgHeight * _imgWidth * _numDepths]);
}

float* RSDReconstructor::ImgAt(shared_ptr<float> p, int wave_num)
{
	assert(p);
	return &(p.get()[wave_num * _waveSize]);
}

float* RSDReconstructor::ImgAt(shared_ptr<float> p, int depth, int wave_num)
{
	assert(p);
	return &(p.get()[depth * _depthSize + wave_num * _waveSize]);
}

std::unique_ptr<std::vector<float>> RSDReconstructor::GetImageData()
{
	return DbgInspect(_imgData.get(), _imgWidth * _imgHeight * 2 * _numComponents);
}

std::unique_ptr<std::vector<float>> RSDReconstructor::GetCubeData()
{
	int cur_idx = _currentCount % 3;
	auto ret = DbgInspect(_cubeImages.get() + cur_idx * _imgWidth * _imgHeight * _numDepths,
		_imgWidth * _imgHeight * _numDepths);
	
	// there's a faster way to do this, but this is only for debug logging...
	int sz[] = { _imgHeight, _imgWidth };
	//#pragma omp parallel
	for (int i = 0; i < _numDepths; i++) {
		float* p = ret->data() + i * _imgHeight * _imgWidth;
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