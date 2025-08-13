#include "../stdafx.h"
#include "rsd_reconstructor.h"


#include <cccl/cub/device/device_reduce.cuh>
#include <cufft.h>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "../CudaHelper.h"
#include "../../fourier.cuh"
#include "../../transient_postprocessing.cuh"
#include "rsd_cuda_kernels.h"
#include "../../CudaPerf.h"

using namespace std;

extern float* g_histo;

RSDReconstructor::RSDReconstructor()
	: _info()
	  , _useDDA(true)
	  , _numFrequencies(0)
	  , _numDepths(0)
	  , _imgHeight(0)
	  , _imgWidth(0)
	  , _apertureFullSize{0.0f, 0.0f}
	  , _apertureDst{0.0f, 0.0f}
	  , _samplingSpace(0.0f)
	  , _diffTopLeft(0), _diffLowerRight(0), _sliceSize(0)
	  , _frequencyCubeSize(0), _precalculated(false)
	  , _currentCount(0)
	  , _imgData(nullptr), _img2D(nullptr), _rsd(nullptr), _uTotalFFTs(nullptr)
	  , _uOut(nullptr), _uSum(nullptr), _cubeImages(nullptr), _ddaWeights(nullptr), _dWeights(nullptr)
	  , _maxValue(nullptr), _minValue(nullptr), _tempStorage(nullptr), _tempStorageBytes(0), _fftPlan2D(0)
	  , _blockSize1D(0), _gridSize1D(0)
{
}

RSDReconstructor::~RSDReconstructor()
{
	CudaHelper::free(_imgData);
	CudaHelper::free(_img2D);
	CudaHelper::free(_rsd);
	CudaHelper::free(_uTotalFFTs);
	CudaHelper::free(_uOut);
	CudaHelper::free(_uSum);
	CudaHelper::free(_cubeImages);
	CudaHelper::free(_ddaWeights);
	CudaHelper::free(_dWeights);
	CudaHelper::free(_maxValue);
	CudaHelper::free(_minValue);
	CudaHelper::free(_tempStorage);

	if (_fftPlan2D != 0)
		CUFFT_CHECK(cufftDestroy(_fftPlan2D));

	for (auto& stream : _cudaStreams)
		CudaHelper::checkError(cudaStreamDestroy(stream));
}

void RSDReconstructor::Initialize(const DatasetInfo& info)
{
	_info = info;

	// For now, aperature_dst is going to be the same as ApertureFullSize
	_apertureDst[0] = info.apt_dst_width;
	_apertureDst[1] = info.apt_dst_height;
	_numDepths = static_cast<int>(std::round((info.d_max - info.d_min) / info.d_d)) + 1;
}

void RSDReconstructor::EnableDepthDependentAveraging(bool useDDA)
{
	_useDDA = useDDA;
}

void RSDReconstructor::SetNumFrequencies(int n)
{
	assert(!_numFrequencies);
	assert(!_precalculated);
	_numFrequencies = n;
}

void RSDReconstructor::SetWeights(const float* weights)
{
	assert(_numFrequencies);
	assert(!_precalculated);
	_weights.clear();
	_weights.insert(_weights.end(), weights, weights + _numFrequencies);
}

void RSDReconstructor::SetLambdas(const float* lambdas)
{
	assert(_numFrequencies);
	assert(!_precalculated);
	_lambdas.clear();
	_lambdas.insert(_lambdas.end(), lambdas, lambdas + _numFrequencies);
}

void RSDReconstructor::SetOmegas(const float* omegas)
{
	assert(_numFrequencies);
	assert(!_precalculated);
	_omegas.clear();
	_omegas.insert(_omegas.end(), omegas, omegas + _numFrequencies);
}

void RSDReconstructor::SetSamplingSpace(const float sampling_space)
{
	assert(!_precalculated);
	_samplingSpace = sampling_space;
}

void RSDReconstructor::SetApertureFullSize(const float* apt)
{
	assert(!_precalculated);
	memcpy(_apertureFullSize, apt, sizeof(float) * 2);
}

void RSDReconstructor::SetImageDimensions(int width, int height)
{
	assert(_numFrequencies);
	assert(_apertureFullSize[0] > 0.0 && _apertureFullSize[1] > 0.0);
	assert(!_precalculated);

	// Calculate virtual aperature
	// Calculate the sampling spacing, need for re-calculated virtual aperture size
	float dx = (_apertureFullSize[0] / height); // Aperture sampling spacing

	// Calculate the difference between aperture
	float diff[2] { _info.apt_dst_width - _apertureFullSize[0], _info.apt_dst_height - _apertureFullSize[1] };

	// Zero padding
	diff[0] = diff[0] / dx; // transfer into pixel block
	diff[1] = diff[1] / dx; // transfer into pixel block

	_diffTopLeft = static_cast<int>(round(diff[0] / 2));
	_diffLowerRight = static_cast<int>(round(diff[1] / 2));

	_imgWidth = width;
	_imgHeight = height;

	_sliceSize = _imgHeight * _imgWidth;
	_frequencyCubeSize = _numFrequencies * _sliceSize;

	// Update the virtual aperture size
	_apertureDst[0] = static_cast<float>(_imgHeight) * dx;
	_apertureDst[1] = static_cast<float>(_imgWidth) * dx;

	// Allocate storage on gpu for images data
	CudaHelper::initializeBuffer(_imgData, SliceNumPixels() * _numFrequencies);
}

void RSDReconstructor::SetFFTData(const cufftComplex* data) const
{
	assert(_numFrequencies);
	assert(_imgWidth != 0 && _imgHeight != 0);

	cudaMemcpy(_imgData, data, static_cast<size_t>(_numFrequencies) * SliceNumPixels() * sizeof(cufftComplex), cudaMemcpyHostToDevice);
}

void RSDReconstructor::DumpInfo() const
{
	// Print values for testing
	cout << "name:                 " << _info.name << endl;
	cout << "number of components: " << _numFrequencies << endl;
	cout << "weight:               " << _weights.size() << endl;
	cout << "lambda_loop:          " << _lambdas.size() << endl;
	cout << "omega_space:          " << _omegas.size() << endl;
	cout << "aperture full size:    [" << _apertureFullSize[0] << ", " << _apertureFullSize[1] << "]" << endl;
	cout << "sampling spacing:     " << _samplingSpace << endl;
}

// Allocates all necessary GPU data for RSD Array and for reconstruction, and
// precalculates RSD. Basically sets everything up.
void RSDReconstructor::PrecalculateRSD()
{
	int i = 0;
	CudaHelper::initializeBuffer(_rsd, SliceNumPixels() * _numDepths * _numFrequencies);

	// Allocate the FFT plans
	CUFFT_CHECK(cufftPlan2d(&_fftPlan2D, _imgHeight, _imgWidth, CUFFT_C2C));

	//
	glm::uint depthIdx;
	float depth;

	for (depth = _info.d_min, depthIdx = 0; depth < _info.d_max; depth += _info.d_d, depthIdx++) 
	{
		float time = (depth + _info.d_offset) / LIGHT_SPEED;

		std::cout << "  i=" << i << " depth_idx=" << depthIdx << " depth=" << depth << std::endl;
		for (int wave_num = 0; wave_num < _numFrequencies; wave_num++) 
		{
			float lambda = _lambdas[wave_num];
			float omega = _omegas[wave_num];

			cufftComplex* kernelPtr = _rsd + depthIdx * _frequencyCubeSize + wave_num * _sliceSize;
			RSDKernelConvolution(kernelPtr, _fftPlan2D, lambda, omega, depth, time);
		}

		i++;
	}
	assert(i == _numDepths);

	// Now allocate all the storage that the ReconstructImage() function will need
	CudaHelper::initializeBuffer(_img2D, SliceNumPixels());
	CudaHelper::initializeBuffer(_dWeights, _weights.size(), _weights.data());

	// Allocate intermediate storage used for ffts during reconstruction
	CudaHelper::initializeBuffer(_uTotalFFTs, SliceNumPixels() * _numFrequencies);
	CudaHelper::initializeBuffer(_uOut, SliceNumPixels() * _numFrequencies * _numDepths);
	CudaHelper::initializeBuffer(_uSum, SliceNumPixels() * _numDepths);
	CudaHelper::initializeBuffer(_maxValue, 1);
	CudaHelper::initializeBuffer(_minValue, 1);

	// Temporary storage for max/min
	cub::DeviceReduce::Max(_tempStorage, _tempStorageBytes, _img2D, _maxValue, _sliceSize);
	CudaHelper::initializeBuffer(_tempStorage, _tempStorageBytes);

	// Allocate cufft plan for use during reconstruction
	EnableCubeGeneration(true);
	PrecalculateDDAWeights();

	// Flag that we've done it.
	_precalculated = true;
	_currentCount = -1; // we haven't reconstructed any yet.

	// Prepare streams for parallel processing
	_cudaStreams.resize(_numFrequencies);
	for (int j = 0; j < _numFrequencies; j++) 
	{
		cudaStream_t stream;
		CudaHelper::checkError(cudaStreamCreate(&stream));
		_cudaStreams[j] = stream;
	}

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
		(_imgWidth + _blockSize2D_pix.x - 1) / _blockSize2D_pix.x,
		(_imgHeight + _blockSize2D_pix.y - 1) / _blockSize2D_pix.y
	);
}

void RSDReconstructor::PrecalculateDDAWeights()
{
	std::vector<float> w(_numDepths * 3);

	// TODO: remove assumption that min_depth = 1, and max_depth = 3
	float minDepth = 1.0f;
	float maxDepth = 3.0f;
	float depthRange = maxDepth - minDepth;

	for (int i = 0; i < _numDepths; i++) 
	{
		float cen = (_numDepths - 1) / (depthRange * i + _numDepths - 1);
		float lr = (1.f - cen) / 2.f; // the 3 values are a partition of unity

		w[_numDepths + i] = cen;
		w[i] = w[_numDepths * 2 + i] = lr;
	}

	CudaHelper::initializeBuffer(_ddaWeights, _numDepths * 3, w.data());
}

void RSDReconstructor::EnableCubeGeneration(bool enable)
{
	CudaHelper::initializeBuffer(_cubeImages, SliceNumPixels() * _numDepths * 4);
	// 4 cubes, so we can store 3 reconstructions for dda, plus a temporary one for computation
}

// call after each time full set of images has been added
void RSDReconstructor::ReconstructImage(cv::Mat& img_out)
{
	CudaPerf perf(false);
	perf.setAlgorithmName("RSDReconstructor::ReconstructImage");
	perf.tic();

	assert(_precalculated);

	_currentCount++;

	perf.tic("FFT");
	for (int frequencyIdx = 0; frequencyIdx < _numFrequencies; ++frequencyIdx)
	{
		CUFFT_CHECK(cufftSetStream(_fftPlan2D, _cudaStreams[frequencyIdx]));
		CUFFT_CHECK(cufftExecC2C(_fftPlan2D,
			_imgData + frequencyIdx * _sliceSize,
			_uTotalFFTs + frequencyIdx * _sliceSize,
			CUFFT_FORWARD));
	}

	for (int i = 0; i < _numFrequencies; i++)
		cudaStreamSynchronize(_cudaStreams[i]);

	perf.toc();

	//
	int depthIndex;
	float depth;

	// Temporal output wavefront at the depth plane
	cudaMemset(_uSum, 0, _sliceSize * _numDepths * sizeof(cufftComplex));
	glm::uint idx = _currentCount % 3;

	for (depth = _info.d_min, depthIndex = 0; depth < _info.d_max; depth += _info.d_d, ++depthIndex)
	{
		multiplySpectrumMany<<<_gridSize2D_freq, _blockSize2D_freq, 0, _cudaStreams[depthIndex]>>>(
			_uTotalFFTs,
			_rsd + depthIndex * _frequencyCubeSize,
			_uSum + depthIndex * _sliceSize,
			_dWeights,
			_numFrequencies, _sliceSize);

		// IFFT after integration
		CUFFT_CHECK(cufftSetStream(_fftPlan2D, _cudaStreams[depthIndex]));
		CUFFT_CHECK(cufftExecC2C(_fftPlan2D, _uSum + depthIndex * _sliceSize, _uSum + depthIndex * _sliceSize, CUFFT_INVERSE));

		// Store this slice
		abs<<<_gridSize1D, _blockSize1D, 0, _cudaStreams[depthIndex]>>>(
				_uSum + depthIndex * _sliceSize,
				_cubeImages + idx * CubeNumPixels() + depthIndex * SliceNumPixels(), 
				_sliceSize);
	}

	for (int i = 0; i < depthIndex; i++)
		cudaStreamSynchronize(_cudaStreams[i]);

	cv::Mat img_2d(cv::Size(_imgWidth, _imgHeight), CV_32FC1);
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
				_cubeImages + 3 * _sliceSize * _numDepths, _img2D,
				_imgWidth, _imgHeight, _numDepths, _sliceSize, glm::uvec2(_imgWidth / 2, _imgHeight / 2));
		}
		else 
		{
			// Depth dependent averaging is turned off, just pick the max from the current (previous) single frame.
			maxZ<<<_gridSize2D_pix, _blockSize2D_pix>>>(
				_cubeImages + previousIdx * _sliceSize * _numDepths, _img2D,
				_imgWidth, _imgHeight, _numDepths, _sliceSize, glm::uvec2(_imgWidth / 2, _imgHeight / 2));
		}

		{
			cub::DeviceReduce::Max(_tempStorage, _tempStorageBytes, _img2D, _maxValue, _sliceSize);
			cub::DeviceReduce::Min(_tempStorage, _tempStorageBytes, _img2D, _minValue, _sliceSize);
			normalizeReconstruction<<<_gridSize1D, _blockSize1D>>>(_img2D, _sliceSize, _maxValue, _minValue);
		}

		CudaHelper::synchronize("");
		perf.toc();
		perf.summarize();

		// Copy back to host
		CudaHelper::checkError(cudaMemcpy(img_2d.data, _img2D, SliceNumPixels() * sizeof(float), cudaMemcpyDeviceToHost));
	}
	
	// Shift picture back to center, then flip it
	//FFTShift(img_2d);
	cv::flip(img_2d, img_2d, 1);

	// Visualization
	img_2d.convertTo(img_2d, CV_8UC1, 255.0, 0);
	cv::imwrite("logs/rsd_reconstruction_" + std::to_string(_currentCount) + ".png", img_2d);
}

void RSDReconstructor::RSDKernelConvolution(
	cufftComplex* dKernel, cufftHandle fftPlan, 
	const float lambda, const float omega, 
	const float depth, const float t) const
{
	float apertureX = _apertureDst[0];			// Physical aperture x, unit meter
	glm::uint dim_x = _imgWidth, dim_y = _imgHeight;
	float dx = apertureX / static_cast<float>(dim_x - 1);			// Spatial sampling density

	// Dimensions of thread launch
	dim3 blockSize(16, 16);
	dim3 gridSize(
		(dim_y + blockSize.x - 1) / blockSize.x,
		(dim_x + blockSize.y - 1) / blockSize.y
	);

	// Apply RSD convolution kernel equation
	float z_hat = depth / dx;
	float mulSquare = lambda * z_hat / (static_cast<float>(dim_x) * dx);
	rsdKernel<<<gridSize, blockSize>>>(dKernel, z_hat * z_hat, mulSquare, dim_x, dim_y);
	CudaHelper::synchronize("rsdKernel");

	// Perform fft on gpu
	CUFFT_CHECK(cufftExecC2C(fftPlan, dKernel, dKernel, CUFFT_FORWARD));
	CudaHelper::synchronize("cufftExecC2C");

	// Convolve the two by pointwise multiplication of the spectrums
	multiplySpectrumExpHarmonic<<<gridSize, blockSize>>>(dKernel, omega, t, dim_x, dim_y);
	CudaHelper::synchronize("multiplySpectrumExpHarmonic");
}

float* RSDReconstructor::CubeAt(shared_ptr<float> p, int cube_num) const
{
	assert(p);
	return &(p.get()[cube_num * _imgHeight * _imgWidth * _numDepths]);
}

cufftComplex* RSDReconstructor::ImageAt(cufftComplex* p, int depth) const
{
	assert(p);
	return &(p[depth * _frequencyCubeSize]);
}

void RSDReconstructor::FFTShift(cv::Mat& out)
{
	cv::Size sz = out.size();
	cv::Point pt(0, 0);
	pt.x = static_cast<int>(floor(sz.width / 2.0));
	pt.y = static_cast<int>(floor(sz.height / 2.0));
	CircShift(out, pt);
}

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