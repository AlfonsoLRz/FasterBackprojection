#include "stdafx.h"
#include "FastLCTReconstruction.h"

#include <cccl/cub/device/device_reduce.cuh>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "CudaHelper.h"
#include "CudaPerf.h"
#include "fourier.cuh"
#include "fast_lct.cuh"
#include "transient_postprocessing.cuh"
#include "ViewportSurface.h"

//

FastLCTReconstruction::FastLCTReconstruction()
	: _volumeResolution(0)
	  , _psfKernel(nullptr), _mtx(nullptr)
	  , _multResult(nullptr), _fftPlan2D(0), _blockSize1D(0), _gridSize1D(0)
{
}

FastLCTReconstruction::~FastLCTReconstruction()
{
	FastReconstructionAlgorithm::destroyResources();
	FastLCTReconstruction::destroyResources();
}

void FastLCTReconstruction::destroyResources()
{
	FastReconstructionAlgorithm::destroyResources();

	CudaHelper::reset(_psfKernel);
	CudaHelper::reset(_mtx);
	CudaHelper::reset(_multResult);

	if (_fftPlan2D != 0)
	{
		CUFFT_CHECK(cufftDestroy(_fftPlan2D));
		_fftPlan2D = 0;
	}
}

void FastLCTReconstruction::precalculate()
{
	FastReconstructionAlgorithm::precalculate();

	// Variables
	_volumeResolution = glm::uvec3(_imageHeight, _imageWidth, _numFrequencies);

	// FFT
	//int rank = 3;
	//int n[3] = { static_cast<int>(_volumeResolution.x),
	//			 static_cast<int>(_volumeResolution.y),
	//			 static_cast<int>(_volumeResolution.z) };
	//CUFFT_CHECK(cufftPlanMany(&_fftPlan, rank, n,
	//	NULL, 1, 0,			// idist and odist do not matter when the number of batches is 1
	//	NULL, 1, 0,
	//	CUFFT_C2C, 1));

	// Launch dims
	_blockSize1D = 512;
	_gridSize1D = CudaHelper::getNumBlocks(_frequencyCubeSize, _blockSize1D);

	_blockSize3D = dim3(8, 8, 8);
	_gridSize3D = dim3(
		(_volumeResolution.z + _blockSize3D.x - 1) / _blockSize3D.x,
		(_volumeResolution.y + _blockSize3D.y - 1) / _blockSize3D.y,
		(_volumeResolution.x + _blockSize3D.z - 1) / _blockSize3D.z
	);

	// Forward transform operator (mtxi is simply the transpose of mtx)
	defineTransformOperator(_numFrequencies, _mtx, _cudaStreams[0]);

	// Define the point spread function (PSF) kernel
	float tDistance = static_cast<float>(_numFrequencies) * _info._deltaDistance;
	_psfKernel = definePSFKernel(glm::abs(_info._apertureDstWidth / tDistance), _cudaStreams[1]);

	synchronizeStreams(2);

	// Other buffers
	CudaHelper::initializeZeroBuffer(_multResult, _frequencyCubeSize);
}

// call after each time full set of images has been added
void FastLCTReconstruction::reconstructImage(ViewportSurface* viewportSurface)
{
	assert(_precalculated);

	_currentCount++;

	CudaPerf perf;
	perf.setAlgorithmName("FastLCT");
	perf.tic();

	perf.tic("FFT");
	for (int frequencyIdx = 0; frequencyIdx < _numFrequencies; ++frequencyIdx)
	{
		CUFFT_CHECK(cufftSetStream(_fftPlan2D, _cudaStreams[frequencyIdx]));
		CUFFT_CHECK(cufftExecC2C(_fftPlan2D,
			_spadData + frequencyIdx * _sliceSize,
			_spadData + frequencyIdx * _sliceSize,
			CUFFT_FORWARD));
	}
	synchronizeStreams(_numFrequencies);
	perf.toc();

	perf.tic("Reconstruct Image");
	transformData(_mtx);
	perf.toc();

	//perf.tic("FFT + PSF + IFFT");
	//multiplyKernel(transformedData, psfKernel, volumeResolution, fftPlan);
	//perf.toc();

	//// Inverse transform the data
	//perf.tic("Inverse transform");
	//inverseTransformData(transformedData, intensityGpu, volumeResolution, mtx);
	//perf.toc();

	perf.toc();
	perf.summarize();
}

cufftComplex* FastLCTReconstruction::definePSFKernel(float slope, cudaStream_t stream)
{
	glm::uvec3 totalRes = _volumeResolution * 2u; // Assuming the PSF kernel is twice the resolution in each dimension
	glm::uint size = totalRes.x * totalRes.y * totalRes.z;

	// Gpu-side memory allocation
	float* psf = nullptr, * singleFloat = nullptr;
	cufftComplex* rolledPsf = nullptr;
	void* tempStorage = nullptr;
	size_t tempStorageBytes = 0;

	CudaHelper::initializeBufferAsync(psf, size, static_cast<float*>(nullptr), stream);
	CudaHelper::initializeBufferAsync(rolledPsf, size, static_cast<cufftComplex*>(nullptr), stream);
	CudaHelper::initializeBufferAsync(singleFloat, 1, static_cast<float*>(nullptr), stream);

	cub::DeviceReduce::Sum(nullptr, tempStorageBytes, psf, psf, size, stream);
	cudaMallocAsync(&tempStorage, tempStorageBytes, stream);

	// RSD
	{
		fast_lct::computePSFKernel<<<_gridSize3D, _blockSize3D, 0, stream>>>(psf, totalRes, slope);
	}

	// Find minimum along z-axis and binarize (only 1 value per xy, or a few at most)
	{
		dim3 blockSize2D(16, 16);
		dim3 gridSize2D(
			(totalRes.x + blockSize2D.x - 1) / blockSize2D.x,
			(totalRes.y + blockSize2D.y - 1) / blockSize2D.y
		);

		fast_lct::findMinimumBinarize<<<gridSize2D, blockSize2D, 0, stream>>>(psf, totalRes);
	}

	// Normalization according to center 
	{
		const glm::uint sumBaseIndex = _volumeResolution.x * totalRes.y * totalRes.z + _volumeResolution.y * totalRes.z;

		CudaHelper::checkError(cudaMemsetAsync(singleFloat, 0, sizeof(float), stream));

		cub::DeviceReduce::Sum(nullptr, tempStorageBytes, psf + sumBaseIndex, singleFloat, totalRes.z, stream);
		cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, psf + sumBaseIndex, singleFloat, totalRes.z, stream);

		fast_lct::normalizePSF<<<_gridSize3D, _blockSize3D, 0, stream>>>(psf, singleFloat, totalRes);
	}

	// L2 normalization
	{
		CudaHelper::checkError(cudaMemsetAsync(singleFloat, 0, sizeof(float), stream));

		cub::DeviceReduce::Sum(nullptr, tempStorageBytes, psf, singleFloat, size, stream);
		cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, psf, singleFloat, size, stream);

		fast_lct::l2NormPSF<<<_gridSize3D, _blockSize3D, 0, stream>>>(psf, singleFloat, totalRes);
	}

	fast_lct::rollPSF<<<_gridSize3D, _blockSize3D, 0, stream>>>(psf, rolledPsf, _volumeResolution, totalRes);

	// Fourier transform the PSF kernel
	{
		//CUFFT_CHECK(cufftSetStream(_fftPlan, stream));
		//CUFFT_CHECK(cufftExecC2C(_fftPlan, rolledPsf, rolledPsf, CUFFT_FORWARD));
	}

	// Wiener filter
	{
		fast_lct::wienerFilterPsf<<<_gridSize3D, _blockSize3D, 0, stream>>>(rolledPsf, totalRes, 8e-1);
	}

	CudaHelper::freeAsync(psf, stream);
	CudaHelper::freeAsync(singleFloat, stream);
	CudaHelper::freeAsync(tempStorage, stream);

	return rolledPsf;
}

void FastLCTReconstruction::defineTransformOperator(glm::uint M, float*& d_mtx, cudaStream_t stream)
{
	using namespace Eigen;
	using SparseMatrixF_RowMajor = Eigen::SparseMatrix<float, Eigen::RowMajor>;  // For efficient row access
	using TripletF = Eigen::Triplet<float>;

	glm::uint M2 = M * M;

	// This is the initial matrix; not a diagonal matrix but kind of
	SparseMatrixF_RowMajor mtx(M2, M);
	{
		std::vector<TripletF> triplets(M2);

#pragma omp parallel for
		for (int i = 0; i < M2; ++i)
		{
			float sqrt_i = std::sqrt(static_cast<float>(i + 1));
			int col = static_cast<int>(std::ceil(sqrt_i) - 1);
			triplets[i] = TripletF(i, col, 1.0f / sqrt_i);
		}

		mtx.setFromTriplets(triplets.begin(), triplets.end());
		mtx.makeCompressed();
	}

	// Hierarchical downsampling
	int K = static_cast<int>(std::round(std::log2(M)));

	for (int k = 0; k < K; ++k)
	{
		// Downsample rows (efficient with row-major mtx) 
		Index newRows = mtx.rows() / 2;
		SparseMatrixF_RowMajor mtx_new(newRows, mtx.cols());

		// Thread-local storage for triplets 
		std::vector<std::vector<TripletF>> thread_avg_triplets(omp_get_max_threads());

#pragma omp parallel
		{
			int threadID = omp_get_thread_num();
			// Reserve based on expected non-zeros per thread (rough estimate)
			thread_avg_triplets[threadID].reserve(mtx.nonZeros() / (2 * omp_get_num_threads()) + 10);

#pragma omp for nowait
			for (int i = 0; i < newRows; ++i)
			{
				int row1 = 2 * i;
				int row2 = 2 * i + 1;

				// Get iterators for the two rows
				SparseMatrixF_RowMajor::InnerIterator it1(mtx, row1);
				SparseMatrixF_RowMajor::InnerIterator it2(mtx, row2);

				// Merge-like operation for sparse vector addition
				while (it1 || it2)
				{
					if (it1 && (!it2 || it1.col() < it2.col()))
					{
						// row1
						thread_avg_triplets[threadID].emplace_back(i, it1.col(), 0.5f * it1.value());
						++it1;
					}
					else if (it2 && (!it1 || it2.col() < it1.col()))
					{
						// row2
						thread_avg_triplets[threadID].emplace_back(i, it2.col(), 0.5f * it2.value());
						++it2;
					}
					else if (it1 && it2 && it1.col() == it2.col())
					{
						// From both rows, sum values
						float sum_val = 0.5f * (it1.value() + it2.value());
						// Only add if not effectively zero (due to floating point arithmetic)
						if (std::abs(sum_val) > glm::epsilon<float>())
							thread_avg_triplets[threadID].emplace_back(i, it1.col(), sum_val);
						++it1;
						++it2;
					}
					else
					{
						break;
					}
				}
			}
		}

		// Collect all thread-local triplets into a single vector
		std::vector<TripletF> combinedAvgTriplets;
		for (const auto& local_vec : thread_avg_triplets)
			combinedAvgTriplets.insert(combinedAvgTriplets.end(), local_vec.begin(), local_vec.end());

		mtx_new.setFromTriplets(combinedAvgTriplets.begin(), combinedAvgTriplets.end());
		mtx_new.makeCompressed();
		mtx = mtx_new;
	}

	// Iterate over the sparse matrix and copy to dense
	std::vector<float> mtxHost(M2);
#pragma omp parallel for
	for (int k = 0; k < mtx.outerSize(); ++k)
		for (SparseMatrixF_RowMajor::InnerIterator it(mtx, k); it; ++it)
			mtxHost[it.col() * M + it.row()] = it.value();

	CudaHelper::initializeBufferAsync(d_mtx, mtxHost.size(), mtxHost.data(), stream);
}

void FastLCTReconstruction::multiplyKernel()
{
}

void FastLCTReconstruction::transformData(const float* mtx)
{
	// Scale the intensity values according to the material type (diffuse or not)
	float divisor = 1.0f / (static_cast<float>(_volumeResolution.z) - 1.0f);
	fast_lct::scaleIntensity<false><<<_gridSize1D, _blockSize1D>>>(_spadData, _volumeResolution, _frequencyCubeSize, divisor);

	// Multiply intensity by the transform matrix
	fast_lct::multiplyTransformTranspose<<<_gridSize3D, _blockSize3D >>>(_spadData, mtx, _multResult, _volumeResolution);
}

void FastLCTReconstruction::inverseTransformData()
{
}
