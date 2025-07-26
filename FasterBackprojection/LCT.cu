#include "stdafx.h"
#include "LCT.h"

#include <future>
#include <cub/device/device_reduce.cuh>

#include "cusparse.h"
#include "lct.cuh"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>

#include "CudaHelper.h"
#include "ChronoUtilities.h"
#include "FileUtilities.h"
#include "fourier.cuh"
#include "Image.h"
#include "TransientImage.h"
#include "TransientParameters.h"
#include "transient_postprocessing.cuh"

//

void LCT::reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	const glm::uvec3 volumeResolution = glm::uvec3(_nlosData->_dims[0], _nlosData->_dims[1], _nlosData->_dims[2]);
	float tDistance = static_cast<float>(recInfo._numTimeBins) * recInfo._timeStep;
	float* intensityGpu = recBuffers._intensity;
	float* mtx;

	_perf.tic("PSF, transform operators & transform data");

	// Define two cuda streams for asynchronous operations
	cudaStream_t stream1, stream2;
	CudaHelper::checkError(cudaStreamCreate(&stream1));
	CudaHelper::checkError(cudaStreamCreate(&stream2));

	// Forward and backward transform operators
	std::future<void> future = std::async(std::launch::async,
	                                      defineTransformOperator,
											recInfo._numTimeBins,
	                                      std::ref(mtx));

	// Define the point spread function (PSF) kernel
	cufftComplex* psfKernel = definePSFKernel(volumeResolution, glm::abs(_nlosData->_temporalWidth / tDistance), stream1);

	// Transform data using previous operators
	future.get();
	float* transformedData = transformData(intensityGpu, volumeResolution, mtx, stream2);

	cuStreamSynchronize(stream1);
	cuStreamSynchronize(stream2);
	emptyQueue();

	_perf.toc();

	// FFT + PSF + IFFT
	_perf.tic("FFT + PSF + IFFT");
	multiplyKernel(transformedData, psfKernel, volumeResolution);
	_perf.toc();

	// Inverse transform the data
	_perf.tic("Inverse transform");
	inverseTransformData(transformedData, intensityGpu, volumeResolution, mtx);
	_perf.toc();

	emptyQueue();
	CudaHelper::free(transformedData);
	CudaHelper::free(psfKernel);
	CudaHelper::free(mtx);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}

void LCT::reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo)
{
}

cufftComplex* LCT::definePSFKernel(const glm::uvec3& dataResolution, float slope, cudaStream_t stream)
{
	glm::uvec3 totalRes = dataResolution * 2u; // Assuming the PSF kernel is twice the resolution in each dimension
    glm::uint size = totalRes.x * totalRes.y * totalRes.z;

	// Gpu-side memory allocation
	float* psf = nullptr, *singleFloat = nullptr;
	cufftComplex* rolledPsf = nullptr;
	void* tempStorage = nullptr;
	size_t tempStorageBytes = 0;

	CudaHelper::initializeBufferGPU(psf, size);
	CudaHelper::initializeBufferGPU(rolledPsf, size);
	CudaHelper::initializeBufferGPU(singleFloat, 1);

	cub::DeviceReduce::Sum(nullptr, tempStorageBytes, psf, psf, size);
	cudaMalloc(&tempStorage, tempStorageBytes);
	
    dim3 blockSize(16, 8, 8);
    dim3 gridSize(
		(totalRes.z + blockSize.x - 1) / blockSize.x,
        (totalRes.y + blockSize.y - 1) / blockSize.y,
        (totalRes.x + blockSize.z - 1) / blockSize.z
	);

	// FFT
	cufftHandle fftPlan;
	int rank = 3; 
	int n[3] = { static_cast<int>(totalRes.x),
				 static_cast<int>(totalRes.y),
				 static_cast<int>(totalRes.z) };
	CUFFT_CHECK(cufftPlanMany(&fftPlan, rank, n,
		NULL, 1, 0,			// idist and odist do not matter when the number of batches is 1
		NULL, 1, 0,
		CUFFT_C2C, 1));

	// RSD
	{
		computePSFKernel<<<gridSize, blockSize, 0, stream>>>(psf, totalRes, slope);
	}

	// Find minimum along z-axis and binarize (only 1 value per xy, or a few at most)
	{
		dim3 blockSize2D(16, 16);
		dim3 gridSize2D(
			(totalRes.x + blockSize2D.x - 1) / blockSize2D.x,
			(totalRes.y + blockSize2D.y - 1) / blockSize2D.y
		);

		findMinimumBinarize<<<gridSize2D, blockSize2D, 0, stream>>>(psf, totalRes);
	}

	// Normalization according to center 
	{
		const glm::uint sumBaseIndex = dataResolution.x * totalRes.y * totalRes.z + dataResolution.y * totalRes.z;

		CudaHelper::checkError(cudaMemset(singleFloat, 0, sizeof(float)));

		cub::DeviceReduce::Sum(nullptr, tempStorageBytes, psf + sumBaseIndex, singleFloat, totalRes.z);
		cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, psf + sumBaseIndex, singleFloat, totalRes.z);

		normalizePSF<<<gridSize, blockSize, 0, stream>>>(psf, singleFloat, totalRes);
	}

	// L2 normalization
	{
		CudaHelper::checkError(cudaMemset(singleFloat, 0, sizeof(float)));

		cub::DeviceReduce::Sum(nullptr, tempStorageBytes, psf, singleFloat, size);
		cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, psf, singleFloat, size);

		l2NormPSF<<<gridSize, blockSize, 0, stream>>>(psf, singleFloat, totalRes);
	}

    rollPSF<<<gridSize, blockSize, 0, stream>>>(psf, rolledPsf, dataResolution, totalRes);

	// Fourier transform the PSF kernel
	{
		CUFFT_CHECK(cufftSetStream(fftPlan, stream));
		CUFFT_CHECK(cufftExecC2C(fftPlan, rolledPsf, rolledPsf, CUFFT_FORWARD));
	}

	// Wiener filter
	{
		wienerFilterPsf<<<gridSize, blockSize, 0, stream>>>(rolledPsf, totalRes, 8e-1);
	}

	_deleteFloatQueue.push_back(psf);
	_deleteFloatQueue.push_back(singleFloat);
	_deleteVoidQueue.push_back(tempStorage);
	_deleteCufftHandles.push_back(fftPlan);

	return rolledPsf;
}

void LCT::defineTransformOperator(glm::uint M, float*& d_mtx)
{
	using namespace Eigen;
	using SparseMatrixF_RowMajor = Eigen::SparseMatrix<float, Eigen::RowMajor>;  // For efficient row access
	using SparseMatrixF_ColMajor = Eigen::SparseMatrix<float, Eigen::ColMajor>;  // For efficient column access
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

	std::vector<float> mtxHost(M2);

	// Iterate over the sparse matrix and copy to dense
	#pragma omp parallel for
	for (int k = 0; k < mtx.outerSize(); ++k)
		for (SparseMatrixF_RowMajor::InnerIterator it(mtx, k); it; ++it) 
			mtxHost[it.col() * M + it.row()] = it.value();

	CudaHelper::initializeBufferGPU(d_mtx, mtxHost.size(), mtxHost.data());
}

void LCT::multiplyKernel(float* volumeGpu, const cufftComplex* inversePSF, const glm::uvec3& dataResolution)
{
	//
	glm::uvec3 newDims = dataResolution * glm::uvec3(2);
	size_t newDimProduct = static_cast<size_t>(newDims.x) * newDims.y * newDims.z;

	// Transfer H to a padded H
	cufftComplex* d_H = nullptr;
	CudaHelper::initializeZeroBufferGPU(d_H, newDimProduct);

	ChronoUtilities::startTimer();
	dim3 blockSize(16, 8, 8);
	dim3 gridSize(
		(dataResolution.z + blockSize.x - 1) / blockSize.x,
		(dataResolution.y + blockSize.y - 1) / blockSize.y,
		(dataResolution.x + blockSize.z - 1) / blockSize.z
	);

	glm::uint rollStride = 16;
	dim3 blockSizeRolled(8, 8, 8);
	dim3 gridSizeRolled(
		(dataResolution.z / rollStride + blockSize.x - 1) / blockSize.x,
		(dataResolution.y + blockSize.y - 1) / blockSize.y,
		(dataResolution.x + blockSize.z - 1) / blockSize.z
	);
	padIntensityFFT<<<gridSize, blockSize>>>(volumeGpu, d_H, dataResolution, newDims);

	//
	cufftHandle planH;
	int rank = 3;  // 2D FFT
	int n[3] = { static_cast<int>(newDims[0]),
				 static_cast<int>(newDims[1]),
				 static_cast<int>(newDims[2])};

	CUFFT_CHECK(cufftPlanMany(&planH, rank, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(planH, d_H, d_H, CUFFT_FORWARD));

	// Multiply by inverse PSF
	constexpr glm::uint blockSize1D  = 256;
	const glm::uint numBlocks1D = (static_cast<glm::uint>(newDimProduct) + blockSize1D - 1) / blockSize1D;
	multiplyPSF<<<numBlocks1D, blockSize1D>>>(d_H, inversePSF, newDimProduct);

	// Inverse FFT
	CUFFT_CHECK(cufftExecC2C(planH, d_H, d_H, CUFFT_INVERSE));
	_deleteCufftHandles.push_back(planH);

	// IFFT requires normalization, but it also produces very small values, so we avoid this and produce valid results by normalizing later
	//normalizeIFFT<<<CudaHelper::getNumBlocks(newDimProduct, 512), 512>>>(d_H, newDimProduct, 1.0f / newDimProduct);

	//
	unpadIntensityFFT<<<gridSize, blockSize>>>(volumeGpu, d_H, dataResolution, newDims);
}

float* LCT::transformData(float* volumeGpu, const glm::uvec3& dataResolution, const float* mtx, cudaStream_t stream)
{
	float* multResult = nullptr;
	CudaHelper::initializeZeroBufferGPU(multResult, static_cast<size_t>(dataResolution.x) * dataResolution.y * dataResolution.z);

	glm::uint numElements = dataResolution.x * dataResolution.y * dataResolution.z;

	dim3 blockSize3D(16, 8, 8);
	dim3 gridSize3D(
		(dataResolution.z + blockSize3D.x - 1) / blockSize3D.x,
		(dataResolution.y + blockSize3D.y - 1) / blockSize3D.y,
		(dataResolution.x + blockSize3D.z - 1) / blockSize3D.z
	);

	glm::uint blockSize = 512;
	glm::uint gridSize = CudaHelper::getNumBlocks(numElements, blockSize);  

	// Scale the intensity values according to the material type (diffuse or not)
	float divisor = 1.0f / (static_cast<float>(dataResolution.z) - 1.0f);
	scaleIntensity<false><<<gridSize, blockSize, 0, stream>>>(volumeGpu, dataResolution, numElements, divisor);

	// Multiply intensity by the transform matrix
	multiplyTransformTranspose<<<gridSize3D, blockSize3D, 0, stream>>>(volumeGpu, mtx, multResult, dataResolution);

	return multResult;
}

void LCT::inverseTransformData(const float* volumeGpu, float* multResult, const glm::uvec3& dataResolution, float*& inverseMtx)
{
	dim3 blockSize3D(16, 8, 8);
	dim3 gridSize3D(
		(dataResolution.z + blockSize3D.x - 1) / blockSize3D.x,
		(dataResolution.y + blockSize3D.y - 1) / blockSize3D.y,
		(dataResolution.x + blockSize3D.z - 1) / blockSize3D.z
	);

	multiplyTransformTransposeInv<<<gridSize3D, blockSize3D>>>(volumeGpu, inverseMtx, multResult, dataResolution);
}

void LCT::reconstructDepths(NLosData* nlosData, const ReconstructionInfo& recInfo,
                            const ReconstructionBuffers& recBuffers, const TransientParameters& transientParams,
                            const std::vector<float>& depths)
{
}

void LCT::reconstructVolume(
	NLosData* nlosData, const ReconstructionInfo& recInfo,
	const ReconstructionBuffers& recBuffers, const TransientParameters& transientParams)
{
	_nlosData = nlosData;

	_perf.setAlgorithmName("LCT");
	_perf.tic();

	if (transientParams._compensateLaserCosDistance && 
		!glm::all(glm::epsilonEqual(_nlosData->_laserPosition, glm::vec3(0.0f), glm::epsilon<float>())) &&
		!glm::all(glm::epsilonEqual(_nlosData->_cameraPosition, glm::vec3(0.0f), glm::epsilon<float>())))
		compensateLaserCosDistance(recInfo, recBuffers);

	if (recInfo._captureSystem == CaptureSystem::Confocal)
		reconstructVolumeConfocal(nullptr, recInfo, recBuffers);
	else
		throw std::runtime_error("Unsupported capture system for LCT reconstruction.");

	const glm::uvec3 volumeResolution = glm::uvec3(nlosData->_dims[0], nlosData->_dims[1], nlosData->_dims[2]);
	float* volumeGpu = recBuffers._intensity;

	// Post-process the activation matrix
	_postprocessingFilters[transientParams._postprocessingFilterType]->compute(volumeGpu, volumeResolution, transientParams);
	normalizeMatrix(recBuffers._intensity, volumeResolution.x * volumeResolution.y * volumeResolution.z);

	_perf.toc();
	_perf.summarize();

	if (transientParams._saveMaxImage)
		LCT::saveMaxImage(
			transientParams._outputFolder + transientParams._outputMaxImageName,
			volumeGpu,
			volumeResolution,
			false, true);
}

//

void LCT::emptyQueue()
{
	for (auto& ptr : _deleteFloatQueue)
	{
		if (ptr != nullptr)
			CudaHelper::free(ptr);
	}

	_deleteFloatQueue.clear();
	for (auto& ptr : _deleteVoidQueue)
	{
		if (ptr != nullptr)
			CudaHelper::free(ptr);
	}

	_deleteVoidQueue.clear();
	for (auto& handle : _deleteCufftHandles)
	{
		if (handle != 0)
			CUFFT_CHECK(cufftDestroy(handle));
	}
	_deleteCufftHandles.clear();
}