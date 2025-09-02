#include "stdafx.h"
#include "PhasorFields.h"

#include "CudaHelper.h"

#include "fourier.cuh"
#include "phasor_fields.cuh"

#include <cccl/cub/device/device_reduce.cuh>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>

#include "transient_postprocessing.cuh"

//

void PhasorFields::reconstructDepths(
	NLosData* nlosData, const ReconstructionInfo& recInfo,
	const ReconstructionBuffers& recBuffers, const TransientParameters& transientParams,
	const std::vector<float>& depths)
{
}

void PhasorFields::reconstructVolume(
	NLosData* nlosData, const ReconstructionInfo& recInfo,
	const ReconstructionBuffers& recBuffers, const TransientParameters& transientParams)
{
	_nlosData = nlosData;

	_perf.setAlgorithmName("Phasor Fields");
	_perf.tic();

	const glm::uvec3 volumeResolution = glm::uvec3(nlosData->_dims[0], nlosData->_dims[1], nlosData->_dims[2]);
	float* volumeGpu = recBuffers._intensity;

	compensateLaserCosDistance(transientParams, recInfo, recBuffers);

	if (recInfo._captureSystem == CaptureSystem::Confocal)
		reconstructVolumeConfocal(volumeGpu, recInfo, recBuffers);
	else
		throw std::runtime_error("Unsupported capture system for Phasor Fields reconstruction.");

	// Post-process the activation matrix
	_postprocessingFilters[transientParams._postprocessingFilterType]->compute(volumeGpu, volumeResolution, transientParams);
	normalizeMatrix(volumeGpu, volumeResolution.x * volumeResolution.y * volumeResolution.z);

	_perf.toc();
	_perf.summarize();

	if (transientParams._saveMaxImage)
		saveMaxImage(
			transientParams._outputFolder + transientParams._outputMaxImageName,
			volumeGpu,
			volumeResolution,
			false);
}

//

void PhasorFields::definePSFKernel(const glm::uvec3& dataResolution, float slope, cufftComplex*& rolledPsf, cudaStream_t stream)
{
	glm::uvec3 totalRes = dataResolution * 2u; // Assuming the PSF kernel is twice the resolution in each dimension
	glm::uint size = totalRes.x * totalRes.y * totalRes.z;

	// Gpu-side memory allocation
	float* psf = nullptr, * singleFloat = nullptr;
	void* tempStorage = nullptr;
	size_t tempStorageBytes = 0;

	CudaHelper::initializeBuffer(psf, size);
	CudaHelper::initializeBuffer(rolledPsf, size);
	CudaHelper::initializeBuffer(singleFloat, 1);

	cub::DeviceReduce::Sum(nullptr, tempStorageBytes, psf, psf, size, stream);
	CudaHelper::initializeBuffer(tempStorage, tempStorageBytes);

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
	CUFFT_CHECK(cufftSetStream(fftPlan, stream));

	// RSD
	{
		computePSFKernel_pf<<<gridSize, blockSize, 0, stream>>>(psf, totalRes, slope);
	}

	// Find minimum along z-axis and binarize (only 1 value per xy, or a few at most)
	{
		dim3 blockSize2D(16, 16);
		dim3 gridSize2D(
			(totalRes.x + blockSize2D.x - 1) / blockSize2D.x,
			(totalRes.y + blockSize2D.y - 1) / blockSize2D.y
		);

		findMinimumBinarize_pf<<<gridSize2D, blockSize2D, 0, stream>>>(psf, totalRes);
	}

	// Normalization according to center 
	{
		const glm::uint sumBaseIndex = dataResolution.x * totalRes.y * totalRes.z + dataResolution.y * totalRes.z;

		CudaHelper::checkError(cudaMemset(singleFloat, 0, sizeof(float)));

		cub::DeviceReduce::Sum(nullptr, tempStorageBytes, psf + sumBaseIndex, singleFloat, totalRes.z, stream);
		cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, psf + sumBaseIndex, singleFloat, totalRes.z, stream);

		normalizePSF_pf<<<gridSize, blockSize, 0, stream>>>(psf, singleFloat, totalRes);
	}

	// L2 normalization
	{
		CudaHelper::checkError(cudaMemset(singleFloat, 0, sizeof(float)));

		cub::DeviceReduce::Sum(nullptr, tempStorageBytes, psf, singleFloat, size, stream);
		cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, psf, singleFloat, size, stream);

		l2NormPSF_pf <<<gridSize, blockSize, 0, stream>>>(psf, singleFloat, totalRes);
	}

	rollPSF_pf<<<gridSize, blockSize, 0, stream>>>(psf, rolledPsf, dataResolution, totalRes);

	// Fourier transform the PSF kernel
	{
		CUFFT_CHECK(cufftExecC2C(fftPlan, rolledPsf, rolledPsf, CUFFT_FORWARD));
	}

	// Get the conjugate of the PSF kernel
	{
		extractConjugate_pf<<<gridSize, blockSize, 0, stream>>>(rolledPsf, totalRes);
	}

	_cleanupQueue.push_back([psf, singleFloat, tempStorage, fftPlan]
	{
		CudaHelper::free(psf);
		CudaHelper::free(singleFloat);
		CudaHelper::free(tempStorage);
		cufftDestroy(fftPlan);
	});
}

void PhasorFields::defineTransformOperator(glm::uint M, float*& d_mtx)
{
	using namespace Eigen;
	using SparseMatrixF_RowMajor = SparseMatrix<float, Eigen::RowMajor>;  // For efficient row access
	using SparseMatrixF_ColMajor = Eigen::SparseMatrix<float, Eigen::ColMajor>;  // For efficient column access
	using TripletF = Triplet<float>;

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

	CudaHelper::initializeBuffer(d_mtx, mtxHost.size(), mtxHost.data());
}

void PhasorFields::virtualWaveConvolution(
	float* data, const glm::uvec3& dataResolution, float deltaDistance, float virtualWavelength, float cycles,
	float*& phasorCos, float*& phasorSin,
	cudaStream_t stream)
{
	const glm::uint numSamples = static_cast<int>(roundf(cycles * virtualWavelength / deltaDistance));
	const float numCycles = static_cast<float>(numSamples) * deltaDistance / virtualWavelength, sigma = 1.0f / 0.3f;

	// Build virtual waves
	float* virtualWaveSin = nullptr, *virtualWaveCos = nullptr;
	CudaHelper::initializeBuffer(virtualWaveSin, numSamples);
	CudaHelper::initializeBuffer(virtualWaveCos, numSamples);
	CudaHelper::initializeBuffer(phasorCos, dataResolution.x * dataResolution.y * dataResolution.z);
	CudaHelper::initializeBuffer(phasorSin, dataResolution.x * dataResolution.y * dataResolution.z);

	dim3 blockSize(256);
	dim3 gridSize((numSamples + blockSize.x - 1) / blockSize.x);

	dim3 blockSize3D(16, 8, 8);
	dim3 gridSize3D(
		(dataResolution.z + blockSize3D.x - 1) / blockSize3D.x,
		(dataResolution.y + blockSize3D.y - 1) / blockSize3D.y,
		(dataResolution.x + blockSize3D.z - 1) / blockSize3D.z
	);

	createVirtualWaves<<<gridSize, blockSize, 0, stream>>>(virtualWaveCos, virtualWaveSin, numSamples, numCycles, sigma);
	convolveVirtualWaves<<<gridSize3D, blockSize3D, 0, stream>>>(data, dataResolution, virtualWaveSin, virtualWaveCos, phasorCos, phasorSin, numSamples);
}

void PhasorFields::transformData(
	const glm::uvec3& dataResolution, 
	const float* mtx, 
	float* phasorCos, float* phasorSin,
	cufftComplex*& phasorDataCos, cufftComplex*& phasorDataSin,
	cudaStream_t stream)
{
	glm::uvec3 fftResolution = dataResolution * 2u;

	CudaHelper::initializeZeroBuffer(phasorDataCos, static_cast<size_t>(fftResolution.x) * fftResolution.y * fftResolution.z);
	CudaHelper::initializeZeroBuffer(phasorDataSin, static_cast<size_t>(fftResolution.x) * fftResolution.y * fftResolution.z);

	// Multiply phasorCos and phasorSin with the transformation matrix
	dim3 blockSize(16, 8, 8);
	dim3 gridSize(
		(dataResolution.z + blockSize.x - 1) / blockSize.x,
		(dataResolution.y + blockSize.y - 1) / blockSize.y,
		(dataResolution.x + blockSize.z - 1) / blockSize.z
	);

	multiplyTransformTranspose<<<gridSize, blockSize, 0, stream>>>(phasorCos, phasorSin, mtx, phasorDataCos, phasorDataSin, dataResolution, fftResolution);
	CudaHelper::synchronize("multiplyTransformTranspose");
}

void PhasorFields::convolveBackprojection(cufftComplex* phasorDataCos, cufftComplex* phasorDataSin, cufftComplex* psf, const glm::uvec3& dataResolution)
{
	glm::uvec3 fftResolution = dataResolution * 2u;

	cufftHandle planH;
	int rank = 3;  
	int n[3] = { static_cast<int>(fftResolution[0]),
				 static_cast<int>(fftResolution[1]),
				 static_cast<int>(fftResolution[2]) };

	CUFFT_CHECK(cufftPlanMany(&planH, rank, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(planH, phasorDataCos, phasorDataCos, CUFFT_FORWARD));

	CUFFT_CHECK(cufftPlanMany(&planH, rank, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(planH, phasorDataSin, phasorDataSin, CUFFT_FORWARD));

	dim3 blockSize(16, 8, 8);
	dim3 gridSize(
		(fftResolution.z + blockSize.x - 1) / blockSize.x,
		(fftResolution.y + blockSize.y - 1) / blockSize.y,
		(fftResolution.x + blockSize.z - 1) / blockSize.z
	);
	convolveBackprojectionKernel<<<gridSize, blockSize>>>(phasorDataCos, phasorDataSin, psf, fftResolution);

	CUFFT_CHECK(cufftPlanMany(&planH, rank, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(planH, phasorDataCos, phasorDataCos, CUFFT_INVERSE));

	CUFFT_CHECK(cufftPlanMany(&planH, rank, n,
		NULL, 1, 0,
		NULL, 1, 0,
		CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(planH, phasorDataSin, phasorDataSin, CUFFT_INVERSE));

	// IFFT requires normalization, but it also produces very small values, so we avoid this and produce valid results by normalizing later
	size_t fftSize = static_cast<size_t>(fftResolution.x) * fftResolution.y * fftResolution.z;
	normalizeIFFT<<<CudaHelper::getNumBlocks(fftSize, 512), 512>>>(phasorDataCos, fftSize, 1.0f / static_cast<float>(fftSize));
	normalizeIFFT<<<CudaHelper::getNumBlocks(fftSize, 512), 512>>>(phasorDataSin, fftSize, 1.0f / static_cast<float>(fftSize));
}

void PhasorFields::computeMagnitude(cufftComplex* phasorDataCos, cufftComplex* phasorDataSin, float* mtx, float* result1, float* result2, const glm::uvec3& dataResolution)
{
	glm::uvec3 fftResolution = dataResolution * 2u;
	dim3 blockSize(16, 8, 8);
	dim3 gridSize(
		(dataResolution.z + blockSize.x - 1) / blockSize.x,
		(dataResolution.y + blockSize.y - 1) / blockSize.y,
		(dataResolution.x + blockSize.z - 1) / blockSize.z
	);

	computePhasorFieldMagnitude<<<gridSize, blockSize>>>(phasorDataCos, phasorDataSin, result1, fftResolution, dataResolution);
	CudaHelper::synchronize("computePhasorFieldMagnitude");

	multiplyTransformTransposeInv<<<gridSize, blockSize>>>(result1, mtx, result2, dataResolution);
	CudaHelper::synchronize("multiplyTransformTranspose");
}

void PhasorFields::reconstructVolumeConfocal(float*& volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	const glm::uvec3 volumeResolution = glm::uvec3(_nlosData->_dims[0], _nlosData->_dims[1], _nlosData->_dims[2]);
	float tDistance = static_cast<float>(recInfo._numTimeBins) * recInfo._timeStep;
	float* intensityGpu = recBuffers._intensity;
	float* mtx = nullptr;
	float* phasorCos = nullptr, *phasorSin = nullptr;
	cufftComplex* phasorDataCos = nullptr, *phasorDataSin = nullptr;
	cufftComplex* psfKernel = nullptr;

	_perf.tic("Defined transform operators, psf and convolve data with virtual waves");

	// Define two cuda streams for asynchronous operations
	cudaStream_t stream1, stream2;
	CudaHelper::createStreams({ &stream1, &stream2 });

	// Forward transform operator (mtxi is simply the transpose of mtx) ---- STREAM 
	std::future<void> future = std::async(std::launch::async,
		defineTransformOperator,
		recInfo._numTimeBins,
		std::ref(mtx));

	// Waveconv & convolve data with the virtual waves
	float lambdaLimit = _nlosData->_wallWidth * 2.0f / static_cast<float>(volumeResolution.x - 1);
	float samplingCoeff = 2;				// Scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
	float virtualWavelength = samplingCoeff * (lambdaLimit * 2);	
	virtualWaveConvolution(intensityGpu, volumeResolution, recInfo._timeStep, virtualWavelength, 5, phasorCos, phasorSin, stream1);

	// Define the point spread function (PSF) kernel
	definePSFKernel(volumeResolution, glm::abs(_nlosData->_wallWidth / tDistance), psfKernel, stream2);

	// Transform data using previous operators
	transformData(volumeResolution, mtx, phasorCos, phasorSin, phasorDataCos, phasorDataSin, stream1);

	CudaHelper::waitFor({ &stream1, &stream2 });
	emptyCleanupQueue();

	_perf.toc();

	// Convolve with backprojection kernel, psf
	_perf.tic("Convolve PSF");
	convolveBackprojection(phasorDataCos, phasorDataSin, psfKernel, volumeResolution);
	_perf.toc();

	_perf.tic("Compute magnitude");
	computeMagnitude(phasorDataCos, phasorDataSin, mtx, phasorCos, intensityGpu, volumeResolution);
	_perf.toc();

	spdlog::info("Allocated memory: {} MB", CudaHelper::getAllocatedMemory() / static_cast<size_t>(1024 * 1024));

	CudaHelper::freeAsync(psfKernel, stream1);
	CudaHelper::freeAsync(mtx, stream1);
	CudaHelper::freeAsync(phasorCos, stream1);

	CudaHelper::freeAsync(phasorSin, stream2);
	CudaHelper::freeAsync(phasorDataCos, stream2);
	CudaHelper::freeAsync(phasorDataSin, stream2);

	CudaHelper::destroyStreams({ &stream1, &stream2 });
}

void PhasorFields::reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo)
{
}
