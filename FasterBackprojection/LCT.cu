#include "stdafx.h"
#include "LCT.h"

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
	ChronoUtilities::startTimer();

	const glm::uvec3 volumeResolution = glm::uvec3(_nlosData->_dims[0], _nlosData->_dims[1], _nlosData->_dims[2]);
	float tDistance = static_cast<float>(recInfo._numTimeBins) * recInfo._timeStep * LIGHT_SPEED;
	float* intensityGpu = recBuffers._intensity;

	// Define the point spread function (PSF) kernel
	cufftComplex* psfKernel = definePSFKernel(volumeResolution, _nlosData->_temporalWidth / tDistance);

	// Forward and backward transform operators
	float* mtx, *mtxi;
	defineTransformOperator(recInfo._numTimeBins, mtx, mtxi);
	float* transformedData = transformData(intensityGpu, volumeResolution, mtx);

	// FFT + PSF + IFFT
	multiplyKernel(transformedData, psfKernel, volumeResolution);

	// Inverse transform the data
	inverseTransformData(transformedData, intensityGpu, volumeResolution, mtxi);

	// Get maximum value along the temporal dimension
	float* maxZ = getMaximumZ(transformedData, volumeResolution);
	normalizeMatrix(maxZ, volumeResolution.x * volumeResolution.y);
	std::vector<float> maxZHost(static_cast<size_t>(volumeResolution.x) * volumeResolution.y);
	CudaHelper::downloadBufferGPU(maxZ, maxZHost.data(), static_cast<size_t>(volumeResolution.x) * volumeResolution.y, 0);

	TransientImage transientImage(volumeResolution.x, volumeResolution.y);
	transientImage.save(
		"output/transient.png", maxZHost.data(),
		glm::uvec2(volumeResolution.x, volumeResolution.y),
		1, 0, false, false
	);

	CudaHelper::free(maxZ);
	CudaHelper::free(transformedData);
	CudaHelper::free(psfKernel);
	CudaHelper::free(mtx);
	CudaHelper::free(mtxi);
}

void LCT::reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo)
{
}

cufftComplex* LCT::definePSFKernel(const glm::uvec3& dataResolution, float slope)
{
	glm::uvec3 totalRes = dataResolution * 2u; // Assuming the PSF kernel is twice the resolution in each dimension
    glm::uint size = totalRes.x * totalRes.y * totalRes.z;

	// Gpu-side memory allocation
	float* psf = nullptr;
	cufftComplex* rolledPsf = nullptr;

	CudaHelper::initializeBufferGPU(psf, size);
	CudaHelper::initializeBufferGPU(rolledPsf, size);
	
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
		(totalRes.x + blockSize.x - 1) / blockSize.x,
        (totalRes.y + blockSize.y - 1) / blockSize.y,
        (totalRes.z + blockSize.z - 1) / blockSize.z
	);

	// RSD
	{
		computePSFKernel<<<gridSize, blockSize>>>(psf, totalRes, slope);
		CudaHelper::synchronize("computePSFKernel");
	}

	// Find minimum along z-axis and binarize (only 1 value per xy, or a few at most)
	{
		dim3 blockSize2D(16, 16);
		dim3 gridSize2D(
			(totalRes.x + blockSize2D.x - 1) / blockSize2D.x,
			(totalRes.y + blockSize2D.y - 1) / blockSize2D.y
		);

		findMinimumBinarize<<<gridSize2D, blockSize2D>>>(psf, totalRes);
		CudaHelper::synchronize("findMinimumBinarize");
	}

	// Normalization according to center 
	{
		const glm::uint sumBaseIndex = dataResolution.x * totalRes.y * totalRes.z + dataResolution.y * totalRes.z;
		size_t tempStorageBytes = 0;
		void* tempStorage = nullptr;
		float* sumGpu = nullptr;
		CudaHelper::initializeZeroBufferGPU(sumGpu, 1);

		cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, psf + sumBaseIndex, sumGpu, totalRes.z);
		cudaMalloc(&tempStorage, tempStorageBytes);
		cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, psf + sumBaseIndex, sumGpu, totalRes.z);

		normalizePSF<<<gridSize, blockSize>>>(psf, sumGpu, totalRes);
		CudaHelper::synchronize("normalizePSF");

		CudaHelper::free(sumGpu);
		CudaHelper::free(tempStorage);
	}

	// L2 normalization
	{
		size_t tempStorageBytes = 0;
		void* tempStorage = nullptr;
		float* sqrtNormGpu = nullptr;
		CudaHelper::initializeZeroBufferGPU(sqrtNormGpu, 1);

		cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, psf, sqrtNormGpu, size);
		cudaMalloc(&tempStorage, tempStorageBytes);
		cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, psf, sqrtNormGpu, size);

		float sqrtNorm;
		CudaHelper::downloadBufferGPU(sqrtNormGpu, &sqrtNorm, 1, 0);
		sqrtNorm = sqrtf(sqrtNorm);

		l2NormPSF<<<gridSize, blockSize>>>(psf, sqrtNorm, totalRes);
		CudaHelper::synchronize("l2NormPSF");

		CudaHelper::free(sqrtNormGpu);
		CudaHelper::free(tempStorage);
	}

    rollPSF<<<gridSize, blockSize>>>(psf, rolledPsf, dataResolution, totalRes);
	CudaHelper::synchronize("rollPSF");

	// Fourier transform the PSF kernel
	{
		cufftHandle planPSF;

		int rank = 3;  // 3D FFT
		int n[3] = { static_cast<int>(totalRes.x),
					 static_cast<int>(totalRes.y),
					 static_cast<int>(totalRes.z) };

		CUFFT_CHECK(cufftPlanMany(&planPSF, rank, n,
			NULL, 1, 0,			// idist and odist does not matter when the number of batches is 1
			NULL, 1, 0,  
			CUFFT_C2C, 1));       
		CUFFT_CHECK(cufftExecC2C(planPSF, rolledPsf, rolledPsf, CUFFT_FORWARD));
		CUFFT_CHECK(cufftDestroy(planPSF));
	}

	// Wiener filter
	{
		wienerFilterPsf<<<gridSize, blockSize>>>(rolledPsf, totalRes, 8e-1);
		CudaHelper::synchronize("wienerFilterPsf");
	}

	CudaHelper::free(psf);

	return rolledPsf;
}

void LCT::defineTransformOperator(glm::uint M, float*& d_mtx, float*& d_inverseMtx)
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

	// Step 2: Transpose (now column-major)
	SparseMatrixF_ColMajor mtxi = mtx.transpose(); 

	// Step 3: Hierarchical downsampling
	glm::uint K = static_cast<int>(std::round(std::log2(M)));

	for (glm::uint k = 0; k < K; ++k)
	{
		// ------------------
		// Downsample rows
		Index newRows = mtx.rows() / 2;
		SparseMatrixF_RowMajor mtx_new(newRows, mtx.cols());

		// Thread-local storage for triplets to avoid critical section contention
		std::vector<std::vector<TripletF>> threadAvgTriplets(omp_get_max_threads());

		#pragma omp parallel
		{
			int threadID = omp_get_thread_num();
			threadAvgTriplets[threadID].reserve(mtx.nonZeros() / (2 * omp_get_num_threads()) + 10);

			#pragma omp for nowait
			for (int i = 0; i < newRows; ++i)
			{
				int row1 = 2 * i;
				int row2 = 2 * i + 1;

				std::map<Index, float> combined;
				for (SparseMatrixF_RowMajor::InnerIterator it1(mtx, row1); it1; ++it1)
					combined[it1.col()] += 0.5f * it1.value();
				for (SparseMatrixF_RowMajor::InnerIterator it2(mtx, row2); it2; ++it2)
					combined[it2.col()] += 0.5f * it2.value();

				for (const auto& entry : combined)
					threadAvgTriplets[threadID].emplace_back(i, entry.first, entry.second);
			}
		}

		// Merge results from all threads
		std::vector<TripletF> combinedAvgTriplets;
		for (const auto& local_vec : threadAvgTriplets)
			combinedAvgTriplets.insert(combinedAvgTriplets.end(), local_vec.begin(), local_vec.end());

		mtx_new.setFromTriplets(combinedAvgTriplets.begin(), combinedAvgTriplets.end());
		mtx_new.makeCompressed();
		mtx = mtx_new;

		// ------------------
		// Downsample columns
		Index newCols = mtxi.cols() / 2;
		SparseMatrixF_ColMajor mtxi_new(mtxi.rows(), newCols);

		std::vector<std::vector<TripletF>> threadAvgTriplets_i(omp_get_max_threads());

		#pragma omp parallel
		{
			int threadID = omp_get_thread_num();
			threadAvgTriplets_i[threadID].reserve(mtxi.nonZeros() / (2 * omp_get_num_threads()) + 10);

			#pragma omp for nowait
			for (int j = 0; j < newCols; ++j)
			{
				int col1 = 2 * j;
				int col2 = 2 * j + 1;

				std::map<Index, float> combined;
				for (SparseMatrixF_ColMajor::InnerIterator it1(mtxi, col1); it1; ++it1)
					combined[it1.row()] += 0.5f * it1.value();
				for (SparseMatrixF_ColMajor::InnerIterator it2(mtxi, col2); it2; ++it2)
					combined[it2.row()] += 0.5f * it2.value();

				for (const auto& entry : combined)
					threadAvgTriplets_i[threadID].emplace_back(entry.first, j, entry.second);
			}
		}

		// Combine results from all threads
		std::vector<TripletF> combinedAvgTriplets_col;
		for (const auto& local_vec : threadAvgTriplets_i)
			combinedAvgTriplets_col.insert(combinedAvgTriplets_col.end(), local_vec.begin(), local_vec.end());

		mtxi_new.setFromTriplets(combinedAvgTriplets_col.begin(), combinedAvgTriplets_col.end());
		mtxi_new.makeCompressed();
		mtxi = mtxi_new;
	}

	std::vector<float> mtxHost(M2);

	// Iterate over the sparse matrix and copy to dense
	#pragma omp parallel for
	for (int k = 0; k < mtx.outerSize(); ++k)
		for (SparseMatrixF_RowMajor::InnerIterator it(mtx, k); it; ++it) 
			mtxHost[it.row() * M + it.col()] = it.value();

	CudaHelper::initializeBufferGPU(d_mtx, mtxHost.size(), mtxHost.data());

	// Prepare inverse matrix(transpose of mtx)
	#pragma omp parallel for
	for (int k = 0; k < mtxi.outerSize(); ++k)
		for (SparseMatrixF_ColMajor::InnerIterator it(mtxi, k); it; ++it) 
			mtxHost[it.row() * M + it.col()] = it.value();

	CudaHelper::initializeBufferGPU(d_inverseMtx, mtxHost.size(), mtxHost.data());
}

void LCT::multiplyKernel(float* volumeGpu, const cufftComplex* inversePSF, const glm::uvec3& dataResolution)
{
	glm::uint nt = dataResolution.z, nt_pad = nt * 2;

	//
	glm::uvec3 newDims = dataResolution * glm::uvec3(2);
	size_t newDimProduct = static_cast<size_t>(newDims.x) * newDims.y * newDims.z;

	// Transfer H to a padded H
	cufftComplex* d_H = nullptr;
	CudaHelper::initializeZeroBufferGPU(d_H, newDimProduct);

	dim3 blockSize(8, 8, 8);
	dim3 gridSize(
		(dataResolution.x + blockSize.x - 1) / blockSize.x,
		(dataResolution.y + blockSize.y - 1) / blockSize.y,
		(dataResolution.z + blockSize.z - 1) / blockSize.z
	);
	padIntensityFFT<<<gridSize, blockSize>>>(volumeGpu, d_H, dataResolution, newDims);
	CudaHelper::synchronize("padIntensityFFT");

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
	multiplyPSF<<<CudaHelper::getNumBlocks(newDimProduct, 512), 512>>>(d_H, inversePSF, newDimProduct);
	CudaHelper::synchronize("multiplyHK");

	// Inverse FFT
	CUFFT_CHECK(cufftExecC2C(planH, d_H, d_H, CUFFT_INVERSE));
	CUFFT_CHECK(cufftDestroy(planH));

	// IFFT requires normalization, but it also produces very small values, so we avoid this and produce valid results by normalizing later
	//normalizeIFFT<<<CudaHelper::getNumBlocks(newDimProduct, 512), 512>>>(d_H, newDimProduct, 1.0f / newDimProduct);

	//
	unpadIntensityFFT<<<gridSize, blockSize>>>(volumeGpu, d_H, dataResolution, newDims);
	CudaHelper::synchronize("unpadIntensityFFT");
}

float* LCT::transformData(float* volumeGpu, const glm::uvec3& dataResolution, float* mtx)
{
	dim3 blockSize3D(8, 8, 8);
	dim3 gridSize3D(
		(dataResolution.x + blockSize3D.x - 1) / blockSize3D.x,
		(dataResolution.y + blockSize3D.y - 1) / blockSize3D.y,
		(dataResolution.z + blockSize3D.z - 1) / blockSize3D.z
	);

	// Scale the intensity values according to the material type (diffuse or not)
	scaleIntensity<<<gridSize3D, blockSize3D>>>(volumeGpu, dataResolution, false);
	CudaHelper::synchronize("scaleIntensity");

	// ---- Multiply by the transform matrix (time dimension * time dimension)
	float* multResult = nullptr;
	CudaHelper::initializeZeroBufferGPU(multResult, static_cast<size_t>(dataResolution.x) * dataResolution.y * dataResolution.z);

	// Multiply intensity by the transform matrix
	multiplyTransformTranspose<<<gridSize3D, blockSize3D>>>(volumeGpu, mtx, multResult, dataResolution);
	CudaHelper::synchronize("multiplyTransformTranspose");

	return multResult;
}

void LCT::inverseTransformData(float* volumeGpu, float* multResult, const glm::uvec3& dataResolution, float*& inverseMtx)
{
	dim3 blockSize3D(8, 8, 8);
	dim3 gridSize3D(
		(dataResolution.x + blockSize3D.x - 1) / blockSize3D.x,
		(dataResolution.y + blockSize3D.y - 1) / blockSize3D.y,
		(dataResolution.z + blockSize3D.z - 1) / blockSize3D.z
	);

	multiplyTransformTranspose<<<gridSize3D, blockSize3D>>>(volumeGpu, inverseMtx, multResult, dataResolution);
	CudaHelper::synchronize("multiplyTransformTranspose");
}

float* LCT::getMaximumZ(float* volumeGpu, const glm::uvec3& dataResolution)
{
	float* maxZ = nullptr;
	CudaHelper::initializeBufferGPU(maxZ, dataResolution.x * dataResolution.y);

	dim3 blockSize(16, 16);
	dim3 gridSize(
		(dataResolution.x + blockSize.x - 1) / blockSize.x,
		(dataResolution.y + blockSize.y - 1) / blockSize.y
	);
	composeImage<<<gridSize, blockSize>>>(volumeGpu, maxZ, dataResolution);
	CudaHelper::synchronize("formImage");

	std::vector<float> maxZHost(dataResolution.x * dataResolution.y);
	CudaHelper::downloadBufferGPU(maxZ, maxZHost.data(), dataResolution.x * dataResolution.y, 0);

	return maxZ;
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

	const glm::uvec3 voxelResolution = recInfo._voxelResolution;
	const glm::uint numVoxels = voxelResolution.x * voxelResolution.y * voxelResolution.z;
	float* volumeGpu = nullptr;
	CudaHelper::initializeZeroBufferGPU(volumeGpu, numVoxels);

	//if (transientParams._useFourierFilter)
	//	filter_H_cuda(recBuffers._intensity, recInfo._timeStep * 10.0f, .0f);

	if (recInfo._captureSystem == CaptureSystem::Confocal)
		reconstructVolumeConfocal(volumeGpu, recInfo, recBuffers);
	//else if (recInfo._captureSystem == CaptureSystem::Exhaustive)
	//	reconstructVolumeExhaustive(volumeGpu, recInfo);

	std::cout << "Reconstruction finished in " << getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	// Save volume & free resources
	saveReconstructedAABB(transientParams._outputFolder + "aabb.cube", volumeGpu, numVoxels);
	CudaHelper::free(volumeGpu);
}
