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
#include "fourier.cuh"
#include "TransientParameters.h"

//

void LCT::reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	ChronoUtilities::startTimer();

	const glm::uvec3 volumeResolution = glm::uvec3(_nlosData->_dims[0], _nlosData->_dims[1], _nlosData->_dims[2]);
	float tDistance = static_cast<float>(recInfo._numTimeBins) * recInfo._timeStep * LIGHT_SPEED;

	cufftComplex* psfKernel = definePSFKernel(volumeResolution, _nlosData->_temporalWidth / tDistance);

	float* mtx, *mtxi;
	defineTransformOperator(recInfo._numTimeBins, mtx, mtxi);
	//multiplyKernel(recInfo, recBuffers);
	transformData(recBuffers._intensity, volumeResolution, mtx, mtxi);

	//backprojectConfocalVoxel << <gridSize, blockSize >> > (volume, sliceSize);
	//CudaHelper::synchronize("backprojectConfocalVoxel");
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

	std::vector<cufftComplex> psfHost(size);
	
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

	CudaHelper::downloadBufferGPU(rolledPsf, psfHost.data(), totalRes.z, 0);

	// Fourier transform the PSF kernel
	{
		glm::uint dimProduct = totalRes.x * totalRes.y * totalRes.z;
		cufftHandle planPSF;

		int rank = 3;  // 3D FFT
		int n[3] = { static_cast<int>(totalRes.x),
					 static_cast<int>(totalRes.y),
					 static_cast<int>(totalRes.z) };

		CUFFT_CHECK(cufftPlanMany(&planPSF, rank, n,
			NULL, 1, dimProduct,  
			NULL, 1, dimProduct,  
			CUFFT_C2C, 1));       
		CUFFT_CHECK(cufftExecC2C(planPSF, rolledPsf, rolledPsf, CUFFT_FORWARD));

		CUFFT_CHECK(cufftDestroy(planPSF));
	}

	// Wiener filter
	{
		wienerFilterPsf<<<gridSize, blockSize>>>(rolledPsf, totalRes, 8e-1);
		CudaHelper::synchronize("wienerFilterPsf");

		//CudaHelper::downloadBufferGPU(rolledPsf, psfHost.data(), size, 0);
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

		// Debug output
		//std::cout << "k=" << k
		//	<< ", mtx rows: " << mtx.rows() << " nnz: " << mtx.nonZeros()
		//	<< ", mtxi cols: " << mtxi.cols() << " nnz: " << mtxi.nonZeros()
		//	<< std::endl;
	}

	std::vector<float> mtxHost(M2);

	// Iterate over the sparse matrix and copy to dense
	for (int k = 0; k < mtx.outerSize(); ++k)
		for (SparseMatrixF_RowMajor::InnerIterator it(mtx, k); it; ++it) 
			mtxHost[it.row() * M + it.col()] = it.value();

	CudaHelper::initializeBufferGPU(d_mtx, mtxHost.size(), mtxHost.data());

	// Prepare inverse matrix(transpose of mtx)
	for (int k = 0; k < mtxi.outerSize(); ++k)
		for (SparseMatrixF_ColMajor::InnerIterator it(mtxi, k); it; ++it) 
			mtxHost[it.row() * M + it.col()] = it.value();

	CudaHelper::initializeBufferGPU(d_inverseMtx, mtxHost.size(), mtxHost.data());
}

cufftComplex* LCT::prepareIntensity(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	return nullptr;
}

void LCT::multiplyKernel(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers)
{
	glm::uint nt = recInfo._numTimeBins, nt_pad = nt * 2;

	//
	glm::uint newDimProduct = 1, dimProduct = 1;
	std::vector<size_t> newDims = _nlosData->_dims;
	for (size_t& dim: newDims)
		dim *= 2;	

	for (size_t dim : _nlosData->_dims)
		dimProduct *= dim;
	for (size_t dim : newDims)
		newDimProduct *= dim;

	//
	const glm::uint sliceSize = dimProduct / nt;
	const glm::uint newSliceSize = newDimProduct / nt_pad;

	// Transfer H to a padded H
	cufftComplex* d_H = nullptr;
	CudaHelper::initializeBufferGPU(d_H, newDimProduct);

	dim3 blockSize(8, 8);
	dim3 gridSize(
		(sliceSize + blockSize.x - 1) / blockSize.x,
		(nt + blockSize.y - 1) / blockSize.y
	);

	std::vector<cufftComplex> H_host(newDimProduct, cufftComplex{ .0f, .0f });

	padIntensityFFT<<<gridSize, blockSize>>>(recBuffers._intensity, d_H, sliceSize, nt, nt_pad);
	CudaHelper::synchronize("padIntensityFFT");

	CudaHelper::downloadBufferGPU(d_H, H_host.data(), newDimProduct, 0);

	//
	cufftHandle planH;
	int n[1] = { static_cast<int>(nt_pad) };
	int rank = 1;
	int inStride = 1, outStride = 1;														// Stride between elements in time dimension (contiguous)
	int inDistance = static_cast<int>(nt_pad), outDistance = static_cast<int>(nt_pad);		// Distance between consecutive batches in output

	// Create 1D FFT plan for batches
	CUFFT_CHECK(cufftPlanMany(&planH, rank, n, NULL, inStride, inDistance, NULL, outStride, outDistance, CUFFT_C2C, newSliceSize));
	CUFFT_CHECK(cufftExecC2C(planH, d_H, d_H, CUFFT_FORWARD));

	// 
	CUFFT_CHECK(cufftDestroy(planH));
}

cufftComplex* LCT::transformData(float* volumeGpu, const glm::uvec3& dataResolution, float*& mtx, float*& inverseMtx)
{
	dim3 blockSize3D(8, 8, 8);
	dim3 gridSize3D(
		(dataResolution.x + blockSize3D.x - 1) / blockSize3D.x,
		(dataResolution.y + blockSize3D.y - 1) / blockSize3D.y,
		(dataResolution.z + blockSize3D.z - 1) / blockSize3D.z
	);

	scaleIntensity<<<gridSize3D, blockSize3D >>>(volumeGpu, dataResolution, true);
	CudaHelper::synchronize("scaleIntensity");

	std::vector<float> mtxHost(static_cast<size_t>(dataResolution.x) * dataResolution.y * dataResolution.z);
	CudaHelper::downloadBufferGPU(volumeGpu, mtxHost.data(), dataResolution.z, 0);

	// ---- Multiply by the transform matrix (time dimension * time dimension)
	const glm::uint XY = dataResolution.x * dataResolution.y, Z = dataResolution.z;
	float* multResult = nullptr;
	CudaHelper::initializeZeroBufferGPU(multResult, XY * Z);

	dim3 blockSize(32, 16);
	dim3 gridSize(
		(XY + blockSize.x - 1) / blockSize.x,
		(Z + blockSize.y - 1) / blockSize.y
	);
	multiplyTransformTranspose<<<gridSize, blockSize>>>(volumeGpu, mtx, multResult, XY, Z);
	CudaHelper::synchronize("multiplyTransformTranspose");

	std::vector<float> multHost(XY * Z);
	CudaHelper::downloadBufferGPU(multResult, multHost.data(), XY * Z, 0);

	return nullptr;
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

	// Post-process the activation matrix
	_postprocessingFilters[transientParams._postprocessingFilterType]->compute(volumeGpu, voxelResolution, transientParams);
	normalizeMatrix(volumeGpu, numVoxels);

	std::cout << "Reconstruction finished in " << getElapsedTime(ChronoUtilities::MILLISECONDS) << " milliseconds.\n";

	// Save volume & free resources
	saveReconstructedAABB(transientParams._outputFolder + "aabb.cube", volumeGpu, numVoxels);
	CudaHelper::free(volumeGpu);
}
