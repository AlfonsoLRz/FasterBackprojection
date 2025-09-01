#pragma once

#include <cufft.h>

#include "FastReconstructionAlgorithm.h"
#include "data/SceneParameters.h"

class ViewportSurface;

class FastLCTReconstruction: public rtnlos::FastReconstructionAlgorithm
{
	// Device pointers
	//cufftComplex* 				_rsd;
	//cufftComplex*				_imageConvolution;
	//float*						_cubeImages;
	//float*						_dWeights;
	//cufftHandle					_fftPlan2D;

	//glm::uint					_blockSize1D, _gridSize1D;
	//dim3						_blockSize2D_freq, _blockSize2D_depth, _blockSize2D_pix;
	//dim3						_gridSize2D_freq, _gridSize2D_depth, _gridSize2D_pix;

	glm::uvec3 				_volumeResolution;

	// Device pointers
	cufftComplex*			_psfKernel;
	float*					_mtx;
	cufftComplex*			_multResult;

	cufftHandle				_fftPlan;

	glm::uint				_blockSize1D, _gridSize1D;
	dim3					_blockSize3D, _gridSize3D;

public:
	FastLCTReconstruction();
	~FastLCTReconstruction() override;

	void destroyResources() override;

	// Allocates all necessary GPU data, precalculates RSD. Basically sets everything up.
	void precalculate() override; 

	// Call after each time full set of images has been added
	void reconstructImage(ViewportSurface* viewportSurface) override;

private:
	cufftComplex* definePSFKernel(float slope, cudaStream_t stream);
	static void defineTransformOperator(glm::uint M, float*& d_mtx, cudaStream_t stream);
	void multiplyKernel();
	void transformData(const float* mtx);
	void inverseTransformData();

};
