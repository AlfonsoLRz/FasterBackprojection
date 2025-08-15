#pragma once

#include <cufft.h>

#include "FastReconstructionAlgorithm.h"
#include "data/SceneParameters.h"

class ViewportSurface;

class FastRSDReconstruction: public rtnlos::FastReconstructionAlgorithm
{
	// Device pointers
	cufftComplex* 				_rsd;
	cufftComplex*				_imageConvolution;
	float*						_cubeImages;
	float*						_dWeights;
	cufftHandle					_fftPlan2D;

	glm::uint					_blockSize1D, _gridSize1D;
	dim3						_blockSize2D_freq, _blockSize2D_depth, _blockSize2D_pix;
	dim3						_gridSize2D_freq, _gridSize2D_depth, _gridSize2D_pix;

public:
	FastRSDReconstruction();
	~FastRSDReconstruction() override;

	// Allocates all necessary GPU data, precalculates RSD. Basically sets everything up.
	void precalculate() override; 

	// Call after each time full set of images has been added
	void reconstructImage(ViewportSurface* viewportSurface) override;

private:
	void RSDKernelConvolution(cufftComplex* dKernel, cufftHandle fftPlan, const float lambda, const float omega, const float depth, const float t, cudaStream_t cudaStream) const;
};