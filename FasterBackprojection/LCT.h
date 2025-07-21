#pragma once

#include "GpuStructs.cuh"
#include "Reconstruction.h"

class LCT : public Reconstruction
{
protected:
	std::vector<float*>			_deleteFloatQueue;
	std::vector<cufftComplex*>	_deleteComplexQueue;
	std::vector<cufftHandle>	_deleteCufftHandles;

protected:
	void reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers) const;
	static void reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo);

	static cufftComplex* definePSFKernel(const glm::uvec3& dataResolution, float slope, cudaStream_t stream);
	static void defineTransformOperator(glm::uint M, float*& d_mtx, float*& d_inverseMtx);

	static void multiplyKernel(float* volumeGpu, const cufftComplex* inversePSF, const glm::uvec3& dataResolution);

	static float* transformData(float* volumeGpu, const glm::uvec3& dataResolution, float* mtx, cudaStream_t stream);
	static void inverseTransformData(float* volumeGpu, float* multResult, const glm::uvec3& dataResolution, float*& inverseMtx);

public:
	void reconstructDepths(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams, const std::vector<float>& depths) override;

	void reconstructVolume(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams) override;
};

