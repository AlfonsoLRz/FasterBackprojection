#pragma once

#include "GpuStructs.cuh"
#include "Reconstruction.h"

class LCT : public Reconstruction
{
protected:
	void reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers);
	static void reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo);

	cufftComplex* definePSFKernel(const glm::uvec3& dataResolution, float slope, cufftHandle fftPlan, cudaStream_t stream);
	static void defineTransformOperator(glm::uint M, float*& d_mtx, cudaStream_t stream);

	void multiplyKernel(float* volumeGpu, const cufftComplex* inversePSF, const glm::uvec3& dataResolution, cufftHandle fftPlan);

	static float* transformData(float* volumeGpu, const glm::uvec3& dataResolution, const float* mtx, cudaStream_t stream);
	static void inverseTransformData(const float* volumeGpu, float* multResult, const glm::uvec3& dataResolution, float*& inverseMtx);

public:
	void reconstructDepths(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams, const std::vector<float>& depths) override;

	void reconstructVolume(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams) override;
};

