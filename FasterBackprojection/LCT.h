#pragma once

#include "GpuStructs.cuh"
#include "Reconstruction.h"

class LCT : public Reconstruction
{
protected:
	void reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers);
	static void reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo);

	static cufftComplex* definePSFKernel(const glm::uvec3& dataResolution, float slope);
	static void defineTransformOperator(glm::uint M, float*& d_mtx, float*& d_inverseMtx);

	static void multiplyKernel(float* volumeGpu, const cufftComplex* inversePSF, const glm::uvec3& dataResolution);

	static float* transformData(float* volumeGpu, const glm::uvec3& dataResolution, float* mtx);
	static void inverseTransformData(float* volumeGpu, float* multResult, const glm::uvec3& dataResolution, float*& inverseMtx);

	static float* getMaximumZ(float* volumeGpu, const glm::uvec3& dataResolution);

public:
	void reconstructDepths(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams, const std::vector<float>& depths) override;

	void reconstructVolume(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams) override;
};

