#pragma once

#include "GpuStructs.cuh"
#include "Reconstruction.h"

class LCT : public Reconstruction
{
protected:
	void reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers);
	static void reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo);

	static cufftComplex* definePSFKernel(const glm::uvec3& dataResolution, float slope);
	static void defineTransformOperator(glm::uint numTimeBins, float*& d_mtx, float*& d_inverseMtx);
	static cufftComplex* prepareIntensity(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers);
	void multiplyKernel(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers);
	cufftComplex* transformData(float* volumeGpu, const glm::uvec3& dataResolution, float*& mtx, float*& inverseMtx);

public:
	void reconstructDepths(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams, const std::vector<float>& depths) override;

	void reconstructVolume(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams) override;
};

