#pragma once

#include "Reconstruction.h"

class PhasorFields : public Reconstruction
{
protected:
	cufftComplex* definePSFKernel(const glm::uvec3& dataResolution, float slope, cudaStream_t stream);
	static void defineTransformOperator(glm::uint M, float*& d_mtx);
	cufftComplex* waveconv(
		float* data, const glm::uvec3& dataResolution, float deltaDistance, float virtualWavelength, float cycles, 
		float*& phasorCos, float*& phasorSin);

	static void transformData(
		const glm::uvec3& dataResolution, 
		const float* mtx, 
		float* phasorCos, float* phasorSin, 
		cufftComplex*& phasorDataCos, cufftComplex*& phasorDataSin);

	static void convolveBackprojection(
		cufftComplex* phasorDataCos, cufftComplex* phasorDataSin,
		cufftComplex* psf,
		const glm::uvec3& dataResolution);

	void reconstructVolumeConfocal(float*& volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers);
	static void reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo);

public:
	void reconstructDepths(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams, const std::vector<float>& depths) override;

	void reconstructVolume(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams) override;
};

