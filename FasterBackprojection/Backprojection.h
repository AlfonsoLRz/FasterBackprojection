#pragma once

#include "Reconstruction.h"

class ComputeShader;

class Backprojection : public Reconstruction
{
protected:
	ComputeShader* _backprojectionConfocalVoxelShader = nullptr;

protected:
	static void buildAliasTables(const std::vector<float>& cdf, std::vector<glm::uint>& aliasTable, std::vector<float>& probTable);

protected:
	void reconstructDepthConfocal(const ReconstructionInfo& recInfo, std::vector<float>& reconstructionDepths);
	void reconstructDepthExhaustive(const ReconstructionInfo& recInfo, std::vector<float>& reconstructionDepths);

	void reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo);
	void reconstructVolumeConfocalGLSL(float* volume, const ReconstructionInfo& recInfo);

	void reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo);

public:
	void reconstructAABBConfocalMIS(float* volume, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers);

public:
	void reconstructDepths(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams, const std::vector<float>& depths) override;

	void reconstructVolume(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams) override;
};

