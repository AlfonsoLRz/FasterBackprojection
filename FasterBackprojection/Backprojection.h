#pragma once

#include "Reconstruction.h"

class Backprojection : public Reconstruction
{
protected:
	static void buildAliasTables(const std::vector<float>& cdf, std::vector<glm::uint>& aliasTable, std::vector<float>& probTable);

protected:
	static void reconstructDepthConfocal(const ReconstructionInfo& recInfo, std::vector<float>& reconstructionDepths);
	static void reconstructDepthExhaustive(const ReconstructionInfo& recInfo, std::vector<float>& reconstructionDepths);

	static void reconstructVolumeConfocal(float* volume, const ReconstructionInfo& recInfo);
	static void reconstructVolumeExhaustive(float* volume, const ReconstructionInfo& recInfo);

public:
	void reconstructAABBConfocalMIS(float* volume, const ReconstructionInfo& recInfo) const;

public:
	void reconstructDepths(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams, const std::vector<float>& depths) override;

	void reconstructVolume(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams) override;
};

