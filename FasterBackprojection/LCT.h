#pragma once

#include "GpuStructs.cuh"
#include "TransientParameters.h"

class LCT
{
	LCT(ReconstructionInfo);
	virtual ~LCT();

	void reconstructAABB(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers, const TransientParameters& transientParameters);
};

