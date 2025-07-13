#pragma once

#include "Reconstruction.h"
#include "TransientParameters.h"

class NLosData;

class Laser
{
protected:
	static Reconstruction* _reconstruction[ReconstructionType::NUM_RECONSTRUCTION_TYPES];

public:
	static void reconstruct(NLosData* nlosData, const TransientParameters& transientParams);
};
