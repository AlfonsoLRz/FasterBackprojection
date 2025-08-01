// ReSharper disable CppExpressionWithoutSideEffects
// ReSharper disable CppClangTidyClangDiagnosticShadow
#include "stdafx.h"
#include "Laser.cuh"

#include "Backprojection.h"
#include "ChronoUtilities.h"
#include "GpuStructs.cuh"
#include "CudaHelper.h"
#include "FK.h"
#include "LCT.h"
#include "NLosData.h"
#include "PhasorFields.h"

//

Reconstruction* Laser::_reconstruction[ReconstructionType::NUM_RECONSTRUCTION_TYPES] = {
	new Backprojection(),
	new LCT(),
	new FK(),
	new PhasorFields()
};

void Laser::reconstruct(NLosData* nlosData, const TransientParameters& transientParams)
{
	ReconstructionInfo recInfo;
	ReconstructionBuffers recBuffers;

	// Transfer data to GPU
	nlosData->downsampleTime(4);
	//nlosData->downsampleSpace(2);
	nlosData->toGpu(recInfo, recBuffers, transientParams);

	CudaHelper::checkError(cudaMemcpyToSymbol(rtRecInfo, &recInfo, sizeof(ReconstructionInfo)));
	CudaHelper::checkError(cudaMemcpyToSymbol(laserTargets, &recBuffers._laserTargets, sizeof(glm::vec3*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(laserTargetsNormals, &recBuffers._laserTargetsNormals, sizeof(glm::vec3*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(sensorTargets, &recBuffers._sensorTargets, sizeof(glm::vec3*)));
	CudaHelper::checkError(cudaMemcpyToSymbol(intensityCube, &recBuffers._intensity, sizeof(float*)));

	std::cout << "Reconstructing shape...\n";

	_reconstruction[transientParams._reconstructionType]->reconstructVolume(nlosData, recInfo, recBuffers, transientParams);

	CudaHelper::free(recBuffers._laserTargets);
	CudaHelper::free(recBuffers._laserTargetsNormals);
	CudaHelper::free(recBuffers._sensorTargets);
	CudaHelper::free(recBuffers._intensity);
}