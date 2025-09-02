#pragma once

#include <complex>
#include <cufft.h>

#include "ChronoUtilities.h"
#include "CudaPerf.h"
#include "GpuStructs.cuh"
#include "NLosData.h"
#include "PostprocessingFilters.h"
#include "TransientParameters.h"

class Camera;
class SceneContent;

class Reconstruction
{
protected:
	NLosData*							_nlosData = nullptr;
	CudaPerf							_perf;
	std::vector<std::function<void()>>	_cleanupQueue;

	// Strategy pattern for post-processing filters
	static const PostprocessingFilters* _postprocessingFilters[PostprocessingFilterType::NUM_POSTPROCESSING_FILTERS];

protected:
	// Fourier related functions
	static std::vector<float> linearSpace(float minValue, float maxValue, int n);
	void padIntensity(float* volumeGpu, cufftComplex*& paddedIntensity, size_t padding, const std::string& mode) const;
	void filter_H_cuda(float* intensityGpu, float wl_mean, float wl_sigma = .0f, const std::string& border = "zero");

	// Pre-processing functions
	void compensateLaserCosDistance(const TransientParameters& transientParams, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers);
	void normalizeMatrix(float* v, glm::uint size);

	// Save functions
	void saveMaxImage(const std::string& filename, const float* volumeGpu, const glm::uvec3& volumeResolution, bool flip = true);
	static bool saveReconstructedAABB(const std::string& filename, float* voxels, glm::uint numVoxels);

	//
	void emptyCleanupQueue();

public:
	virtual ~Reconstruction() = default;

	// This is not implemented in most cases. Mainly useful for backprojection whether depth is known
	virtual void reconstructDepths(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams, const std::vector<float>& depths) = 0;

	virtual void reconstructVolume(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams) = 0;
};

