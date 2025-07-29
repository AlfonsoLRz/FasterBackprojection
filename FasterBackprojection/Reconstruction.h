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
	template <typename T>
	class DeleteQueue {
	public:
		void add(T* ptr)
		{
			queue.emplace_back(ptr);
		}
		void clear()
		{
			for (auto ptr : queue) delete ptr;
			queue.clear();
		}

	private:
		std::vector<T*> queue;
	};

protected:
	NLosData* _nlosData = nullptr;
	CudaPerf  _perf;

	static const PostprocessingFilters* _postprocessingFilters[PostprocessingFilterType::NUM_POSTPROCESSING_FILTERS];

protected:
	// Fourier related functions
	static std::vector<float> linearSpace(float minValue, float maxValue, int n);
	void padIntensity(float* volumeGpu, cufftComplex*& paddedIntensity, size_t padding, const std::string& mode) const;
	void filter_H_cuda(float* intensityGpu, float wl_mean, float wl_sigma = .0f, const std::string& border = "zero");

	// Pre-processing functions
	void compensateLaserCosDistance(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers);
	void normalizeMatrix(float* v, glm::uint size);

	void saveMaxImage(const std::string& filename, const float* volumeGpu, const glm::uvec3& volumeResolution, bool flip = true);
	static bool saveReconstructedAABB(const std::string& filename, float* voxels, glm::uint numVoxels);

public:
	virtual ~Reconstruction() = default;

	virtual void reconstructDepths(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams, const std::vector<float>& depths) = 0;

	virtual void reconstructVolume(
		NLosData* nlosData, const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers,
		const TransientParameters& transientParams) = 0;
};

