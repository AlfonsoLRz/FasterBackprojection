#pragma once

#include <cufft.h>

#include "ApplicationState.h"
#include "TransientParameters.h"

class Camera;
class PostprocessingFilters;
class SceneContent;

using Complex = std::complex<float>;

class Laser
{
private:
	NLosData* _nlosData = nullptr;

	static const PostprocessingFilters* _postprocessingFilters[PostprocessingFilterType::NUM_POSTPROCESSING_FILTERS];

private:
	static std::vector<double> linearSpace(double minValue, double maxValue, int n);
	float* padIntensity(cufftComplex*& paddedIntensity, size_t padding, const std::string& mode) const;
	void filter_H_cuda(float wl_mean, float wl_sigma = .0f, const std::string& border = "zero") const;
	static void normalizeMatrix(float* v, glm::uint size);

	static void buildAliasTables(const std::vector<float>& cdf, std::vector<glm::uint>& aliasTable, std::vector<float>& probTable);

	static void reconstructShapeAABB(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers, const TransientParameters& transientParams);
	static void reconstructShapeDepths(const ReconstructionInfo& recInfo, const TransientParameters& transientParams);

	static void reconstructDepthConfocal(const ReconstructionInfo& recInfo, std::vector<double>& reconstructionDepths);
	static void reconstructDepthExhaustive(const ReconstructionInfo& recInfo, std::vector<double>& reconstructionDepths);

	static void reconstructAABBConfocal(float* volume, const ReconstructionInfo& recInfo);
	static void reconstructAABBExhaustive(float* volume, const ReconstructionInfo& recInfo);

	static bool saveReconstructedAABB(const std::string& filename, float* voxels, glm::uint numVoxels);

public:
	void reconstructAABBConfocalMIS(float* volume, const ReconstructionInfo& recInfo) const;

public:
	Laser(NLosData* nlosData);
	virtual ~Laser();

	void reconstructShape(const TransientParameters& transientParams);
};
