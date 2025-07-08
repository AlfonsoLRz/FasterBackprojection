#pragma once

#include <cufft.h>

#include "ApplicationState.h"
#include "TransientParameters.h"

class Camera;
class SceneContent;

using Complex = std::complex<float>;

class Laser
{
private:
	NLosData* _nlosData = nullptr;

private:
	static std::vector<double> linearSpace(double minValue, double maxValue, int n);
	void padIntensity(std::vector<cufftComplex>& paddedIntensity, size_t padding, const std::string& mode) const;

	static void buildAliasTables(const std::vector<float>& cdf, std::vector<glm::uint>& aliasTable, std::vector<float>& probTable);

	static void fftLoG(float*& inputVoxels, const glm::uvec3& resolution, float sigma);
	static void laplacianFilter(float*& inputVoxels, const glm::uvec3& resolution, glm::uint filterSize);
	static void normalizeMatrix(float* v, glm::uint size);

	void reconstructShapeAABB(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers, bool filterFourier);
	static void reconstructShapeDepths(const ReconstructionInfo& recInfo);

	static void reconstructDepthConfocal(const ReconstructionInfo& recInfo, std::vector<double>& reconstructionDepths);
	static void reconstructDepthExhaustive(const ReconstructionInfo& recInfo, std::vector<double>& reconstructionDepths);

	static void reconstructAABBConfocal(const ReconstructionInfo& recInfo, bool filterFourier);
	static void reconstructAABBExhaustive(const ReconstructionInfo& recInfo, bool filterFourier);

	bool saveReconstructedAABB(const std::string& filename, float* voxels, glm::uint numVoxels);

public:
	void reconstructAABBConfocalMIS(const ReconstructionInfo& recInfo, const ReconstructionBuffers& recBuffers);

public:
	Laser(NLosData* nlosData);
	virtual ~Laser();

	void filter_H_cuda(float wl_mean, float wl_sigma = .0f, const std::string& border = "zero") const;
	void reconstructShape(ReconstructionInfo& recInfo, ReconstructionBuffers& recBuffers, bool reconstructAABB, bool filterFourier = true);
};
