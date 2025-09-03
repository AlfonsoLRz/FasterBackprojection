#pragma once

enum PostprocessingFilterType
{
	NONE,
	LAPLACIAN,
	LOG,
	LOG_FFT,
	NUM_POSTPROCESSING_FILTERS
};

enum ReconstructionType
{
	BACKPROJECTION,
	LCT_REC,
	FK_MIGRATION,
	PHASOR_FIELDS,
	NUM_RECONSTRUCTION_TYPES
};

class TransientParameters
{
public:
	ReconstructionType			_reconstructionType = ReconstructionType::FK_MIGRATION;
	bool						_useFourierFilter = false;
	bool						_compensateLaserCosDistance = true;
	bool						_reconstructAABB = true;
	glm::uint					_numReconstructionDepths = 200;
	glm::uvec3					_voxelResolution = glm::uvec3(256u);

	PostprocessingFilterType	_postprocessingFilterType = PostprocessingFilterType::NONE;
	int							_kernelSize = 5;	
	float						_sigma = 0.3f;

	std::string					_outputFolder = "output/";
	std::string					_outputMaxImageName = "max_activation.png";
	std::string					_outputAABBName = "reconstruction.aabb";

	bool						_saveReconstructedBoundingBox = true;
	bool						_saveMaxImage = true;
};
