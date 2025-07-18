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
	PHASOR_FIELD,
	NUM_RECONSTRUCTION_TYPES
};

class TransientParameters
{
public:
	ReconstructionType			_reconstructionType;
	bool						_useFourierFilter;
	bool						_compensateLaserCosDistance;
	bool						_reconstructAABB;
	glm::uint					_numReconstructionDepths;
	glm::uvec3					_voxelResolution;

	PostprocessingFilterType	_postprocessingFilterType;
	int							_kernelSize;
	float						_sigma;

	std::string					_outputFolder;
	bool						_saveReconstructedBoundingBox;

	TransientParameters() :
		_reconstructionType(ReconstructionType::LCT_REC),
		_useFourierFilter(true),
		_compensateLaserCosDistance(true),
		_reconstructAABB(true),
		_numReconstructionDepths(200),
		_voxelResolution(256),

		_postprocessingFilterType(PostprocessingFilterType::NONE),
		_kernelSize(5),
		_sigma(1.0f),

		_outputFolder("output/"),
		_saveReconstructedBoundingBox(false)
	{
	}
};
