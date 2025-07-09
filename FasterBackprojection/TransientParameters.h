#pragma once

#include "NLosData.h"

enum PostprocessingFilterType 
{
	NONE,
	LAPLACIAN,
	LOG,
	LOG_FFT,
	NUM_POSTPROCESSING_FILTERS
};


class TransientParameters
{
public:
	bool						_useFourierFilter;
	bool						_reconstructAABB;
	glm::uint					_numReconstructionDepths;
	glm::uvec3					_voxelResolution;

	PostprocessingFilterType	_postprocessingFilterType;
	int							_kernelSize;
	float						_sigma;

	std::string					_outputFolder;
	bool						_saveReconstructedBoundingBox;

	TransientParameters() :
		_useFourierFilter(true),
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
