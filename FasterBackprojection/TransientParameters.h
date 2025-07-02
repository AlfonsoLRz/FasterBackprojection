#pragma once

#include "NLosData.h"

class TransientParameters
{
public:
	float				_t0;
	float				_t1;

	glm::uint			_maxBounces;
	glm::uint			_numSamples;
	glm::uint			_numPaths;
	glm::uint			_rrDepth;

	bool				_unwarpCamera;
	bool				_discardDirectLight;
	bool				_hideEmitters;

	glm::uint			_temporalFilter;
	float				_gaussianStdDev;

	// NLOS
	CaptureSystem		_captureSystem;

	glm::vec3			_laserPosition;
	glm::vec3			_cameraPosition;
	float				_sensorFov;

	glm::ivec2			_spadResolution;
	float				_temporalResolution;
	float				_timeOffset;
	int					_numTimeBins;
	int					_numPhotonsPerPoint;
	glm::ivec2			_laserFocusPoint;					// Only valid for simple setup

	bool				_geometrySampling;
	bool				_geometrySamplingDoRoulette;		// As used in tal, but we don't use it here; it does not make sense to always sample the hidden geometry, as it is not always present

	bool				_discardFirstLastBounces;

	int					_contrastNeighbourhoodSize;
	int					_logFilterNeighbourhoodSize;
	float				_wavelengthMean, _wavelengthSigma;

	bool				_useFourierFilter;
	bool				_reconstructHiddenGeometryAABB;
	bool				_reconstructShape;
	std::vector<double>	_reconstructionDepths;
	glm::uvec3			_voxelResolution;

	float				_gaussianSigmaReconstruction;
	std::string			_outputFolder;
	bool				_saveTransientCube;

	TransientParameters() :
		_t0(15.0f),
		_t1(30.0f),
		_maxBounces(10),
		_numSamples(64),
		_numPaths(128),
		_rrDepth(10),
		_unwarpCamera(true),
		_discardDirectLight(false),
		_hideEmitters(false),
		_temporalFilter(0),
		_gaussianStdDev(2.0f),

		//
		_captureSystem(Confocal),
		_laserPosition(-2.6f, 0.28f, 0.7f),
		_cameraPosition(_laserPosition),
		_sensorFov(1.0f),
		_spadResolution(16),
		_temporalResolution(1e-11f),
		_timeOffset(650 * _temporalResolution),
		_numTimeBins(300),
		_numPhotonsPerPoint(200),
		_laserFocusPoint(_spadResolution / 2),
		_geometrySampling(true),
		_geometrySamplingDoRoulette(true),
		_discardFirstLastBounces(true),

		_contrastNeighbourhoodSize(3),
		_logFilterNeighbourhoodSize(9),
		_wavelengthMean(0.25f),
		_wavelengthSigma(0.25f),

		_useFourierFilter(false),
		_reconstructHiddenGeometryAABB(false),
		_reconstructShape(true),
		_reconstructionDepths({0.98f, 0.99f, 1.0f, 1.01f, 1.02f}),
		_voxelResolution(64, 64, 64),

		_gaussianSigmaReconstruction(4.0f),
		_outputFolder("output/"),
		_saveTransientCube(false)
	{
	}
};
