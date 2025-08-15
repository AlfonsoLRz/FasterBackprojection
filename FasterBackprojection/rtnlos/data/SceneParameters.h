#pragma once

#include "../stdafx.h"

namespace rtnlos
{
	class SceneParameters
	{
	public:
		SceneParameters() = default;
		SceneParameters(const std::string& filename)
		{
			Initialize(filename);
		}

		void Initialize(const std::string& filename);

		const float LIGHT_SPEED = 299792458.f;
		const float PICOSECOND = 1e-12f;

		int								_numComponents;
		int								_apertureWidth;
		int								_apertureHeight;
		int								_skipRate;
		int								_spadIDs;
		float							_syncRate;
		float							_galvoRate;
		int								_centeringCoeff;
		std::vector<uint32_t>			_binZeros;
		uint32_t						_binLength;
		float							_deltaT;
		int								_downsamplingRate;
		float							_zGate;
		float							_resolution;
		float							_depthMin;
		float							_depthMax;
		float							_depthDelta;
		float							_depthOffset;
		float							_samplingSpacing;
		std::vector<float>				_weights;
		std::vector<float>				_lambdas;
		std::vector<float>				_omegas;
		std::vector<float>				_freMask;
		std::vector<float>				_apertureFullSize;
		std::vector<float>				_d1;
		std::vector<std::vector<float>> _spadIndex;
		std::vector<std::vector<float>> _offset;
		std::vector<std::vector<float>> _t0Gated;
		std::vector<std::vector<float>> _d4;
	};

	struct DatasetInfo
	{
		std::string _name;
		float		_apertureDstWidth;
		float		_apertureDstHeight;
		float		_minDistance;
		float		_maxDistance;
		float		_deltaDistance;
		float		_distanceOffset;
	};
}
