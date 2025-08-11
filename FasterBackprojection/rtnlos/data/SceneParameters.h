#pragma once
#include <string>

namespace NLOS {
	class SceneParameters {
	public:
		SceneParameters() { };
		SceneParameters(const std::string& filename) {
			Initialize(filename);
		}

		void Initialize(const std::string& filename);

		const float C_Light = 299792458.f;
		const float Picosecond = 1E-12f;

		int NumComponents;
		int AptWidth;
		int AptHeight;
		int SkipRate;
		int SpadIDs;
		float SyncRate;
		float GalvoRate;
		int CenteringCoeff;
		std::vector<uint32_t> BinZeros;
		uint32_t BinLength;
		float DeltaT;
		int DownsamplingRate;
		float Z_Gate;
		float Resolution;
		float DepthMin;
		float DepthMax;
		float DepthDelta;
		float DepthOffset;
		float SamplingSpacing;
		std::vector<float> Weights;
		std::vector<float> Lambdas;
		std::vector<float> Omegas;
		std::vector<float> FreMask;
		std::vector<float> ApertureFullSize;
		std::vector<float> D1;
		std::vector<std::vector<float>> SpadIndex;
		std::vector<std::vector<float>> Offset;
		std::vector<std::vector<float>> T0_Gated;
		std::vector<std::vector<float>> D4;
	};
}
