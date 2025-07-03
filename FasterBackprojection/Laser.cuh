#pragma once

#include "ApplicationState.h"
#include "TransientParameters.h"

class Camera;
class SceneContent;

class Laser
{
private:
	static std::vector<double> fftFrequencies(int n, float d);
	template<typename T>
	static std::vector<double> fftShift(const std::vector<T>& frequencies);
	template<typename T>
	std::vector<T> ifftshift(const std::vector<T>& input);

	static std::vector<double> linearSpace(double minValue, double maxValue, int n);
	void fourierFilter(const TransientParameters& transientParameters);

	static void normalizeMatrix(float* v, glm::uint numVoxels);

	static void reconstructShapeAABB(const ReconstructionInfo& recInfo);
	static void reconstructShapeDepths(const ReconstructionInfo& recInfo);

	static void reconstructDepthConfocal(const ReconstructionInfo& recInfo, std::vector<double>& reconstructionDepths);
	static void reconstructDepthExhaustive(const ReconstructionInfo& recInfo, std::vector<double>& reconstructionDepths);

	static void reconstructAABBConfocal(const ReconstructionInfo& recInfo);
	static void reconstructAABBExhaustive(const ReconstructionInfo& recInfo);

public:
	Laser();
	virtual ~Laser();

	static void reconstructShape(ReconstructionInfo& recInfo, ReconstructionBuffers& recBuffers, bool reconstructAABB);
};

template <typename T>
std::vector<double> Laser::fftShift(const std::vector<T>& frequencies)
{
	int n = static_cast<int>(frequencies.size());
	std::vector<T> shiftedFrequencies(n);
	int half = (n + 1) / 2;

	for (int i = half; i < n; ++i)
		shiftedFrequencies[i - half] = frequencies[i];
	for (int i = 0; i < half; ++i)
		shiftedFrequencies[i + half] = frequencies[i];

	return shiftedFrequencies;
}

template <typename T>
std::vector<T> Laser::ifftshift(const std::vector<T>& input)
{
	int n = static_cast<int>(input.size());
	int shift = n / 2; // floor(n / 2) — works for both even and odd
	std::vector<T> output(n);

	for (int i = 0; i < n; ++i) 
		output[i] = input[(i + shift) % n];

	return output;
}
