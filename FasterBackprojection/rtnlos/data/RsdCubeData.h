#pragma once

namespace rtnlos
{
	template<int NROWS, int NCOLS>
	class RsdCubeData
	{
	public:
		std::unique_ptr<std::vector<float>> _cube;
		int									_numDepths;
		uint32_t							_frameNumber;

	public:
		RsdCubeData(uint32_t frameNumber, std::unique_ptr<std::vector<float>> &cube): _cube(std::move(cube)), _numDepths(0), _frameNumber(frameNumber)
		{
		};
	};
}