#pragma once

#include "acquisition/RawSensorDataReader.h"
#include "binning/FrameHistogramBuilder.h"
#include "parsing/RawSensorDataParser.h"
#include "reconstruction/FastRSDImageReconstructor.h"
#include "data/SceneParameters.h"

namespace rtnlos
{
	template<int NROWS, int NCOLS, int NFREQ>
	class NlosStreamingEngine
	{
		using RawSensorDataParserType = RawSensorDataParser<NROWS, NCOLS>;
		using FrameHistogramBuilderType = FrameHistogramBuilder<NROWS* NCOLS, NFREQ>;
		using FastRSDImageReconstructorType = FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>;

	private:
		SceneParameters					_sceneParameters;

		// processor for each stage
		RawSensorDataReader				_reader;
		RawSensorDataParserType			_parser;
		FrameHistogramBuilderType		_binner;
		FastRSDImageReconstructorType	_reconstructor;
		spdlog::level::level_enum		_logLevel;

		// queues for transition between each stage
		RawSensorDataQueue				_rawSensorDataQueue;
		ParsedSensorDataQueue			_parsedSensorDataQueue;
		FrameHistogramDataQueue			_frameHistogramDataQueue;
		ReconstructedImageDataQueue		_reconstructedImageDataQueue;

		std::atomic<bool>				_isRunning;

	private:
		void Initialize(const std::string& configPath);

	public:
		NlosStreamingEngine(const std::string& dataPath, const std::string& configPath);

		void Start();
		void Stop();
	};

}