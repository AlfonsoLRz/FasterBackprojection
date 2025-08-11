#pragma once

#include "types.h"
#include "acquisition/RawSensorDataReader.h"
#include "parsing/RawSensorDataParser.h"
#include "reconstruction/FastRSDImageReconstructor.h"
#include "util/cxxopts/cxxopts.hpp"
#include "util/LogWriter.h"
#include "data/SceneParameters.h"

namespace NLOS
{
	template<int NROWS, int NCOLS, int NFREQ>
	class NLOSStreamingEngine {
		using RawSensorDataParserType = RawSensorDataParser<NROWS, NCOLS>;
		using FastRSDImageReconstructorType = FastRSDImageReconstructor<NROWS, NCOLS, NFREQ>;

	public:
		NLOSStreamingEngine(int argc, char* argv[]);

		void Initialize(int argc, char* argv[]);

		void Start();
		void Stop();

	private:
		SceneParameters m_sceneParameters;

		// processor for each stage
		RawSensorDataReader m_reader;
		RawSensorDataParserType m_parser;
		FastRSDImageReconstructorType m_reconstructor;
		LogWriter m_logWriter;
		spdlog::level::level_enum m_logLevel;

		const std::vector<DataProcessor*> m_processors;

		// queues for transition between each stage
		RawSensorDataQueue m_rawSensorDataQueue;
		ParsedSensorDataQueue m_parsedSensorDataQueue;
		FrameHistogramDataQueue m_frameHistogramDataQueue;
		ReconstructedImageDataQueue m_reconstructedImageDataQueue;
		PipelineDataQueue m_logQueue;

		// queue of keyboard input from the opencv window
		KeyboardInputQueue m_keyboardInputQueue;
		std::atomic<bool> m_isRunning;

		static std::string s_keyboard_shortcuts;
	};

}