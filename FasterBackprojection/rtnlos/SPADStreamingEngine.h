#pragma once

#include "../StreamingEngine.h"

#include "acquisition/RawSensorDataReader.h"
#include "binning/FrameHistogramBuilder.h"
#include "data/SceneParameters.h"
#include "parsing/RawSensorDataParser.h"
#include "reconstruction/FastRSDImageReconstructor.h"

namespace rtnlos
{
	template<int NROWS, int NCOLS, int NFREQ>
	class SPADStreamingEngine: public StreamingEngine
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

		// queues for transition between each stage
		RawSensorDataQueue				_rawSensorDataQueue;
		ParsedSensorDataQueue			_parsedSensorDataQueue;
		FrameHistogramDataQueue			_frameHistogramDataQueue;

	private:
		void Initialize(const std::string& configPath);

	public:
		SPADStreamingEngine(const std::string& dataPath, const std::string& configPath);

		void Start();
		void Stop();

		ViewportSurface& getViewportSurface() { return _viewportSurface; }
		static int getImageWidth() { return NROWS; }
		static int getImageHeight() { return NCOLS; }
	};

	using ReconstructionEngine = rtnlos::SPADStreamingEngine<NUMBER_OF_SPAD_ROWS, NUMBER_OF_SPAD_COLS, NUMBER_OF_SPAD_FREQUENCIES>;
}