#include "stdafx.h"
#include "SPADStreamingEngine.h"

namespace rtnlos
{
	template<int NROWS, int NCOLS, int NFREQ>
	SPADStreamingEngine<NROWS, NCOLS, NFREQ>::SPADStreamingEngine(const std::string& dataPath, const std::string& configPath)
		: StreamingEngine(NCOLS, NROWS)
		, _reader(dataPath, _rawSensorDataQueue)
		, _parser(_rawSensorDataQueue, _parsedSensorDataQueue)
		, _binner(_parsedSensorDataQueue, _frameHistogramDataQueue)
		, _reconstructor(_frameHistogramDataQueue)
	{
		Initialize(configPath);
	}

	template<int NROWS, int NCOLS, int NFREQ>
	void SPADStreamingEngine<NROWS, NCOLS, NFREQ>::Initialize(const std::string& configPath)
	{
		spdlog::trace("Enabled trace level logging");
		spdlog::set_level(spdlog::level::info);

		_sceneParameters.Initialize(configPath);
		_parser.Initialize(_sceneParameters);
		_binner.Initialize(_sceneParameters);
		_reconstructor.Initialize(_sceneParameters, &_viewportSurface);
	}

	template<int NROWS, int NCOLS, int NFREQ>
	void SPADStreamingEngine<NROWS, NCOLS, NFREQ>::Start()
	{
		spdlog::trace("Starting NLOS Streaming Engine");

		_reader.DoWork();
		_parser.DoWork();
		_binner.DoWork();
		_reconstructor.DoWork();

		spdlog::info("NLOS Streaming Engine Started");
	}

	template<int NROWS, int NCOLS, int NFREQ>
	void SPADStreamingEngine<NROWS, NCOLS, NFREQ>::Stop()
	{
		spdlog::trace("Stopping NLOS Streaming Engine");

		_reconstructor.Stop();
		_binner.Stop();
		_parser.Stop();
		_reader.Stop();

		spdlog::info("Stopped NLOS Streaming Engine");
	}

	template class SPADStreamingEngine<NUMBER_OF_SPAD_ROWS, NUMBER_OF_SPAD_COLS, NUMBER_OF_SPAD_FREQUENCIES>;
}
