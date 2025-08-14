#include "stdafx.h"
#include "NlosStreamingEngine.h"

namespace rtnlos
{
	template<int NROWS, int NCOLS, int NFREQ>
	NlosStreamingEngine<NROWS, NCOLS, NFREQ>::NlosStreamingEngine(const std::string& dataPath, const std::string& configPath, cudaSurfaceObject_t cudaSurface)
		: _reader(dataPath, _rawSensorDataQueue)
		, _parser(_rawSensorDataQueue, _parsedSensorDataQueue)
		, _binner(_parsedSensorDataQueue, _frameHistogramDataQueue)
		, _reconstructor(_frameHistogramDataQueue, _reconstructedImageDataQueue)
		, _logLevel(spdlog::level::info)
		, _reconstructedImageDataQueue(SafeQueuePushBehavior::WaitIfFull, SafeQueuePopBehavior::FailIfEmpty)
	{
		Initialize(configPath);
	}

	template<int NROWS, int NCOLS, int NFREQ>
	void NlosStreamingEngine<NROWS, NCOLS, NFREQ>::Initialize(const std::string& configPath)
	{
		spdlog::trace("Enabled trace level logging");
		spdlog::set_level(_logLevel);

		_sceneParameters.Initialize(configPath);
		_parser.Initialize(_sceneParameters);
		_binner.Initialize(_sceneParameters);
		_reconstructor.Initialize(_sceneParameters);
	}

	template<int NROWS, int NCOLS, int NFREQ>
	void NlosStreamingEngine<NROWS, NCOLS, NFREQ>::Start()
	{
		spdlog::trace("Starting NLOS Streaming Engine");

		_isRunning = true;
		_reader.DoWork();
		_parser.DoWork();
		_binner.DoWork();
		_reconstructor.DoWork();

		spdlog::info("NLOS Streaming Engine Started");
	}

	template<int NROWS, int NCOLS, int NFREQ>
	void NlosStreamingEngine<NROWS, NCOLS, NFREQ>::Stop()
	{
		spdlog::trace("Stopping NLOS Streaming Engine");

		_reconstructor.Stop();
		_binner.Stop();
		_parser.Stop();
		_reader.Stop();

		spdlog::info("Stopped NLOS Streaming Engine");
	}

	template class NlosStreamingEngine<NUMBER_OF_ROWS, NUMBER_OF_COLS, NUMBER_OF_FREQUENCIES>;
}
