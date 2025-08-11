#include "stdafx.h"
#include "NLOSStreamingEngine.h"
#include "compile_time_constants.h"


namespace NLOS
{
	template<int NROWS, int NCOLS, int NFREQ>
	NLOSStreamingEngine<NROWS, NCOLS, NFREQ>::NLOSStreamingEngine(int argc, char* argv[])
		: m_reconstructedImageDataQueue(SafeQueuePushBehavior::WaitIfFull, SafeQueuePopBehavior::FailIfEmpty)
		, m_reader(m_rawSensorDataQueue, m_logWriter)
		, m_parser(m_rawSensorDataQueue, m_parsedSensorDataQueue, m_logWriter)
		, m_reconstructor(m_frameHistogramDataQueue, m_reconstructedImageDataQueue, m_logWriter)
		, m_logWriter(m_logQueue)
		, m_processors{ &m_reader, &m_parser, &m_reconstructor, &m_logWriter }
		, m_logLevel(spdlog::level::info)
	{
		Initialize(argc, argv);
	}

	template<int NROWS, int NCOLS, int NFREQ>
	void NLOSStreamingEngine<NROWS, NCOLS, NFREQ>::Initialize(int argc, char* argv[]) {
		bool showHelp = false;
		bool debugLogging = false;
		std::string sceneParametersFilename;
		cxxopts::Options options(argv[0], "Streaming NLOS Image Reconstruction");

		try {
			options.add_options("General")
				("h,help", "Show help", cxxopts::value<bool>(showHelp))
				("d,debug", "enable debug output", cxxopts::value<bool>(debugLogging))
				("s,scene", "Scene parameters file (yml)", cxxopts::value<std::string>(sceneParametersFilename));

			// each processor has the opportunity to register its own command line options
			for (auto p : m_processors)
				p->InitCmdLineOptions(options);

			// parse the command line
			auto parsedOptions = options.parse(argc, argv);
		}
		catch (const cxxopts::OptionParseException & ex) {
			std::cerr << ex.what();
			std::cerr << options.help({}) << std::endl;
			exit(-1);
		}

		if (showHelp) {
			std::cout << options.help({})
					  << s_keyboard_shortcuts << std::endl;
			exit(0);
		}

		if (debugLogging) {
			m_logLevel = spdlog::level::trace;
			spdlog::trace("Enabled trace level logging");
		}
		spdlog::set_level(m_logLevel);

		m_sceneParameters.Initialize(sceneParametersFilename);

		for (auto p : m_processors)
			p->Initialize(m_sceneParameters);
	}

	template<int NROWS, int NCOLS, int NFREQ>
	void NLOSStreamingEngine<NROWS, NCOLS, NFREQ>::Start() {
		spdlog::trace("Starting NLOS Streaming Engine");
		for (auto p : m_processors)
			p->Start();
		m_isRunning = true;
		spdlog::info("NLOS Streaming Engine Started");

		// listen for keystrokes forever (until someone calls Stop)
		char key;
		while (m_isRunning) {
			m_keyboardInputQueue.Pop(key);
		}
	}

	template<int NROWS, int NCOLS, int NFREQ>
	void NLOSStreamingEngine<NROWS, NCOLS, NFREQ>::Stop() {
		spdlog::trace("Stopping NLOS Streaming Engine");
		for (auto p : m_processors)
			p->Stop();
		spdlog::info("Stopped NLOS Streaming Engine");
	}

	template<int NROWS, int NCOLS, int NFREQ>
	std::string NLOSStreamingEngine<NROWS, NCOLS, NFREQ>::s_keyboard_shortcuts(R"LITERAL(
Keyboard Shortcuts while running:
  Q, q    Quit
  H, h	  Print help
  V, v	  Toggle verbose output
  C, c    Cycle colormap forward / backward
  Z, z    Decrease / Increase window size (zoom out / in)
  R, r    Reset window size to 1:1
  T       Enable adaptive scene threshold
  t       Disable adaptive scene threshold
  B       Reset bandpass filter to [0.1, 0.9]
  b       Set bandpass filter to [0.0, 1.0] (disabled)
  {       Decrease bandpass f_high by 10%
  }       Increase bandpass f_high by 10%
  [       Decrease bandpass f_low by 10%
  ]       Increase bandpass f_low by 10%
  D       Turn ON depth dependent averaging
  d       Turn OFF depth dependent averaging
)LITERAL");

	template class NLOSStreamingEngine<NUMBER_OF_ROWS, NUMBER_OF_COLS, NUMBER_OF_FREQUENCIES>;
}
