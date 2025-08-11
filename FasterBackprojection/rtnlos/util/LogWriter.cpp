#include "stdafx.h"

#include "cxxopts/cxxopts.hpp"
#include "../data/SceneParameters.h"
#include "LogWriter.h"

namespace NLOS {

	void LogWriter::InitCmdLineOptions(cxxopts::Options& options) 
	{ 
		options.add_options(m_name)
			("l,logdir", "log file directory", cxxopts::value<std::string>(m_logDir))
			("lograw", "log raw T3 records", cxxopts::value<bool>(m_logRaw))
			("logpar", "log type for parsed records (bin|yml)", cxxopts::value<std::string>(m_logParOpt))
			("logfdh", "log type for FDH records (bin|yml)", cxxopts::value<std::string>(m_logFdhOpt))
			("logrsd", "log type for RSD records (bin|yml)", cxxopts::value<std::string>(m_logRsdOpt))
			("logimg", "log type for output images (png|bin|yml|monopng)", cxxopts::value<std::string>(m_logImgOpt));
	}

	void LogWriter::Initialize(const SceneParameters& sceneParameters) {

	}

	void LogWriter::Work()
	{
		//make this a low priority thread, since it is only doing logging
		BOOL ret = SetThreadPriority(m_thread.native_handle(), THREAD_PRIORITY_BELOW_NORMAL);  // base_priority-1
		if (!ret)
			spdlog::warn("{:<25}: FAILED to set thread priority to THREAD_PRIORITY_BELOW_NORMAL. err={}", m_name, GetLastError());

		while (!m_stopRequested) {
			// empty the incoming queue before checking for a stop request (process any backlog before exiting)
			while (m_incoming.Size() > 0) {
				PipelineDataPtr data;
				if (m_incoming.Pop(data)) {
					data->LogToFile(this);
				}
				auto remaining = m_incoming.Size();
				if (m_stopRequested && remaining) {
					spdlog::info(",{:<25},Backlog of {} entries left to log before shutdown.", m_name, remaining);
				} 
			}
		}
	}

	void LogWriter::OnStop()
	{
		spdlog::trace("{:<25}: Abort", m_name);
		m_incoming.Abort();
	}

	bool LogWriter::PrepRawFile()
	{
		std::string fname = m_logDir + "/T3Recs.out";
		try {
			m_rawLogStream.exceptions(std::ofstream::failbit | std::ofstream::badbit);
			m_rawLogStream.open(fname, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
		}
		catch (const std::ofstream::failure & e) {
			spdlog::warn(",{:<25},Failed to open {} for raw logging. Error was: {}", m_name, fname, e.what());
			return false;
		}
		return true;
	}

}
