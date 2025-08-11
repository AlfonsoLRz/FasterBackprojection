#include "stdafx.h"
#include "RawSensorDataReader.h"
#include "../compile_time_constants.h"

namespace NLOS {

	void RawSensorDataReader::InitCmdLineOptions(cxxopts::Options& options)
	{
		// add any command line options specific to the data reader here.
		options.add_options(m_name)
			("f,file", "read raw data from file", cxxopts::value<std::string>(m_rawDataFileName));
	}

	// This worker should read T3Rec data from the hardware (or data file)
	// It should not do any parsing of the data.
	// It should package it into an array of whatever size it decides
	// is appropriate (best size can be fine tuned by testing)
	// The resulting array of records should be pushed to the outgoing queue.
	void RawSensorDataReader::Work()
	{
		if (m_rawDataFileName.size() > 0)
			ReadFromFile();
	}

    void RawSensorDataReader::OnStop() {
        // there may be a push blocking if the queue is full, so tell the queue to abort its operation
        spdlog::trace("{:<25}: Abort", m_name);
        m_outgoingRaw.Abort();
    }


	// this function is used to read the input data from a raw log file 
	// for offline testing purposes.
	void RawSensorDataReader::ReadFromFile()
	{
		// open the file
		FILE* fp = fopen(m_rawDataFileName.c_str(), "rb");
		if (fp == NULL) {
			throw std::logic_error("cannot open file");
		}

		// how many records does it contain?
		fseek(fp, 0L, SEEK_END);
		int totalRecords = ftell(fp) / sizeof(T3Rec);
		if (ftell(fp) % sizeof(T3Rec) != 0 || totalRecords == 0)
			throw std::logic_error("file size is not a multiple of record size");
		rewind(fp);

		// read all the records into an array
		std::unique_ptr<T3Rec> recs(new T3Rec[totalRecords]);
		if (recs == nullptr)
			throw std::logic_error("allocation error");
		size_t ret = fread(recs.get(), sizeof(T3Rec), totalRecords, fp);
		if (ret != totalRecords) {
			throw std::logic_error("unable to read full file");
		}

		// being looping through the file's records forever
		int curRecord = 0;
		while (!m_stopRequested) {
			RawSensorDataPtr chunk(new RawSensorDataType());

			if (curRecord >= totalRecords) { // if we're at the end, reset to beginning of file.
				curRecord = 0;
			}

			if (curRecord == 0) { // if we're at the beginning, notify the next stage that we reset
				spdlog::trace("{:<25}: File reader resetting to beginning", m_name);
				chunk->FileReaderWasResetFlag = true;
			}

			chunk->NumRecords = std::min(RAW_SENSOR_READ_BLOCK_SIZE, totalRecords - curRecord);
			memcpy(chunk->Records, &(recs.get()[curRecord]), chunk->NumRecords * sizeof(T3Rec));
			curRecord += chunk->NumRecords;

			//spdlog::trace("{:<25}: filled outgoing buffer with {} records in {} msec. Pushing to queue (sz={})", m_name, chunk->NumRecords, timer.Stop(), m_outgoingRaw.Size());

			// push onto outgoing queue
			m_outgoingRaw.Push(chunk);

            // if we are logging raw data, push it into the queue to be logged.
            if (m_logWriter.LogRawData()) {
                m_logWriter.PushLog(chunk);
            }

			if (m_stopRequested) {
				break;
			}

			// wait a bit (for demo purposes only)
			//std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
	}
}
