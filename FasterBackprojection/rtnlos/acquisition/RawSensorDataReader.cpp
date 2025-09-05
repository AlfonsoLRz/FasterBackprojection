#include "stdafx.h"
#include "RawSensorDataReader.h"

namespace rtnlos
{
	RawSensorDataReader::RawSensorDataReader(const std::string& dataPath, RawSensorDataQueue& outgoing)
		: _dataPath(dataPath),
		  _outgoingRaw(outgoing)
	{
	}

	void RawSensorDataReader::DoWork()
	{
		spdlog::info("Starting RawSensorDataReader worker thread for file '{}'", _dataPath);
		_workerThread = std::jthread(&RawSensorDataReader::ReadFromFile, this, _dataPath);



	}

	void RawSensorDataReader::Stop()
	{
		_workerThread.join();
	}

	void RawSensorDataReader::ReadFromFile(const std::string& dataPath) const
	{
		FILE* fp = nullptr;
		fopen_s(&fp, dataPath.c_str(), "rb");
		if (fp == nullptr) 
			throw std::logic_error("cannot open file");

		// How many records does it contain?
		fseek(fp, 0L, SEEK_END);

		int totalRecords = ftell(fp) / sizeof(T3Rec);
		if (ftell(fp) % sizeof(T3Rec) != 0 || totalRecords == 0)
			throw std::logic_error("File size is not a multiple of record size.");

		rewind(fp);

		// Read all the records into an array
		std::unique_ptr<T3Rec> recs(new T3Rec[totalRecords]);
		if (recs == nullptr)
			throw std::logic_error("Allocation error!");

		size_t ret = fread(recs.get(), sizeof(T3Rec), totalRecords, fp);
		if (ret != totalRecords) 
			throw std::logic_error("Unable to read full file.");

		// Looping through the file's records forever
		int curRecord = 0;
		while (!_stop) 
		{
			RawSensorDataPtr chunk(new RawSensorDataType());

			if (curRecord >= totalRecords) // If we're at the end, reset to beginning of file.
				curRecord = 0;

			if (curRecord == 0)  // If we're at the beginning, notify the next stage that we reset
				chunk->_fileReaderWasResetFlag = true;

			chunk->_numRecords = std::min(RAW_SENSOR_READ_BLOCK_SIZE, totalRecords - curRecord);
			memcpy(chunk->_records, &(recs.get()[curRecord]), chunk->_numRecords * sizeof(T3Rec));
			curRecord += chunk->_numRecords;

			_outgoingRaw.Push(chunk);

			if (_stop) 
				break;

			// Wait a bit (for demo purposes only)
			//std::this_thread::sleep_for(std::chrono::milliseconds(50));
			spdlog::trace("Pushed {} records to the outgoing queue (incoming queue size={})", chunk->_numRecords, _outgoingRaw.Size());
		}

		spdlog::warn("RawSensorDataReader worker thread stopped");
	}
}
