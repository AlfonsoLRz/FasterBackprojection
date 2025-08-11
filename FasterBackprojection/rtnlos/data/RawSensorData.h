#pragma once

#include <memory>
#include "../util/SafeQueue.h"

#include "../util/ILogContext.h"
#include "T3Rec.h"
#include "PipelineData.h"

namespace NLOS
{
	template<int NUM_RECORDS>
	class RawSensorData : public PipelineData {
	public:
		RawSensorData() 
			: NumRecords(0)
			, FileReaderWasResetFlag(false)
		{ }

		int NumRecords;
		T3Rec Records[NUM_RECORDS];
		bool FileReaderWasResetFlag;
	};
}