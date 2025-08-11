#pragma once

#include <memory>
#include "../util/SafeQueue.h"
#include "T3Rec.h"

namespace NLOS
{
	class ILogContext;

	// base class for all data that flows through the pipeline
	class PipelineData {
	public:
		virtual void LogToFile(ILogContext* pContext) { };
	};
};