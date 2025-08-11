#pragma once

#include <memory>
#include "util/SafeQueue.h"
#include "data/RawSensorData.h"
#include "data/ParsedSensorData.h"
#include "data/FrameHistogramData.h"
#include "data/RsdCubeData.h"
#include "data/ReconstructedImageData.h"
#include "compile_time_constants.h"

namespace NLOS {;

	using RawSensorDataType = RawSensorData<RAW_SENSOR_READ_BLOCK_SIZE>;
	using FrameHistogramDataType = FrameHistogramData<NUMBER_OF_ROWS * NUMBER_OF_COLS, NUMBER_OF_FREQUENCIES>;
	using RsdCubeDataType = RsdCubeData<NUMBER_OF_ROWS, NUMBER_OF_COLS>;
	using ReconstructedImageDataType = ReconstructedImageData<NUMBER_OF_ROWS, NUMBER_OF_COLS>;

	using PipelineDataPtr = std::shared_ptr<PipelineData>;
	using RawSensorDataPtr = std::shared_ptr<RawSensorDataType>;
	using ParsedSensorDataPtr = std::shared_ptr<ParsedSensorData>;
	using FrameHistogramDataPtr = std::shared_ptr<FrameHistogramDataType>;
	using RsdCubeDataPtr = std::shared_ptr<RsdCubeDataType>;
	using ReconstructedImageDataPtr = std::shared_ptr<ReconstructedImageDataType>;

	using PipelineDataQueue = SafeQueue<PipelineDataPtr, -1>;
	using RawSensorDataQueue = SafeQueue<RawSensorDataPtr, 200>;
	using ParsedSensorDataQueue = SafeQueue<ParsedSensorDataPtr, 10>;
	using FrameHistogramDataQueue = SafeQueue<FrameHistogramDataPtr, 10>;
	using ReconstructedImageDataQueue = SafeQueue<ReconstructedImageDataPtr, 10>;
	using KeyboardInputQueue = SafeQueue<char, 100>;
}