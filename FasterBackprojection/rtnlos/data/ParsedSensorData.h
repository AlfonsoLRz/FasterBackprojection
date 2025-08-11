#pragma once

#include <memory>
#include "opencv4/opencv2/core.hpp"
#include "../util/SafeQueue.h"
#include "../util/ILogContext.h"
#include "T3Rec.h"
#include "PipelineData.h"


namespace NLOS
{
	class ParsedSensorData : public PipelineData {
	public:
		ParsedSensorData(int frameNumber = 0, size_t hintSize = 0) 
			: FrameNumber(frameNumber)
		{
			IndexData.reserve(hintSize);
			TimeData.reserve(hintSize);
		}

		void ResetData() { IndexData.clear(); TimeData.clear(); }

		std::vector<uint32_t> IndexData;
		std::vector<float> TimeData;
		int FrameNumber;

		virtual void LogToFile(ILogContext* pContext) {
			if (FrameNumber > 100) { // avoid needlessly filling disk
				spdlog::warn("Already logged 100 parsed frames. Skipping logging to save disk space");
				return;
			}

			try {
				std::string fname = pContext->MakeLogFileName("Parsed", FrameNumber, pContext->ParsedLogFormat());

				switch (pContext->ParsedLogFormat()) {
				case LogFileFormat::Binary: {
					std::ofstream f(fname, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
					f.write((const char*)&FrameNumber, 4);
					int numRecs = (int)IndexData.size();
					f.write((const char*)&numRecs, 4);
					f.write((const char*)IndexData.data(), IndexData.size() * 4);
					f.write((const char*)TimeData.data(), IndexData.size() * 4);
					break;
				}
				case LogFileFormat::Yaml: {
					cv::FileStorage fs(fname, cv::FileStorage::WRITE);
					fs << "FrameNumber" << FrameNumber;
					cv::Mat idx(1, { (int)IndexData.size() }, CV_32S, IndexData.data());
					fs << "IndexData" << idx;
					cv::Mat tm(1, { (int)TimeData.size() }, CV_32F, TimeData.data());
					fs << "TimeData" << tm;
					break;
				}
				default:
					spdlog::warn("Cannot write Parsed frame to format {}.", Fmt2Str(pContext->ParsedLogFormat()));

				}
			}
			catch (const std::exception & ex) {
				spdlog::warn("Failed to write parsed records for frame {} to log file. Error was: {}", FrameNumber, ex.what());
			}
		};
	};
	


}