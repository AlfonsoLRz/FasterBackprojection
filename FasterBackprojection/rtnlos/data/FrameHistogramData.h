#pragma once

#include <memory>
#include <opencv4/opencv2/core.hpp>
#include "../util/SafeQueue.h"
#include "../util/ILogContext.h"
#include "PipelineData.h"

namespace NLOS
{
	template<int NINDICES, int NFREQUENCIES>
	class FrameHistogramData : public PipelineData {
	public:
		FrameHistogramData(uint32_t frameNumber = 0)
			: FrameNumber(frameNumber)
		{ }

		float Histogram[NINDICES * NFREQUENCIES * 2];
		uint32_t FrameNumber;

		virtual void LogToFile(ILogContext* pContext) {
			if (FrameNumber > 100) { // avoid needlessly filling disk
				//spdlog::warn("Already logged 100 FDH frames. Skipping logging to save disk space");
				return;
			}

			try {
				std::string fname = pContext->MakeLogFileName("FDH", FrameNumber, pContext->FdhLogFormat());

				switch (pContext->FdhLogFormat()) {
				case LogFileFormat::Binary: {
					std::ofstream f(fname, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
					f.write((const char*)&FrameNumber, 4);
					int sz[] = { NFREQUENCIES, NINDICES, 1 };
					f.write((const char*)sz, 4 * 3);
					f.write((const char*)Histogram, NINDICES * NFREQUENCIES * 2 * 4);
					break;
				}
				case LogFileFormat::Yaml: {
					cv::FileStorage fs(fname, cv::FileStorage::WRITE);
					fs << "FrameNumber" << (int)FrameNumber;
					int sz[] = { NFREQUENCIES, NINDICES, 1 };
					cv::Mat fdh(3, sz, CV_32FC2, Histogram);
					fs << "FDH" << fdh;
					break;
				}
				default:
					std::cout << "Cannot write FDH to format " << Fmt2Str(pContext->FdhLogFormat()) << std::endl;
					//spdlog::warn("Cannot write FDH to format {}.", Fmt2Str(pContext->FdhLogFormat()));
				}
			}
			catch (const std::exception & ex) {
				//spdlog::warn("Failed to write FDH frame {} to log file. Error was: {}", FrameNumber, ex.what());
			}
		};
	};
}