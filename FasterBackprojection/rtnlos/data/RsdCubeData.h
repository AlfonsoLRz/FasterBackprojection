#pragma once

#include <memory>
#include "../util/SafeQueue.h"
#include "opencv4/opencv2/core.hpp"
#include "PipelineData.h"

namespace NLOS
{
	template<int NROWS, int NCOLS>
	class RsdCubeData : public PipelineData {
	public:
		RsdCubeData(uint32_t frameNumber, std::unique_ptr<std::vector<float>> &rsdCube)
			: FrameNumber(frameNumber)
			, RsdCube(std::move(rsdCube))
		{ };

		std::unique_ptr<std::vector<float>> RsdCube;
		int NumDepths;
		uint32_t FrameNumber;

		virtual void LogToFile(ILogContext* pContext) {
			if (FrameNumber > 100) { // avoid needlessly filling disk
				spdlog::warn("Already logged 100 RSD cubes. Skipping logging to save disk space");
				return;
			}

			try {
				std::string fname = pContext->MakeLogFileName("RSD", FrameNumber, pContext->RsdLogFormat());

				switch (pContext->RsdLogFormat()) {
				case LogFileFormat::Binary: {
					std::ofstream f(fname, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
					f.write((const char*)&FrameNumber, 4);
					int nDepths = (int)RsdCube->size() / NROWS / NCOLS;
					int sz[] = { nDepths, NROWS, NCOLS };
					f.write((const char*)sz, 4 * 3);
					f.write((const char*)RsdCube->data(), (int)RsdCube->size() * 4);
					break;
				}
				case LogFileFormat::Yaml: {
					cv::FileStorage fs(fname, cv::FileStorage::WRITE);
					int nDepths = (int)RsdCube->size() / NROWS / NCOLS;
					fs << "FrameNumber" << (int)FrameNumber;
					for (int i = 0; i < nDepths; i++) {
						std::string slice = fmt::format("Slice_{:03}", i);
						int sz[] = { NROWS, NCOLS };
						float* p = RsdCube->data() + i * NROWS * NCOLS;
						cv::Mat sl(2, sz, CV_32FC1, p);
						fs << slice << sl;
					}
					break;
				}
				default:
					spdlog::warn("Cannot write RSD to format {}.", Fmt2Str(pContext->RsdLogFormat()));
				}
			}
			catch (const std::exception & ex) {
				spdlog::warn("Failed to write RSD frame {} to log file. Error was: {}", FrameNumber, ex.what());
			}
		}
	};
}