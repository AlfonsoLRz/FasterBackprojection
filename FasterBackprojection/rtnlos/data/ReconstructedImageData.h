#pragma once

#include <memory>
#include "../util/SafeQueue.h"
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "PipelineData.h"

namespace NLOS
{
	template<int NROWS, int NCOLS>
	class ReconstructedImageData : public PipelineData {
	public:
		ReconstructedImageData(uint32_t frameNumber = 0)
			: FrameNumber(frameNumber)
			, Image2d(cv::Size(NROWS, NCOLS), CV_32FC1)
		{ };
		ReconstructedImageData(uint32_t frameNumber, const cv::Mat& img)
			: FrameNumber(frameNumber)
			, Image2d(img)
		{ };

		cv::Mat Image2d;
		uint32_t FrameNumber;

		virtual void LogToFile(ILogContext* pContext) {
			try {
				if (FrameNumber > 1000) { // avoid needlessly filling disk
					//spdlog::warn("Already logged 1000 reconstructed images. Skipping logging to save disk space");
					return;
				}
				
				std::string fname = pContext->MakeLogFileName("Image", FrameNumber, pContext->ImageLogFormat());
				switch (pContext->ImageLogFormat()) {
				case LogFileFormat::Binary: {
					std::ofstream f(fname, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
					f.write((const char*)&FrameNumber, 4);
					int sz[] = { Image2d.rows, Image2d.cols };
					f.write((const char*)sz, 4 * 2);
					f.write((const char*)Image2d.data, Image2d.rows * Image2d.cols * 4);
					break;
				}
				case LogFileFormat::Yaml: {
					cv::FileStorage fs(fname, cv::FileStorage::WRITE);
					fs << "FrameNumber" << (int)FrameNumber;
					fs << "Image" << Image2d;
					break;
				}
				case LogFileFormat::Png: 
				case LogFileFormat::MonoPng: {
					cv::imwrite(fname, Image2d);
					break;
				}
				default:
					std::cout << std::endl;
					//spdlog::warn("Cannot write image to format {}.", Fmt2Str(pContext->ImageLogFormat()));
				}
			}
			catch (const std::exception & ex) {
				//spdlog::warn("Failed to write image frame {} to log file. Error was: {}", FrameNumber, ex.what());
			}
		}
	};
}