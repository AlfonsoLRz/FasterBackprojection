#pragma once

#include "cufft.h"
#include "opencv4/opencv2/core.hpp"
#include "util/SafeQueue.h"

#define NUMBER_OF_SPAD_ROWS 190
#define NUMBER_OF_SPAD_COLS 190
#define NUMBER_OF_SPAD_FREQUENCIES 208
#define RAW_SENSOR_READ_BLOCK_SIZE 131072 // copied from TTREADMAX

namespace rtnlos
{
	union T3Rec
	{
		uint32_t _allBits;
		struct
		{
			unsigned nsync : 10;
			// last 10 bits, number of sync period
			// this is 2 things:
			// a : for an overflow it's how many overflows since last
			// photon / marker
			// b : for photon / marker it's how many syncs since last overflow
			// | | | | | | | | | | | | | | | | | | | | | | |x|x|x|x|x|x|x|x|x|x|

			unsigned dtime : 15;
			// next 15 bits, delay from last sync in units of chosen resolution
			// the dtime unit depends on "_resolution" that can be obtained from header
			// DTime: Arrival time(units) of Photon after last Sync event
			// DTime* _resolution = Real time arrival of Photon after last Sync event
			// | | | | | | | |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| | | | | | | | | | |

			unsigned channel : 6;
			// next 6 bits, for photons, channel 0~7, for overflows, channel = 63, for markers, special = 1, channel 1~4
			//| |x|x|x|x|x|x| | | | | | | | | | | | | | | | | | | | | | | | | |

			unsigned special : 1;
			// first bit: special = 1 indicates marker or overflows
		} bits;
	};

	template<int NUM_RECORDS>
	class RawSensorData
	{
	public:
		int		_numRecords;
		T3Rec	_records[NUM_RECORDS];
		bool	_fileReaderWasResetFlag;

	public:
		RawSensorData() : _numRecords(0), _records{}, _fileReaderWasResetFlag(false) {}
	};

	class ParsedSensorData
	{
	public:
		std::vector<uint32_t>	_indexData;
		std::vector<float>		_timeData;
		int						_frameNumber;

	public:
		ParsedSensorData(int frameNumber = 0, size_t hintSize = 0) : _frameNumber(frameNumber)
		{
			_indexData.reserve(hintSize);
			_timeData.reserve(hintSize);
		}

		void ResetData() { _indexData.clear(); _timeData.clear(); }
	};

	template<int NINDICES, int NFREQUENCIES>
	class FrameHistogramData
	{
	public:
		cufftComplex	_histogram[NINDICES * NFREQUENCIES * 2];
		uint32_t		_frameNumber;

	public:
		FrameHistogramData(uint32_t frameNumber = 0) : _histogram{}, _frameNumber(frameNumber)
		{
		}
	};

	template<int NROWS, int NCOLS>
	class ReconstructedImageData
	{
	public:
		cv::Mat		_image;
		uint32_t	_frameNumber;

	public:
		ReconstructedImageData(uint32_t frameNumber = 0) : _image(cv::Size(NROWS, NCOLS), CV_32FC1), _frameNumber(frameNumber) {}
		ReconstructedImageData(uint32_t frameNumber, cv::Mat img) : _image(std::move(img)), _frameNumber(frameNumber) {}
	};

	using RawSensorDataType = RawSensorData<RAW_SENSOR_READ_BLOCK_SIZE>;
	using FrameHistogramDataType = FrameHistogramData<NUMBER_OF_SPAD_ROWS* NUMBER_OF_SPAD_COLS, NUMBER_OF_SPAD_FREQUENCIES>;
	using ReconstructedImageDataType = ReconstructedImageData<NUMBER_OF_SPAD_ROWS, NUMBER_OF_SPAD_COLS>;

	using RawSensorDataPtr = std::shared_ptr<RawSensorDataType>;
	using ParsedSensorDataPtr = std::shared_ptr<ParsedSensorData>;
	using FrameHistogramDataPtr = std::shared_ptr<FrameHistogramDataType>;
	using ReconstructedImageDataPtr = std::shared_ptr<ReconstructedImageDataType>;

	using RawSensorDataQueue = SafeQueue<RawSensorDataPtr, 200>;
	using ParsedSensorDataQueue = SafeQueue<ParsedSensorDataPtr, 10>;
	using FrameHistogramDataQueue = SafeQueue<FrameHistogramDataPtr, 10>;
	using ReconstructedImageDataQueue = SafeQueue<ReconstructedImageDataPtr, 10>;
	using KeyboardInputQueue = SafeQueue<char, 100>;
}