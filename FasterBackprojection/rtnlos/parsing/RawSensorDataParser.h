#pragma once

#include "NlosDataProcessor.h"
#include "data/SceneParameters.h"
#include "data/SensorData.h"

namespace rtnlos
{
    // Stage 2: parse the raw records into grid indices and times, accumulating a full frame.
    template<int ROWS, int COLS>
    class RawSensorDataParser: public NlosDataProcessor
	{
    public:
        std::string                         _rawDataFileName; // If set to file name, read raw data from this file rather than from device
        RawSensorDataQueue&                 _incomingRaw;
        ParsedSensorDataQueue&              _outgoingParsed;

        // Initialized in Initialize() from the SceneParameters
        float                               _dsMultiplier;
        float                               _photonMinTime;
        float                               _photonMaxTime;
        std::vector<float>                  _indexDeltas;
        std::vector<uint32_t>               _binZeros;
        uint32_t                            _binLength;
        std::vector<std::vector<float>>     _channelDeltas;
        uint32_t                            _skipRate;
        uint32_t                            _numScannedRows;
        uint32_t                            _numScannedIndices;
        std::vector<int>                    _spadRowShift;

        uint32_t                            _indicesPerBin;
        uint32_t                            _centeringCoeff;
        uint32_t                            _offset;

        const uint32_t OVERFLOW_SIZE = 1024;
        const uint32_t NUM_INDICES = ROWS * COLS;

    public:
        RawSensorDataParser(RawSensorDataQueue& incoming, ParsedSensorDataQueue& outgoing)
	        : _incomingRaw(incoming), _outgoingParsed(outgoing),
    		  _dsMultiplier(0), _photonMinTime(0), _photonMaxTime(0), _binLength(0), _skipRate(0),
	          _numScannedRows(0),
	          _numScannedIndices(0),
	          _spadRowShift({9, 8, 8, 7, 7, 6, 5, 5, 4, 3, 3, 2, 2, 1}),
	          _indicesPerBin(0),
	          _centeringCoeff(0), _offset(0)
        {
        }

    public:
        void Initialize(const SceneParameters& params);
        void DoWork();
        void Stop() const;

    private:
        void Work() const;

        uint32_t SyncTimeToGridIndex(uint32_t sync_time) const
        {
            return (sync_time - _offset) / _indicesPerBin;
        }

        uint32_t RasterizeIndex(uint32_t bin, int shiftRowsBy) const
        {
            // Deal with the fact that odd numbered rows need to be reversed.
            int row = bin / COLS;
            int col = bin % COLS;

            int full_row = (row * _skipRate) * COLS;
            full_row += shiftRowsBy * COLS;

            if (row % 2 != 0)  // Odd numbered row, so it needs to be reversed
                bin = full_row + COLS - col - 1;
            else 
                bin = full_row + col;

            return bin;
        }
    };
}
