#include "stdafx.h"
#include "RawSensorDataParser.h"

namespace rtnlos
{
    template<int ROWS, int COLS>
    void RawSensorDataParser<ROWS, COLS>::Initialize(const SceneParameters& params)
    {
        _dsMultiplier = 1.f / (1 << params._downsamplingRate);
        _skipRate = params._skipRate;
        _numScannedRows = (ROWS - 1) / _skipRate + 1;
        _numScannedIndices = _numScannedRows * COLS;
        _binZeros = params._binZeros;
        _binLength = params._binLength;

        const float ts = params._resolution * params.PICOSECOND / _dsMultiplier;

        _indicesPerBin = static_cast<uint32_t>(params._syncRate / params._galvoRate);
        _centeringCoeff = params._centeringCoeff;
        _offset = _indicesPerBin * _centeringCoeff;

        _photonMinTime = params._zGate * _dsMultiplier;
        _photonMaxTime = std::round(2 * params._depthMax / (ts * params.LIGHT_SPEED));

        _indexDeltas = params._d1;
        for (auto i = 0; i < params._d4.size(); i++) 
        {
            _channelDeltas.emplace_back(std::vector<float>(params._d4[i].size()));
            for (auto j = 0; j < params._d4[i].size(); j++) 
                _channelDeltas[i][j] = -params._d4[i][j] + (params._t0Gated[i][j] + params._offset[i][j]) / params._deltaT;
        }
    }

    template <int ROWS, int COLS>
    void RawSensorDataParser<ROWS, COLS>::DoWork()
    {
		spdlog::info("Starting RawSensorDataParser worker thread");
		_workerThread = std::jthread(&RawSensorDataParser<ROWS, COLS>::Work, this);
    }

    // This worker should receive T3Rec data from the incoming queue and
    // parse the records into indices and times, and watch for frame markers
    // when a full frame has been parsed, the indices and times should be 
    // pushed into the outgoing queue
    template<int ROWS, int COLS>
    void RawSensorDataParser<ROWS, COLS>::Work() const
    {
        uint32_t frameNumber = 0;
        uint32_t oflCorrection = 0;             // Amount of time accumulated by overflows
        uint32_t startOffset = 0;               // The nsync timer when the start marker was received
        bool isCurrentFrameLegit = false;       // Data before the first frame is garbage
        size_t largestFrame = 0;                // To aid in pre-allocation of future frames for performance.

        ParsedSensorDataPtr pendingFrame(new ParsedSensorData(frameNumber));
        RawSensorDataPtr incomingData;

        while (!_stop)
        {
            // Get the next chunk of records
            if (!_incomingRaw.Pop(incomingData)) 
            {
                spdlog::critical("Failed to receive raw data. Exiting.");
                break;
            }

            // Special case if the provider is a file reader, it will reset. If that happens, we need to zero our overflows and frame numbers
            if (incomingData->_fileReaderWasResetFlag) 
            {
                spdlog::trace("FileReader was reset for frame {}. idx={}", frameNumber, pendingFrame->_indexData.size());
                pendingFrame->ResetData();

                isCurrentFrameLegit = false;
            }

            for (int i = 0; i < incomingData->_numRecords; i++)
            {
                spdlog::stopwatch sw;

                T3Rec& rec = incomingData->_records[i];
                if (rec.bits.special == 1) 
                {
                    if (rec.bits.channel == 0x3F)  // Overflow
                        oflCorrection += OVERFLOW_SIZE * rec.bits.nsync;
 
                    if (rec.bits.channel == 1) // Marker # is stored in the channel number
                    { 
                        if (isCurrentFrameLegit) 
                        {
                            largestFrame = std::max(pendingFrame->_indexData.size(), largestFrame);

                            //spdlog::debug(",{},{:8.2f} ms to parse frame. Pushing {} records to outgoing queue (incoming queue size={}).", frameNumber, 0.0f, pendingFrame->_indexData.size(), _incomingRaw.Size());
                            _outgoingParsed.Push(pendingFrame);
                            pendingFrame.reset(new ParsedSensorData(++frameNumber, largestFrame));
                        }
                        else 
                        {
                            pendingFrame->ResetData();
                            isCurrentFrameLegit = true; // next frame will be legitimate
                        }

                        oflCorrection = 0;
                        startOffset = rec.bits.nsync;
                    }
                }
                else 
                {
                    if (isCurrentFrameLegit) 
                    {
                        // Adjust the nsync time by the number of overflows and the value that nsync was when the start marker was received
                        uint32_t syncTime = oflCorrection - startOffset + rec.bits.nsync;

                        // Convert the sync_time to the index on the grid
                        uint32_t gridIdx = SyncTimeToGridIndex(syncTime);

                        // Make sure we're on the grid
                        if (gridIdx >= 0 && gridIdx < _numScannedIndices) 
{
                            int logicalSpad = -1, time = rec.bits.dtime, spadID, spadOffset;

                            if (time <= 5000) 
                            {
                                logicalSpad = 0;
                                spadID = 0;
                                spadOffset = 0;
                            }
                            else if (time <= 10000) 
                            {
                                logicalSpad = 1;
                                spadID = 0;
                                spadOffset = 1;
                            }
                            else if (time >= 11250 && time <= 16250) 
                            {
                                logicalSpad = 2;
                                spadID = 1;
                                spadOffset = 0;
                            }
                            else if (time >= 17500 && time <= 22500) 
                            {
                                logicalSpad = 3;
                                spadID = 1;
                                spadOffset = 1;
                            }

                            if (logicalSpad >= 0) 
                            {
                                int channel = (rec.bits.channel - 1) * 2 + spadOffset; // 2 logical (SPAD) channels come in on same physical (TCSPC) channel
                                float channel_offset = -static_cast<float>(_binZeros[logicalSpad]);

                                float adjustment = -_indexDeltas[gridIdx] + _channelDeltas[spadID][channel];
                                float dtime = static_cast<float>(rec.bits.dtime) + adjustment + channel_offset;
                                dtime = dtime * _dsMultiplier; // Downsample (todo: replace with bit-shift once everything is working)

                                if (dtime > _photonMinTime && dtime < _photonMaxTime) 
                                {
                                    // Optionally, shift the index in the x direction to account for the offset off the spads
                                    int shift_right_by = _spadRowShift[channel];
                                    // RasterizeIndex() reverses odd numbered rows, and re-indexes for skipped rows
                                    int idx = RasterizeIndex(gridIdx, shift_right_by);

                                    if (idx < ROWS * COLS) {
                                        pendingFrame->_indexData.push_back(idx);
                                        pendingFrame->_timeData.push_back(dtime);
                                    }
                                }
                            }
                        }
                    }
                }

				spdlog::info("Parsed record {} of frame {} in {:8.2f} ms", i, frameNumber, sw.elapsed().count() * 1000.0f);
            }
        }
    }

    template<int ROWS, int COLS>
    void RawSensorDataParser<ROWS, COLS>::Stop() const
    {
        _incomingRaw.Abort();
        _outgoingParsed.Abort();
    }

    // Explicit instantiation
    template class RawSensorDataParser<NUMBER_OF_ROWS, NUMBER_OF_COLS>;
}

