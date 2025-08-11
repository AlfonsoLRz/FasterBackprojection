#include "stdafx.h"
#include "RawSensorDataParser.h"
#include "../compile_time_constants.h"

namespace NLOS {

    template<int ROWS, int COLS>
    void RawSensorDataParser<ROWS, COLS>::Initialize(const SceneParameters& params)
    {
        m_ds_multiplier = 1.f / (1 << params.DownsamplingRate);
        m_skipRate = params.SkipRate;
        m_numScannedRows = (ROWS - 1) / m_skipRate + 1;
        m_numScannedIndices = m_numScannedRows * COLS;
        m_binZeros = params.BinZeros;
        m_binLength = params.BinLength;

        const float ts = params.Resolution * params.Picosecond / m_ds_multiplier;

        m_indices_per_bin = (uint32_t)(params.SyncRate / params.GalvoRate);
        m_centering_coeff = params.CenteringCoeff;
        m_offset = m_indices_per_bin * m_centering_coeff;

        m_photon_minT = params.Z_Gate * m_ds_multiplier;
        m_photon_maxT = std::round(2 * params.DepthMax / (ts * params.C_Light));

        m_index_deltas = params.D1;
        for (auto i = 0; i < params.D4.size(); i++) {
            m_channel_deltas.emplace_back(std::vector<float>(params.D4[i].size()));
            for (auto j = 0; j < params.D4[i].size(); j++) {
                m_channel_deltas[i][j] = -params.D4[i][j] + (params.T0_Gated[i][j] + params.Offset[i][j]) / params.DeltaT;
            }
        }
    }

    // This worker should receive T3Rec data from the incoming queue and
    // parse the records into indices and times, and watch for frame markers
    // when a full frame has been parsed, the indices and times should be 
    // pushed into the outgoing queue
    template<int ROWS, int COLS>
    void RawSensorDataParser<ROWS, COLS>::Work()
    {
        uint32_t frameNumber = 0;
        uint32_t oflcorrection = 0; // amount of time accumulated by overflows
        uint32_t startOffset = 0; // the nsync timer when the start marker was received
        bool isCurFrameLegit = false; // data before the first frame is garbage
        size_t largestFrame = 0; // to aid in pre-allocation of future frames for performance.

        ParsedSensorDataPtr pendingFrame(new ParsedSensorData(frameNumber));
        RawSensorDataPtr incomingData;

        int zero_cnt = 0;
        while (!m_stopRequested)
        {
            // get the next chunk of records
            if (!m_incomingRaw.Pop(incomingData)) {
                spdlog::critical("{:<25}: failed to receive raw data. Exiting.", m_name);
                break;
            }

            if (m_stopRequested)
                break;

            // special case if the provider is a file reader, it will reset. If that happens, we need to zero our overflows and frame numbers
            if (incomingData->FileReaderWasResetFlag) {
                spdlog::trace("{:<25}: FileReader was reset for frame {}. idx={}", m_name, frameNumber, pendingFrame->IndexData.size());
                uint32_t oflcorrection = 0; // amount of time accumulated by overflows
                uint32_t startOffset = 0; // the nsync timer when the start marker was received
                pendingFrame->ResetData();

                isCurFrameLegit = false;
            }

            for (int i = 0; i < incomingData->NumRecords; i++)
            {
                T3Rec& rec = incomingData->Records[i];
                if (rec.bits.special == 1) {
                    if (rec.bits.channel == 0x3F) { //overflow
                        oflcorrection += c_overflowSize * rec.bits.nsync;
                    }
                    if (rec.bits.channel == 1) { //marker # is stored in the channel number
                        if (isCurFrameLegit) {
                            largestFrame = std::max(pendingFrame->IndexData.size(), largestFrame);

                            //spdlog::trace("{:<25}: {:8.2f} ms Split ({}-{})", m_name, split, frameNumber, timer.Count());
                            //spdlog::debug(",{:<25},{},{:8.2f} ms to parse frame. Pushing {} records to outgoing queue (incoming queue size={}).", m_name, frameNumber, time, pendingFrame->IndexData.size(), m_incomingRaw.Size());
                            m_outgoingParsed.Push(pendingFrame);

                            // if we are logging parsed data, push it into the queue to be logged.
                            if (m_logWriter.LogParsedData()) {
                                m_logWriter.PushLog(pendingFrame);
                            }

                            pendingFrame.reset(new ParsedSensorData(++frameNumber, largestFrame));
                        }
                        else {
                            pendingFrame->ResetData();
                            isCurFrameLegit = true; // next frame will be legitimate
                        }
                        oflcorrection = 0;
                        startOffset = static_cast<uint32_t>(rec.bits.nsync);
                    }
                }
                else {
                    if (isCurFrameLegit) {
                        // adjust the nsync time by the number of overflows and the value that nsync was when the start marker was received
                        uint32_t sync_time = oflcorrection - startOffset + static_cast<uint32_t>(rec.bits.nsync);

                        // uncomment to only bin channel 1
                        //if (rec.bits.channel != 1 || rec.bits.dtime > 5000)
                        //	continue;

                        // convert the sync_time to the index on the grid
                        uint32_t grid_idx = SyncTimeToGridIndex(sync_time);

                        // make sure we're on the grid
                        if (grid_idx >= 0 && grid_idx < m_numScannedIndices) {
                            int logicalSpad = -1;
                            //for (int i = 0; i < m_binZeros.size() - 1; i++) {
                            //    if (rec.bits.dtime >= m_binZeros[i] && rec.bits.dtime < m_binZeros[i] + m_binLength) {
                            //        logicalSpad = i;
                            //        break;
                            //    }
                            //}
                            int tim = rec.bits.dtime;
                            int SpadID;
                            int SpadOffset;
                            if (tim <= 5000) {
                                logicalSpad = 0;
                                SpadID = 0;
                                SpadOffset = 0;
                            }
                            else if (tim <= 10000) {
                                logicalSpad = 1;
                                SpadID = 0;
                                SpadOffset = 1;
                            }
                            else if (tim >= 11250 && tim <= 16250) {
                                logicalSpad = 2;
                                SpadID = 1;
                                SpadOffset = 0;
                            }
                            else if (tim >= 17500 && tim <= 22500) {
                                logicalSpad = 3;
                                SpadID = 1;
                                SpadOffset = 1;
                            }

                            if (logicalSpad >= 0) {
                                int channel = (rec.bits.channel - 1) * 2 + SpadOffset; // 2 logical (SPAD) channels come in on same physical (TCSPC) channel
                                // uncomment and change sign to pick spad1 or spad2 only.
                                // (commented => use both SPADs together)
                                //if (rec.bits.dtime < 10000)
                                //    continue;
                                float channel_offset = -(float)(m_binZeros[logicalSpad]);

                                float adjustment = -m_index_deltas[grid_idx] + m_channel_deltas[SpadID][channel];
                                float dtime = static_cast<float>(rec.bits.dtime) + adjustment + channel_offset;
                                dtime = dtime * m_ds_multiplier; // downsample (todo: replace with bit-shift once everything is working)
                                if (dtime > m_photon_minT && dtime < m_photon_maxT) {
                                    // optionally, shift the index in the x direction to account for the offset off the spads
                                    int shift_right_by = m_spad_row_shift[channel];
                                    // RasterizeIndex() reverses odd numbered rows, and re-indexes for skipped rows
                                    int idx = RasterizeIndex(grid_idx, shift_right_by);

                                    if (idx < ROWS * COLS) {
                                        pendingFrame->IndexData.push_back(idx);
                                        pendingFrame->TimeData.push_back(dtime);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //spdlog::trace("{:<25}: {:8.2f} ms Split ({}-{})", m_name, split, frameNumber, timer.Count());
            //spdlog::trace("{:<25}: parsed {} recs for frame {} in {} msec. idx={}", m_name, incomingData->NumRecords, frameNumber, split, pendingFrame->IndexData.size());
        }
    }

    template<int ROWS, int COLS>
    void RawSensorDataParser<ROWS, COLS>::OnStop() {
        // there may be a push blocking if the queue is full, so tell the queue to abort its operation
        spdlog::trace("{:<25}: Abort", m_name);
        m_incomingRaw.Abort();
        m_outgoingParsed.Abort();
    }

    // explicit instantiation
    template class RawSensorDataParser<NUMBER_OF_ROWS, NUMBER_OF_COLS>;
}

