#pragma once

#include "../types.h"
#include "../data/SceneParameters.h"
#include "../util/DataProcessor.h"
#include "../util/cxxopts/cxxopts.hpp"
#include "../util/LogWriter.h"

namespace NLOS {

    // stage 2: parse the raw records into grid indices and times, accumulating a full frame.
    template<int ROWS, int COLS>
    class RawSensorDataParser : public DataProcessor {
    public:
        RawSensorDataParser(RawSensorDataQueue& incoming, ParsedSensorDataQueue& outgoing, LogWriter& logWriter)
            : DataProcessor("RawSensorDataParser", logWriter)
            , m_incomingRaw(incoming)
            , m_outgoingParsed(outgoing)
            , m_spad_row_shift({ 9, 8, 8, 7, 7, 6, 5, 5, 4, 3, 3, 2, 2, 1 })
        {}

    protected:
        virtual void Initialize(const SceneParameters& params);
        virtual void Work();
        virtual void OnStop();
    private:
        void OnFrameMarker(ParsedSensorDataPtr& pendingFrame)
        {
        }

        uint32_t SyncTimeToGridIndex(uint32_t sync_time) {
            return (sync_time - m_offset) / m_indices_per_bin;
        }

        uint32_t RasterizeIndex(uint32_t bin, int shift_rows_by) {
            // deal with the fact that odd numbered rows need to be reversed.
            int row = bin / COLS;
            int col = bin % COLS;

            int full_row = (row * m_skipRate) * COLS;
            full_row += shift_rows_by * COLS;

            if (row % 2 != 0) { // odd numbered row, so it needs to be reversed
                bin = full_row + COLS - col - 1;
            }
            else {
                bin = full_row + col;
            }

            return bin;
        }

        std::string m_rawDataFileName; // if set to file name, read raw data from this file rather than from device
        RawSensorDataQueue& m_incomingRaw;
        ParsedSensorDataQueue& m_outgoingParsed;

        // initialized in Initialize() from the SceneParameters
        float m_ds_multiplier;
        float m_photon_minT;
        float m_photon_maxT;
        std::vector<float> m_index_deltas;
        std::vector<uint32_t> m_binZeros;
        uint32_t m_binLength;
        std::vector<std::vector<float>> m_channel_deltas;
        uint32_t m_skipRate;
        uint32_t m_numScannedRows;
        uint32_t m_numScannedIndices;
        std::vector<int> m_spad_row_shift;

        uint32_t m_indices_per_bin;
        uint32_t m_centering_coeff;
        uint32_t m_offset;
        const uint32_t c_overflowSize = 1024;
        const uint32_t NUM_INDICES = ROWS * COLS;
    };
}
