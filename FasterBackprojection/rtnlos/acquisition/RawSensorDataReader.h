#pragma once

#include "../types.h"
#include "../util/DataProcessor.h"
#include "../util/LogWriter.h"
#include "../util/cxxopts/cxxopts.hpp"

namespace NLOS {

    // stage 1: read raw data from the hardware.
    class RawSensorDataReader : public DataProcessor {
    public:
        RawSensorDataReader(RawSensorDataQueue& outgoing, LogWriter& logWriter)
            : DataProcessor("RawSensorDataReader", logWriter)
            , m_outgoingRaw(outgoing)
        {}

    protected:
        virtual void InitCmdLineOptions(cxxopts::Options& options);
        virtual void Work();
        virtual void OnStop();
    private:
        void ReadFromFile();

#if _WIN64
        void ReadFromDevice();
        int HardwareStart();
        void HardwareStop(int deviceId);
#endif

        std::string m_rawDataFileName; // if set to file name, read raw data from this file rather than from device
        RawSensorDataQueue& m_outgoingRaw;

        
#if _WIN64 // live hardware acquisition only available on windows
#endif
    };
}
