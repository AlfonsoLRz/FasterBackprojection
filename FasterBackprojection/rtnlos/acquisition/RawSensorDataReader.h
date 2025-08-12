#pragma once

#include "data/SensorData.h"
#include "NlosDataProcessor.h"

namespace rtnlos
{
	class RawSensorDataReader : public NlosDataProcessor
	{
    protected:
		std::string         _dataPath; 
        RawSensorDataQueue& _outgoingRaw;

    public:
        RawSensorDataReader(const std::string& dataPath, RawSensorDataQueue& outgoing);
        void DoWork();
        void Stop();

    private:
        void ReadFromFile(const std::string& dataPath) const;
    };
}
