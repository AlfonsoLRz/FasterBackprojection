#pragma once

#include "Singleton.h"

class PcResourceTracker 
{
public:
	enum MeasurementType: uint8_t { CPU, RAM, VIRTUAL_MEMORY, GPU, NUM_MEASUREMENT_TYPES };
	const static std::string MeasurementStr[MeasurementType::NUM_MEASUREMENT_TYPES];

	struct Measurement
	{
		size_t _measurement[MeasurementType::NUM_MEASUREMENT_TYPES];
		size_t _total[MeasurementType::NUM_MEASUREMENT_TYPES];

		void write(std::string& stream) const
		{
			for (uint8_t i = 0; i < NUM_MEASUREMENT_TYPES; ++i)
				stream += MeasurementStr[i] + ": " + std::to_string(_measurement[i]) + ' ' + std::to_string(_total[i]) + '\n';
		}
	};

protected:
	bool						_interrupt;
	std::thread					_thread;
	std::vector<Measurement>	_measurements;

	// CPU information
#ifdef _WIN32
	ULARGE_INTEGER				_lastCPU, _lastSysCPU, _lastUserCPU;
	int							_numProcessors;
	HANDLE						_self;
#endif

protected:
	void initCPUResources();

	void trackCpuUsage(Measurement& measurement);
	static void trackGpuMemory(Measurement& measurement);
	static void trackMemoryUsage(Measurement& measurement);
	void threadedWatch(long waitMilliseconds);

public:
	PcResourceTracker();
	~PcResourceTracker();

	void track(long waitMilliseconds = 20);
	void stop();

	void print() const;
	std::string summarize(bool clear = true);
};
