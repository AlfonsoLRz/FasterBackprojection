#pragma once

//#define SYNCHRONIZE_KERNELS

class PcResourceTracker;

class CudaPerf
{
	typedef std::chrono::time_point<std::chrono::high_resolution_clock> Chrono;
	typedef std::chrono::duration<double> ElapsedTime;
	constexpr static long long TimeMeasurementDivisor = 1000000;

protected:
	std::string							_algName;

	// Timing events
	std::map<std::string, Chrono>		_stageStartTime;				// Start times for different stages
	std::queue<std::string>				_stageNames;					// Queue to keep track of stage names for timing
	std::map<std::string, long long>	_timings;						// Timings for different stages

	// Other resources from the pc
	PcResourceTracker*					_pcResourceTracker = nullptr;	// Pointer to the resource tracker	

public:
	CudaPerf(bool trackResources = true);
	~CudaPerf();

	void tic(const std::string& name = "");
	void toc(const std::string& name = "");

	void setAlgorithmName(const std::string& algName) { _algName = algName; }
	void summarize() const;
	void write(const std::string& outPath) const;
};

