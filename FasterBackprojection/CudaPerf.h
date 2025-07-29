#pragma once

#define SYNCHRONIZE_KERNELS

class CudaPerf
{
	typedef std::chrono::time_point<std::chrono::high_resolution_clock> Chrono;
	typedef std::chrono::duration<double> ElapsedTime;

protected:
	std::string _algName;

	cudaEvent_t							_startEvent, _stopEvent; // CUDA events for timing the whole algorithm
	std::map<std::string, Chrono>		_stageStartTime;	// Start times for different stages
	std::queue<std::string>				_stageNames;		// Queue to keep track of stage names for timing
	std::map<std::string, long long>	_timings;			// Timings for different stages

public:
	CudaPerf(std::string algName = "");

	void tic(const std::string& name = "");
	void toc(const std::string& name = "");

	void setAlgorithmName(const std::string& algName) { _algName = algName; }

	void summarize();
};

