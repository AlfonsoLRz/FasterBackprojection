#include "stdafx.h"
#include "CudaPerf.h"

#include "CudaHelper.h"
#include "PcResourceTracker.h"

#include <pprint/pprint.hpp>

//

CudaPerf::CudaPerf(bool trackResources)
{
	if (trackResources)
		_pcResourceTracker = new PcResourceTracker;
}

CudaPerf::~CudaPerf()
{
	delete _pcResourceTracker;
}

void CudaPerf::tic(const std::string& name)
{
	if (name.empty())
	{
		_stageStartTime[_algName] = std::chrono::high_resolution_clock::now();
		if (_pcResourceTracker)
			_pcResourceTracker->track();
	}
	else
	{
		_stageStartTime[name] = std::chrono::high_resolution_clock::now();
		_stageNames.push(name);
	}
}

void CudaPerf::toc(const std::string& name)
{
	std::string key = name;
	if (name.empty())
	{
		if (!_stageNames.empty())
		{
			std::string stageName = _stageNames.front();
			_stageNames.pop();
			key = stageName;
		}
		else
		{
			key = _algName;
			if (_pcResourceTracker)
				_pcResourceTracker->stop();
		}
	}

#ifdef SYNCHRONIZE_KERNELS
	CudaHelper::synchronize(key);
#endif

	auto startTimeIt = _stageStartTime.find(key);
	std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
	long long elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTimeIt->second).count();

	_timings[key] += elapsedTime;
}

void CudaPerf::summarize() const
{
	pprint::PrettyPrinter printer;

	// Beautifully print the timings
	std::cout << "CudaPerf Summary for algorithm: " << _algName << '\n';

	// Per stage timings
	auto timings = _timings;
	for (auto& timing : timings)
		timing.second /= TimeMeasurementDivisor; // ms
	printer.print(timings);

	// Global elapsed time
	long long globalElapsedTime;
	_timings.contains(_algName) ? 
		globalElapsedTime = _timings.at(_algName) / TimeMeasurementDivisor :
		globalElapsedTime = std::accumulate(_timings.begin(), _timings.end(), long long{},
		[](const long long& sum, const std::pair<std::string, long long>& timing) {
			return sum + timing.second / TimeMeasurementDivisor; // ms
		});

	std::cout << std::left << "Total Time" << ": " << globalElapsedTime << " milliseconds\n";
}

void CudaPerf::write(const std::string& outPath) const
{
	std::ofstream outFile(outPath);
	if (!outFile.is_open())
	{
		std::cerr << "Failed to open output file: " << outPath << '\n';
		return;
	}

	// Stage and global timings
	std::string timingStr;

	auto timings = _timings;
	for (auto& timing : timings)
		timing.second /= TimeMeasurementDivisor; // ms

	for (const auto& timing : timings)
		timingStr += timing.first + ": " + std::to_string(timing.second) + " ms\n";

	// Global elapsed time
	timingStr += "Total: ";
	_timings.contains(_algName) ?
		timingStr += std::to_string(_timings.at(_algName) / TimeMeasurementDivisor):
		timingStr += std::accumulate(_timings.begin(), _timings.end(), long long{},
			[](const long long& sum, const std::pair<std::string, long long>& timing) {
				return sum + timing.second / TimeMeasurementDivisor; // ms
			});

	// Resource tracker summary
	if (_pcResourceTracker)
	{
		std::string resourceInfo = _pcResourceTracker->summarize();
		outFile << resourceInfo;
	}
}