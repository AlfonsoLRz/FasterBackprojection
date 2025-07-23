#include "stdafx.h"
#include "CudaPerf.h"

#include "CudaHelper.h"

#include <pprint/pprint.hpp>

//

CudaPerf::CudaPerf(std::string algName): _algName(std::move(algName))
{
}

void CudaPerf::tic(const std::string& name)
{
	if (name.empty())
		_stageStartTime[_algName] = std::chrono::high_resolution_clock::now();
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

void CudaPerf::summarize()
{
	pprint::PrettyPrinter printer;

	// Beautifully print the timings
	std::cout << "CudaPerf Summary for algorithm: " << _algName << '\n';

	// Per stage timings
	for (auto& timing : _timings)
		timing.second /= 1000000; // ms
	printer.print(_timings);

	// Global elapsed time
	long long globalElapsedTime = std::accumulate(_timings.begin(), _timings.end(), long long{},
		[](const long long& sum, const std::pair<std::string, long long>& timing) {
			return sum + timing.second / 1000000; // ms
		});
	if (_timings.contains(_algName))
		globalElapsedTime = _timings[_algName];

	std::cout << std::setw(30) << std::left << "Total Time" << ": " << std::fixed << std::setprecision(6) << globalElapsedTime << " milliseconds\n";
}
