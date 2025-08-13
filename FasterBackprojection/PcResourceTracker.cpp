#include "stdafx.h"
#include "PcResourceTracker.h"

#include "CudaHelper.h"

const std::string PcResourceTracker::MeasurementStr[] = { "CPU", "RAM", "VIRTUAL_MEMORY", "GPU" };

// 

PcResourceTracker::PcResourceTracker()
{
	this->initCPUResources();
}

PcResourceTracker::~PcResourceTracker() = default;

void PcResourceTracker::print() const
{
	std::string summary;
	for (const Measurement& measurement: _measurements)
		measurement.write(summary);

	std::cout << summary << '\n';
	std::cout << "----------------------------------------" << '\n';
}

void PcResourceTracker::track(long waitMilliseconds)
{
	_interrupt = false;
	_thread = std::thread(&PcResourceTracker::threadedWatch, this, waitMilliseconds);
}

void PcResourceTracker::stop()
{
	_interrupt = true;
	_thread.join();
}

std::string PcResourceTracker::summarize(bool clear)
{
	std::string summary;
	for (const Measurement& measurement: _measurements)
		measurement.write(summary);
	if (clear)
		_measurements.clear();
	return summary;
}

//

void PcResourceTracker::initCPUResources()
{
#ifdef _WIN32
	SYSTEM_INFO sysInfo;
	FILETIME ftime, fsys, fuser;

	GetSystemInfo(&sysInfo);
	_numProcessors = sysInfo.dwNumberOfProcessors;

	GetSystemTimeAsFileTime(&ftime);
	memcpy(&_lastCPU, &ftime, sizeof(FILETIME));

	_self = GetCurrentProcess();
	GetProcessTimes(_self, &ftime, &ftime, &fsys, &fuser);
	memcpy(&_lastSysCPU, &fsys, sizeof(FILETIME));
	memcpy(&_lastUserCPU, &fuser, sizeof(FILETIME));
#endif
}

void PcResourceTracker::trackCpuUsage(Measurement& measurement)
{
#ifdef _WIN32
	FILETIME ftime, fsys, fuser;
	ULARGE_INTEGER now, sys, user;

	GetSystemTimeAsFileTime(&ftime);
	memcpy(&now, &ftime, sizeof(FILETIME));

	GetProcessTimes(_self, &ftime, &ftime, &fsys, &fuser);
	memcpy(&sys, &fsys, sizeof(FILETIME));
	memcpy(&user, &fuser, sizeof(FILETIME));

	ULONGLONG percent = sys.QuadPart - _lastSysCPU.QuadPart + (user.QuadPart - _lastUserCPU.QuadPart);
	percent = now.QuadPart - _lastCPU.QuadPart == 0 ? 0 : percent / (now.QuadPart - _lastCPU.QuadPart);
	percent = _numProcessors > 0 ? percent / _numProcessors : 0;

	_lastCPU = now;
	_lastUserCPU = user;
	_lastSysCPU = sys;

	constexpr ULONGLONG percentMultiplier = 100000;
	measurement._measurement[CPU] = percent * percentMultiplier;
	measurement._total[CPU] = percentMultiplier;
#endif
}

void PcResourceTracker::trackGpuMemory(Measurement& measurement)
{
	measurement._measurement[GPU] = CudaHelper::getAllocatedMemory() / 1024 / 1024;
	measurement._total[GPU] = CudaHelper::getMaxAllocatableMemory() / 1024;
}

void PcResourceTracker::trackMemoryUsage(Measurement& measurement)
{
#ifdef _WIN32
	MEMORYSTATUSEX memInfo;
	memInfo.dwLength = sizeof(MEMORYSTATUSEX);
	GlobalMemoryStatusEx(&memInfo);

	PROCESS_MEMORY_COUNTERS_EX pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));

	// Virtual memory
	DWORDLONG totalVirtualMemory = memInfo.ullTotalPageFile;
	SIZE_T virtualMemoryUsedByMe = pmc.PrivateUsage;

	// RAM
	DWORDLONG totalRAMMemory = memInfo.ullTotalPhys;
	SIZE_T RAMMemORYUsedByMe = pmc.WorkingSetSize;

	measurement._measurement[MeasurementType::RAM] = RAMMemORYUsedByMe / 1024 / 1024;
	measurement._total[MeasurementType::RAM] = totalRAMMemory / 1024 / 1024;

	measurement._measurement[MeasurementType::VIRTUAL_MEMORY] = virtualMemoryUsedByMe / 1024 / 1024;
	measurement._total[MeasurementType::VIRTUAL_MEMORY] = totalVirtualMemory / 1024 / 1024;
#endif
}

void PcResourceTracker::threadedWatch(long waitMilliseconds)
{
	Measurement measurement;
	while (!_interrupt)
	{
		trackCpuUsage(measurement);
		trackMemoryUsage(measurement);
		trackGpuMemory(measurement);
		_measurements.push_back(measurement);

		// Write measurements to file
		if (!_interrupt)
			std::this_thread::sleep_for(std::chrono::milliseconds(waitMilliseconds));
		else
			break;
	}
}