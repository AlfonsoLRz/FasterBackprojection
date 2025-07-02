#pragma once

#include "stdafx.h"

namespace ChronoUtilities
{
	//!< Units we can use to return the measured time
	enum TimeUnit: int
	{
		SECONDS = 1000000000, MILLISECONDS = 1000000, MICROSECONDS = 1000, NANOSECONDS = 1
	};

	namespace
	{
		std::chrono::high_resolution_clock::time_point _initTime;
	}

	long long getElapsedTime(const TimeUnit timeUnit = ChronoUtilities::MILLISECONDS);

	void startTimer();
}

inline long long ChronoUtilities::getElapsedTime(const TimeUnit timeUnit)
{
	std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
	long long measuredTime = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - ChronoUtilities::_initTime).count();
	
	return measuredTime / timeUnit;
}

inline void ChronoUtilities::startTimer()
{
	ChronoUtilities::_initTime = std::chrono::high_resolution_clock::now();
}
