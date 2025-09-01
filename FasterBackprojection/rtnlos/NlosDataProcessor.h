#pragma once

#include "stdafx.h"
#include "InputManager.h"

namespace rtnlos
{
	class NlosDataProcessor: public WindowCloseListener
	{
	protected:
		bool			_stop = false;
		std::jthread	_workerThread;

	public:
		void windowCloseEvent() override;
	};

	inline void NlosDataProcessor::windowCloseEvent()
	{
		spdlog::info("{}: NLOS Data Processor: Window close event received, stopping processing.", typeid(*this).name());
		_stop = true;
	}
}
