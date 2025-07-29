#pragma once

#include "stdafx.h"

#include "NLosData.h"

class TransientFileReader
{
public:
	virtual ~TransientFileReader() = default;
	virtual bool read(const std::string& filename, NLosData& nlosData) = 0;
};