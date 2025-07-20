#pragma once

#include "stdafx.h"

#include "NLosData.h"

class TransientFileReader
{
public:
	virtual bool read(const std::string& filename, NLosData& nlosData) = 0;
};