#pragma once

#include "TransientFileReader.h"

class MatFkReader: public TransientFileReader
{
public:
	bool read(const std::string& filename, NLosData& nlosData) override;
};

