#pragma once

#include "TransientFileReader.h"

class MatLCTReader: public TransientFileReader
{
protected:
	static glm::uint getZOffset(const std::string& filename);

public:
	bool read(const std::string& filename, NLosData& nlosData) override;
};

