#pragma once

#include "TransientParameters.h"

class PostprocessingFilters
{
public:
	virtual void compute(float*& input, const glm::uvec3& size, const TransientParameters& transientParameters) const = 0;
};

class None : public PostprocessingFilters
{
public:
	void compute(float*& input, const glm::uvec3& size, const TransientParameters& transientParameters) const override
	{
		// No operation, just return the input as is
	}
};

class Laplacian : public PostprocessingFilters
{
public:
	void compute(float*& input, const glm::uvec3& size, const TransientParameters& transientParameters) const override;
};

class LoG : public PostprocessingFilters
{
protected:
	static std::vector<float> calculateLaplacianKernel(int size, float std1);

public:
	void compute(float*& input, const glm::uvec3& size, const TransientParameters& transientParameters) const override;
};

class LoGFFT: public PostprocessingFilters
{
public:
	void compute(float*& input, const glm::uvec3& size, const TransientParameters& transientParameters) const override;
};

