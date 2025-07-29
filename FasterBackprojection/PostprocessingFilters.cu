#include "stdafx.h"
#include "PostprocessingFilters.h"

#include <cufft.h>

#include "CudaHelper.h"
#include "fourier.cuh"
#include "transient_postprocessing.cuh"

//

void LoG::compute(float*& input, const glm::uvec3& size, const TransientParameters& transientParameters) const
{
	std::vector<float> kernel = calculateLaplacianKernel(transientParameters._kernelSize, transientParameters._sigma);

	float* dKernel = nullptr, * dOutput = nullptr;
    CudaHelper::initializeBuffer(dKernel, kernel.size(), kernel.data());
	CudaHelper::initializeZeroBuffer(dOutput, size.x * size.y * size.z * sizeof(float));

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((size.x + blockSize.x - 1) / blockSize.x,
                  (size.y + blockSize.y - 1) / blockSize.y,
				  (size.z + blockSize.z - 1) / blockSize.z);

    LoGFilter <<<gridSize, blockSize>>>(input, dOutput, size, dKernel, transientParameters._kernelSize, (transientParameters._kernelSize - 1) / 2);
    CudaHelper::synchronize("filterLaplacianKernel");

	// Copy the output back to the input reference
	std::swap(input, dOutput);
	CudaHelper::free(dKernel);
	CudaHelper::free(dOutput);
}

//

void LoGFFT::compute(float*& input, const glm::uvec3& size, const TransientParameters& transientParameters) const
{
    // Calculate sizes
    glm::uint numVoxels = size.x * size.y * size.z;
    glm::uint complexDimensionSize = (size.x / 2 + 1) * size.y * size.z;

    cufftComplex* fourierReconstruction, * kernel;
    CudaHelper::initializeZeroBuffer(fourierReconstruction, complexDimensionSize);
    CudaHelper::initializeZeroBuffer(kernel, complexDimensionSize);

    cufftHandle forward_plan, inverse_plan;
    cufftPlan3d(&forward_plan, size.x, size.y, size.z, CUFFT_R2C);
    cufftPlan3d(&inverse_plan, size.x, size.y, size.z, CUFFT_C2R);

    // Forward FFT
    CUFFT_CHECK(cufftExecR2C(forward_plan, (cufftReal*)input, fourierReconstruction));

    // Build LoG kernel - must respect complex_size layout!
    dim3 block(8, 8, 8);
    dim3 grid(
        (size.x / 2 + 1 + block.x - 1) / block.x,
        (size.y + block.y - 1) / block.y,
        (size.z + block.z - 1) / block.z
    );
    buildLoGKernel3D<<<grid, block>>>(kernel, size.x, size.y, size.z, transientParameters._sigma);
    CudaHelper::synchronize("buildLoGKernel3D");

    // Multiply in frequency domain
    glm::uint threads = 256;
    glm::uint blocks = CudaHelper::getNumBlocks(complexDimensionSize, threads);
    multiplyKernel<<<blocks, threads>>>(fourierReconstruction, kernel, complexDimensionSize);
    CUFFT_CHECK(cufftExecC2R(inverse_plan, fourierReconstruction, (cufftReal*)input));

    // Normalize
	//normalizeIFFT<<<blocks, threads >>>(input, numVoxels);        // Again, values are so small; I don't think this is necessary

    // Cleanup
    cufftDestroy(forward_plan);
    cufftDestroy(inverse_plan);
    CudaHelper::free(fourierReconstruction);
    CudaHelper::free(kernel);
}

//

void Laplacian::compute(float*& input, const glm::uvec3& size, const TransientParameters& transientParameters) const
{
    float* dOutput = nullptr;
    CudaHelper::initializeZeroBuffer(dOutput, static_cast<size_t>(size.x) * size.y * size.z);

    dim3 blockSize = dim3(8, 8, 8);
    dim3 gridSize = dim3(
        (size.x + blockSize.x - 1) / blockSize.x,
        (size.y + blockSize.y - 1) / blockSize.y,
        (size.z + blockSize.z - 1) / blockSize.z
    );

    laplacianFilter<<<gridSize, blockSize>>>(input, dOutput, size, transientParameters._kernelSize);
	CudaHelper::synchronize("laplacianFilter");

    std::swap(input, dOutput);
    CudaHelper::free(dOutput);
}

//

std::vector<float> LoG::calculateLaplacianKernel(int size, float std1)
{
    int lim = (size - 1) / 2; // Half-size of the kernel (e.g., 2 for size=5)
    float std2 = std1 * std1;

    // Vectors to store coordinates for each kernel element
    std::vector<float> x_coords(size * size * size);
    std::vector<float> y_coords(size * size * size);
    std::vector<float> z_coords(size * size * size);

    // Step 1: Generate grid coordinates and initial Gaussian weights
    std::vector<float> w_initial(size * size * size);
    float maxW_initial = 0.0f;

    for (int k_z = 0; k_z < size; ++k_z) {
        for (int k_y = 0; k_y < size; ++k_y) {
            for (int k_x = 0; k_x < size; ++k_x) {
                float x = static_cast<float>(k_x - lim);
                float y = static_cast<float>(k_y - lim);
                float z = static_cast<float>(k_z - lim);

                int idx = k_z * size * size + k_y * size + k_x; // 1D index for the 3D kernel
                x_coords[idx] = x;
                y_coords[idx] = y;
                z_coords[idx] = z;

                // Calculate Gaussian weight
                w_initial[idx] = expf(-(x * x + y * y + z * z) / (2.0f * std2));
                maxW_initial = std::max(w_initial[idx], maxW_initial);
            }
        }
    }

    // Step 2: Apply thresholding (w < eps * max(w(:))) and sum for normalization
    const float threshold = glm::epsilon<float>() * maxW_initial;
    float sum_w = 0.0f;

    for (float& i : w_initial)
    {
        if (i < threshold)
            i = 0.0f;
        sum_w += i;
    }

    // Step 3: Normalize w_initial
    if (sum_w != 0.0f)
    {
        for (float& i : w_initial)
            i /= sum_w;
    }

    // Step 4: Calculate Laplacian of Gaussian part 
    std::vector<float> w1(size * size * size);
    float sum_w1 = 0.0f;

    for (int i = 0; i < w_initial.size(); ++i)
    {
        float x = x_coords[i];
        float y = y_coords[i];
        float z = z_coords[i];

        // LoG formula: w * (r^2 - 3*sigma^2) / sigma^4
        w1[i] = w_initial[i] * (x * x + y * y + z * z - 3.0f * std2) / (std2 * std2);
        sum_w1 += w1[i];
    }

    // Step 5: Make the filter sum to zero (final `w` in MATLAB)
    std::vector<float> w_final(size * size * size);
    float adjustment = sum_w1 / static_cast<float>(size * size * size); // Average value to subtract
    for (int i = 0; i < w_final.size(); ++i)
        w_final[i] = w1[i] - adjustment;

    return w_final;
}
