#include "stdafx.h"
#include "MatFkReader.h"

#include "math.cuh"

//

bool MatFkReader::read(const std::string& filename, NLosData& nlosData)
{
	mat_t* dataMatFile = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
	if (!dataMatFile)
		throw std::runtime_error("NLosData: Failed to load LCT data from .mat file.");

	matvar_t* dataVar = Mat_VarRead(dataMatFile, "meas");
	if (!dataVar)
	{
		Mat_Close(dataMatFile);
		return false;
	}
	float* rectData = static_cast<float*>(dataVar->data);

	// Dimensions of data
	nlosData._dims.resize(dataVar->rank);
	for (int i = 0; i < dataVar->rank; ++i)
		nlosData._dims[i] = dataVar->dims[i];

	// Extract crop size for z dimension
	glm::uint crop = 512;
	if (matvar_t* cropSizeVar = Mat_VarRead(dataMatFile, "crop"))
	{
		crop = static_cast<glm::uint*>(dataVar->data)[0];
		Mat_VarFree(cropSizeVar);
	}

	//
	size_t globalSize = 1;
	for (size_t i = 0; i < nlosData._dims.size(); ++i)
		globalSize *= dataVar->dims[i];

	// Fill data 
	nlosData._data.resize(globalSize);
	#pragma omp parallel for
	for (size_t x = 0; x < nlosData._dims[0]; ++x)
	{
		for (size_t y = 0; y < nlosData._dims[1]; ++y)
		{
			for (size_t t = 0; t < nlosData._dims[2]; ++t)
			{
				size_t sourceIdx = t * nlosData._dims[0] * nlosData._dims[1] + y * nlosData._dims[0] + x;
				size_t destIdx = y * (nlosData._dims[0] * nlosData._dims[2]) + x * nlosData._dims[2] + t;
				nlosData._data[destIdx] = rectData[sourceIdx];
			}
		}
	}

	// Bin resolution
	if (matvar_t* binResVar = Mat_VarRead(dataMatFile, "bin_resolution"))
	{
		double* binResData = static_cast<double*>(binResVar->data);
		nlosData._deltaT = static_cast<float>(binResData[0]) * LIGHT_SPEED;
		Mat_VarFree(binResVar);
	}
	else
	{
		nlosData._deltaT = 32e-12f * LIGHT_SPEED; 
	}

	// Find the tof file
	std::vector<float> tof;

	std::string folder = filename.substr(0, filename.find_last_of('/'));
	std::string tofFile = folder + "/tof.mat";
	if (!std::filesystem::exists(tofFile))
	{
		// Search in parent folder
		folder = folder.substr(0, folder.find_last_of('/'));
		tofFile = folder + "/tof.mat";
	}

	if (std::filesystem::exists(tofFile))
	{
		if (mat_t* tofMatFile = Mat_Open(tofFile.c_str(), MAT_ACC_RDONLY))
		{
			if (matvar_t* tofVar = Mat_VarRead(tofMatFile, "tofgrid"))
			{
				double* tofData = static_cast<double*>(tofVar->data);

				for (size_t x = 0; x < nlosData._dims[0]; ++x)
				{
					for (size_t y = 0; y < nlosData._dims[1]; ++y)
					{
						size_t idx = x * nlosData._dims[1] + y;
						tof.push_back(static_cast<float>(tofData[idx]));
					}
				}
			}
		}
	}

	// Convolve data with tof
	if (!tof.empty())
	{
		const size_t width = nlosData._dims[0];
		const size_t height = nlosData._dims[1];
		const size_t depth = nlosData._dims[2];

		for (size_t ii = 0; ii < width; ++ii) 
		{
			for (size_t jj = 0; jj < height; ++jj) 
			{
				// Compute linear index (row-major: [x][y][t])
				size_t index3D = (ii * height + jj) * depth;

				// Calculate and normalize shift
				int shift = static_cast<int>(std::floor(tof[ii * height + jj] / (nlosData._deltaT / LIGHT_SPEED * 1e12f)));

				// Circular shift in-place
				std::vector<float> temp(depth);
				for (size_t k = 0; k < depth; ++k) 
				{
					int shiftK = (shift + static_cast<int>(depth) + static_cast<int>(k)) % static_cast<int>(depth);
					temp[k] = nlosData._data[index3D + shiftK];
				}

				for (size_t k = 0; k < depth; ++k) 
					nlosData._data[index3D + k] = temp[k];
			}
		}
	}

	// Rectify dims
	std::vector dataCopy(nlosData._dims[0] * nlosData._dims[1] * crop, 0.0f);
	for (size_t x = 0; x < nlosData._dims[0]; ++x)
	{
		for (size_t y = 0; y < nlosData._dims[1]; ++y)
		{
			for (size_t t = 0; t < crop; ++t)
			{
				size_t sourceIdx = x * nlosData._dims[1] * nlosData._dims[2] + y * nlosData._dims[2] + t;
				size_t destIdx = x * nlosData._dims[1] * crop + y * crop + t;
				dataCopy[destIdx] = nlosData._data[sourceIdx];
			}
		}
	}
	nlosData._data.swap(dataCopy);
	nlosData._dims[2] = crop; // Set z dimension to crop size

	// Fill with fake laser and sensor grid positions and normals
	const size_t numRelayWallTargets = nlosData._dims[0] * nlosData._dims[1];
	nlosData._cameraGridPositions.resize(numRelayWallTargets);
	nlosData._cameraGridNormals.resize(numRelayWallTargets);
	nlosData._laserGridPositions.resize(numRelayWallTargets);
	nlosData._laserGridNormals.resize(numRelayWallTargets);
	nlosData._cameraPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	nlosData._laserPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	nlosData._wallWidth = 1.0f;
	nlosData._cameraGridSize = glm::vec2(-nlosData._wallWidth, -nlosData._wallWidth);
	nlosData._laserGridSize = glm::vec2(nlosData._wallWidth, nlosData._wallWidth);
	nlosData._discardFirstLastBounces = true;
	nlosData._isConfocal = true;
	nlosData._t0 = .0f;
	nlosData._zOffset = 0;

	#pragma omp parallel for
	for (size_t x = 0; x < nlosData._dims[0]; ++x)
	{
		for (size_t y = 0; y < nlosData._dims[1]; ++y)
		{
			size_t idx = x * nlosData._dims[1] + y;

			nlosData._cameraGridPositions[idx] = glm::vec3(
				static_cast<float>(x) / (nlosData._wallWidth * 2.0f) - nlosData._wallWidth,
				0.0f,
				static_cast<float>(y) / (nlosData._wallWidth * 2.0f) - nlosData._wallWidth);
			nlosData._cameraGridNormals[idx] = glm::vec3(0.0f, -1.0f, 0.0f);

			nlosData._laserGridPositions[idx] = glm::vec3(
				static_cast<float>(x) / (nlosData._wallWidth * 2.0f) - nlosData._wallWidth,
				0.0f,
				static_cast<float>(y) / (nlosData._wallWidth * 2.0f) - nlosData._wallWidth);
			nlosData._laserGridNormals[idx] = glm::vec3(0.0f, -1.0f, 0.0f);
		}
	}

	return true;
}
