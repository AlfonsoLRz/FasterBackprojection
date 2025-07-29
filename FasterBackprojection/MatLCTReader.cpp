#include "stdafx.h"
#include "MatLCTReader.h"

#include "math.cuh"

//

bool MatLCTReader::read(const std::string& filename, NLosData& nlosData)
{
	// Open the .mat file
	mat_t* matFile = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
	if (!matFile)
		throw std::runtime_error("NLosData: Failed to load LCT data from .mat file.");

	constexpr glm::uint zTrim = 600; // Ignore first 600 samples in the z dimension
	const glm::uint zOffset = getZOffset(filename);

	//matvar_t* matvar;
	//while ((matvar = Mat_VarReadNextInfo(matFile)) != nullptr) {
	//	std::cout << "Found variable: " << matvar->name << std::endl;
	//	Mat_VarFree(matvar);
	//}

	matvar_t* rectDataVar = Mat_VarRead(matFile, "rect_data");
	if (!rectDataVar)
	{
		Mat_Close(matFile);
		return false;
	}

	glm::uint16_t* rectData = static_cast<glm::uint16_t*>(rectDataVar->data);

	size_t globalSize = 1;
	nlosData._dims.resize(rectDataVar->rank);
	for (int i = 0; i < rectDataVar->rank; ++i)
	{
		nlosData._dims[i] = rectDataVar->dims[i];
		globalSize *= rectDataVar->dims[i];
	}

	nlosData._data.resize(globalSize);
	#pragma omp parallel for
	for (size_t x = 0; x < nlosData._dims[0]; ++x)
	{
		for (size_t y = 0; y < nlosData._dims[1]; ++y)
		{
			for (size_t t = 0; t < nlosData._dims[2]; ++t)
			{
				size_t idx = y * nlosData._dims[1] * nlosData._dims[2] + x * nlosData._dims[2] + t;
				nlosData._data[idx] = 
					static_cast<float>(rectData[t * nlosData._dims[0] * nlosData._dims[1] + x * nlosData._dims[0] + y]) * 
					static_cast<float>(t >= zTrim);
			}
		}
	}

	matvar_t* widthVar = Mat_VarRead(matFile, "width");
	if (!widthVar)
	{
		Mat_VarFree(rectDataVar);
		Mat_Close(matFile);
		return false;
	}

	double* temporalWidth = static_cast<double*>(widthVar->data);
	nlosData._wallWidth = static_cast<float>(temporalWidth[0]);

	// Fill with fake laser and sensor grid positions and normals
	const size_t numRelayWallTargets = nlosData._dims[0] * nlosData._dims[1];
	nlosData._cameraGridPositions.resize(numRelayWallTargets);
	nlosData._cameraGridNormals.resize(numRelayWallTargets);
	nlosData._laserGridPositions.resize(numRelayWallTargets);
	nlosData._laserGridNormals.resize(numRelayWallTargets);
	nlosData._cameraPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	nlosData._laserPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	nlosData._cameraGridSize = glm::vec2(nlosData._dims[1], nlosData._dims[0]);
	nlosData._laserGridSize = glm::vec2(nlosData._dims[1], nlosData._dims[0]);
	nlosData._discardFirstLastBounces = true;
	nlosData._isConfocal = true;
	nlosData._deltaT = static_cast<float>(4e-12) * LIGHT_SPEED;
	nlosData._t0 = .0f;
	nlosData._zOffset = zOffset;

	#pragma omp parallel for
	for (size_t i = 0; i < nlosData._cameraGridPositions.size(); ++i)
	{
		nlosData._cameraGridPositions[i] = glm::vec3(
			static_cast<float>(i) / static_cast<float>(nlosData._dims[0]), 
			0.0f, 
			static_cast<float>(i % nlosData._dims[0]));
		nlosData._cameraGridNormals[i] = glm::vec3(0.0f, -1.0f, 0.0f);

		nlosData._laserGridPositions[i] = glm::vec3(
			static_cast<float>(i) / static_cast<float>(nlosData._dims[0]), 
			0.0f, 
			static_cast<float>(i % nlosData._dims[0]));
		nlosData._laserGridNormals[i] = glm::vec3(0.0f, -1.0f, 0.0f);
	}

	Mat_VarFree(rectDataVar);
	Mat_VarFree(widthVar);
	Mat_Close(matFile);

	return true;
}

glm::uint MatLCTReader::getZOffset(const std::string& filename)
{
	if (filename.find("data_resolution_chart_40cm") != std::string::npos)
		return 350;
	if (filename.find("data_resolution_chart_65cm") != std::string::npos)
		return 700;
	if (filename.find("data_dot_chart_40cm") != std::string::npos)
		return 350;
	if (filename.find("data_dot_chart_65cm") != std::string::npos)
		return 700;
	if (filename.find("data_mannequin") != std::string::npos)
		return 300;
	if (filename.find("data_exit_sign") != std::string::npos)
		return 600;
	if (filename.find("data_s_u") != std::string::npos)
		return 800;
	if (filename.find("data_outdoor_s") != std::string::npos)
		return 700;
	if (filename.find("data_diffuse_s") != std::string::npos)
		return 100;

	return 0;
}
