// ReSharper disable All
#include "stdafx.h"
#include "DrawEllipsoids.h"

#include "ChronoUtilities.h"
#include "FileUtilities.h"
#include "ShaderProgramDB.h"

DrawEllipsoids::DrawEllipsoids(NLosData* nlosData): _numInstances(0), _nlosData(nlosData)
{
	_components.resize(1);
	Component* component = &_components.front();

	// Multi-instance rendering
	glm::uint timeIdx = 544;
	float* timeSlice = nlosData->getTimeSlice(timeIdx);

	std::vector<glm::vec3> ellipsoidPositions, ellipsoidScale;
	std::vector<float> ellipsoidIntensities;

	for (int t = 543; t < nlosData->_temporalResolution; ++t)
	{
		float* timeSlice = _nlosData->getTimeSlice(t);

		if (_nlosData->_isConfocal)
		{
			glm::uint pixelIdx = 0;

			for (const glm::vec3& pos : _nlosData->_laserGridPositions)
			{
				if (timeSlice[pixelIdx] > glm::epsilon<float>())
				{
					const glm::vec3 lPos = _nlosData->_laserGridPositions[pixelIdx];
					float traversalDistance = static_cast<float>(t) * _nlosData->_deltaT + _nlosData->_t0;
					//if (_nlosData-> == 0)		// Measurements did not discard first and last bounces
					traversalDistance -= (glm::distance(lPos, _nlosData->_laserPosition) + glm::distance(lPos, _nlosData->_cameraPosition));
					//traversalDistance /= 2.0f;		// Disabled because radius is 0.5f and not 1.0f

					ellipsoidPositions.push_back(pos);
					ellipsoidScale.emplace_back(traversalDistance); // Scale for confocal ellipsoids
					ellipsoidIntensities.push_back(timeSlice[pixelIdx]);
				}

				++pixelIdx;
			}
		}
	}

	// Normalize intensity - (Same as before, check for maxIntensity == minIntensity for division by zero)
	//float maxIntensity = *std::max_element(ellipsoidIntensities.begin(), ellipsoidIntensities.end());
	//float minIntensity = *std::min_element(ellipsoidIntensities.begin(), ellipsoidIntensities.end());
	//std::vector<uint32_t> ellipsoidUnsignedIntensity(ellipsoidIntensities.size());
	//for (size_t i = 0; i < ellipsoidIntensities.size(); ++i)
	//{
	//	float normalizedIntensity = (ellipsoidIntensities[i] - minIntensity) / (maxIntensity - minIntensity);
	//	ellipsoidUnsignedIntensity[i] = static_cast<uint32_t>(normalizedIntensity * 255.0f);
	//}

	// 
	createHalfEllipsoid(component, 180, 180);
	component->buildVao();

	component->_vao->createMultiInstanceVBO(Vao::TRANSLATION, glm::vec3(.0f), .0f, GL_FLOAT);
	component->_vao->setVBOData(Vao::TRANSLATION, ellipsoidPositions.data(), static_cast<GLsizei>(ellipsoidPositions.size()));

	component->_vao->createMultiInstanceVBO(Vao::SCALE, glm::vec3(.0f), .0f, GL_FLOAT);
	component->_vao->setVBOData(Vao::SCALE, ellipsoidScale.data(), static_cast<GLsizei>(ellipsoidScale.size()));

	component->_vao->createMultiInstanceVBO(Vao::INTENSITY, .0f, .0f, GL_FLOAT);
	component->_vao->setVBOData(Vao::INTENSITY, ellipsoidIntensities.data(), static_cast<GLsizei>(ellipsoidIntensities.size()));

	// Set up component properties
	_numInstances = static_cast<GLuint>(ellipsoidPositions.size());

	// Change model matrix according to difference between half ellipsoid and wall normal
	const glm::vec3 halfEllipsoid = glm::vec3(.0f, .0f, 1.0f);
	const glm::vec3 wallNormal = glm::vec3(.0f, .0f, -1.0f);
	const float angle = glm::acos(glm::dot(halfEllipsoid, wallNormal));
	_modelMatrix = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(1.0f, 0.0f, 0.0f));

	// Retrieve shaders for rendering ellipsoids later
	_triangleShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::TRIANGLE_RENDERING);
	_lineShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::LINE_RENDERING);
	_multiInstanceTriangleShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::MULTI_INSTANCE_TRIANGLE_RENDERING);
	_backprojectionShader = ShaderProgramDB::getInstance()->getShader(ShaderProgramDB::BACKPROJECTION_RENDERING);
}

DrawEllipsoids::~DrawEllipsoids()
{
}

void DrawEllipsoids::draw(MatrixRenderInformation* matrixInformation, ApplicationState* appState)
{
	const glm::mat4 viewProjectionMatrix =
		matrixInformation->_matrix[MatrixRenderInformation::VIEW_PROJECTION] *
		matrixInformation->_matrix[MatrixRenderInformation::MODEL] *
		this->_modelMatrix;

	Component* component = &_components.front();

	const AABB hiddenGeometry = _nlosData->_hiddenGeometry;
	glm::vec3 minBounds = hiddenGeometry.minPoint(), maxBounds = hiddenGeometry.maxPoint();

	if (appState->_primitiveEnabled[Vao::TRIANGLE] && component->_enabled && component->_vao)
	{
		_multiInstanceTriangleShader->use();
		_multiInstanceTriangleShader->setUniform("mModelViewProj", viewProjectionMatrix);
		_multiInstanceTriangleShader->applyActiveSubroutines();
		component->_vao->drawObject(
			Vao::POINT, GL_POINTS, 
			static_cast<GLsizei>(component->_indices[Vao::POINT].size()), static_cast<GLsizei>(glm::min(1u, _numInstances)));
	}
}

void DrawEllipsoids::createEllipsoid(Component* component, int stacks, int sectors)
{
	std::vector<Vao::Vertex> geometry;
	float sectorStep = 2.0f * glm::pi<float>() / static_cast<float>(sectors);
	float stackStep = glm::pi<float>() / static_cast<float>(stacks);

	for (int i = 0; i <= stacks; ++i)
	{
		float stackAngle = glm::pi<float>() / 2.0f - static_cast<float>(i) * stackStep;		// Starting from pi/2 to -pi/2
		float xy = 0.5f * cosf(stackAngle);						
		float z = 0.5f * sinf(stackAngle);		

		// Add (sectorCount + 1) vertices per stack the first and last vertices have same position and normal, but different texture coordinates
		for (int j = 0; j <= sectors; ++j)
		{
			float sectorAngle = static_cast<float>(j) * sectorStep;					// Starting from 0 to 2pi

			Vao::Vertex vertexData;
			vertexData._position = glm::vec3(xy * cosf(sectorAngle), xy * sinf(sectorAngle), z);	// r * cos(u) * cos(v), r * cos(u) * sin(v), r * sin(u)
			vertexData._normal = glm::vec3(vertexData._position.x, vertexData._position.y, vertexData._position.z);
			vertexData._texCoords = glm::vec2(static_cast<float>(j) / static_cast<float>(sectors), static_cast<float>(i) / static_cast<float>(stacks));

			geometry.push_back(vertexData);
		}
	}

	for (int i = 0; i < stacks; ++i)
	{
		int k1 = i * (sectors + 1);     // Beginning of current stack
		int k2 = k1 + sectors + 1;      // Beginning of next stack

		for (int j = 0; j < sectors; ++j, ++k1, ++k2)
		{
			// 2 triangles per sector excluding first and last stacks
			// k1 => k2 => k1+1
			if (i != 0)
			{
				component->_indices[Vao::TRIANGLE].push_back(k1);
				component->_indices[Vao::TRIANGLE].push_back(k2);
				component->_indices[Vao::TRIANGLE].push_back(k1 + 1);
			}

			// k1 + 1 => k2 => k2 + 1
			if (i != stacks - 1)
			{
				component->_indices[Vao::TRIANGLE].push_back(k1 + 1);
				component->_indices[Vao::TRIANGLE].push_back(k2);
				component->_indices[Vao::TRIANGLE].push_back(k2 + 1);
			}
		}
	}

	component->_vertices = std::move(geometry);
}

void DrawEllipsoids::createHalfEllipsoid(Component* component, int stacks, int sectors)
{
	std::vector<Vao::Vertex> geometry;
	float sectorStep = glm::pi<float>() / static_cast<float>(sectors);
	float stackStep = glm::pi<float>() / static_cast<float>(stacks);

	for (int i = 0; i <= stacks; ++i)
	{
		float stackAngle = glm::pi<float>() / 2.0f - static_cast<float>(i) * stackStep;		// Starting from pi/2 to -pi/2
		float xy = 0.5f * cosf(stackAngle);
		float z = 0.5f * sinf(stackAngle);

		// Add (sectorCount + 1) vertices per stack the first and last vertices have same position and normal, but different texture coordinates
		for (int j = 0; j <= sectors; ++j)
		{
			float sectorAngle = static_cast<float>(j) * sectorStep;					// Starting from 0 to 2pi

			Vao::Vertex vertexData;
			vertexData._position = glm::vec3(xy * cosf(sectorAngle), xy * sinf(sectorAngle), z);	// r * cos(u) * cos(v), r * cos(u) * sin(v), r * sin(u)
			vertexData._normal = glm::vec3(vertexData._position.x, vertexData._position.y, vertexData._position.z);
			vertexData._texCoords = glm::vec2(static_cast<float>(j) / static_cast<float>(sectors), static_cast<float>(i) / static_cast<float>(stacks));

			geometry.push_back(vertexData);
		}
	}

	for (int i = 0; i < stacks; ++i)
	{
		int k1 = i * (sectors + 1);     // Beginning of current stack
		int k2 = k1 + sectors + 1;      // Beginning of next stack

		for (int j = 0; j < sectors; ++j, ++k1, ++k2)
		{
			// 2 triangles per sector excluding first and last stacks
			// k1 => k2 => k1+1
			if (i != 0)
			{
				component->_indices[Vao::TRIANGLE].push_back(k1);
				component->_indices[Vao::TRIANGLE].push_back(k2);
				component->_indices[Vao::TRIANGLE].push_back(k1 + 1);
			}

			// k1 + 1 => k2 => k2 + 1
			if (i != stacks - 1)
			{
				component->_indices[Vao::TRIANGLE].push_back(k1 + 1);
				component->_indices[Vao::TRIANGLE].push_back(k2);
				component->_indices[Vao::TRIANGLE].push_back(k2 + 1);
			}
		}
	}

	component->_vertices = std::move(geometry);
	component->generatePointCloud();
}

void DrawEllipsoids::solveBackprojection()
{
	Component* component = &_components.front();

	// 
	const GLsizei BATCH_SIZE = 65536; // Example batch size
	const glm::ivec3 voxelRes(256, 256, 256);
	GLsizei numEllipsoids = static_cast<GLsizei>(_numInstances);
	std::printf("DrawEllipsoids::solveBackprojection: Number of instances: %i\n", numEllipsoids);

	//
	const AABB hiddenGeometry = _nlosData->_hiddenGeometry;
	glm::vec3 minBounds = hiddenGeometry.minPoint(), maxBounds = hiddenGeometry.maxPoint();
	glm::mat4 voxelProjectionMatrix = glm::ortho(
		minBounds.x, maxBounds.x, // left, right
		minBounds.y, maxBounds.y, // bottom, top
		minBounds.z, maxBounds.z  // near, far
	);

	// 3D texture
	GLuint voxelTex;
	glGenTextures(1, &voxelTex);
	glBindTexture(GL_TEXTURE_3D, voxelTex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32F, voxelRes.x, voxelRes.y, voxelRes.z);
	glBindImageTexture(0, voxelTex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

	// Clean texture
	float clearValue = 0.0f; // Clear with a float 0.0
	glClearTexImage(voxelTex, 0, GL_RED, GL_FLOAT, &clearValue);

	// Set up rendering state
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);

	// Shader Uniforms - (All correct uniforms now passed to the shader)
	_backprojectionShader->use();
	_backprojectionShader->setUniform("voxelMin", minBounds);
	_backprojectionShader->setUniform("voxelMax", maxBounds);
	_backprojectionShader->setUniform("voxelRes", voxelRes);
	_backprojectionShader->setUniform("voxelModelMatrix", this->_modelMatrix);
	_backprojectionShader->setUniform("voxelProjectionMatrix", voxelProjectionMatrix); 

	ChronoUtilities::startTimer();

	// Draw Call - (Standard instanced draw)
	glBindVertexArray(component->_vao->_vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, component->_vao->_ibo[Vao::POINT]);

	for (GLsizei i = 0; i < numEllipsoids; i += BATCH_SIZE) {
		GLsizei currentBatchSize = glm::min(BATCH_SIZE, numEllipsoids - i);
		GLuint offset = i; 

		glDrawElementsInstancedBaseInstance(
			GL_POINTS,
			static_cast<GLsizei>(component->_indices[Vao::POINT].size()),
			GL_UNSIGNED_INT,
			nullptr, 
			currentBatchSize,
			offset);
		glFinish();
		std::cout << "DrawEllipsoids::solveBackprojection: Drawn batch " << (i / BATCH_SIZE) + 1 << " of " << (numEllipsoids + BATCH_SIZE - 1) / BATCH_SIZE << "\n";
	}

	// Clean up and synchronize - (All good)
	glBindVertexArray(0);
	glUseProgram(0);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glFinish();

	std::cout << "DrawEllipsoids::solveBackprojection: Backprojection completed in " << ChronoUtilities::getElapsedTime() << " milliseconds.\n";

	// Read Back and Verify - (Correctly reads back the voxel texture)
	glBindTexture(GL_TEXTURE_3D, voxelTex);
	size_t voxelCount = voxelRes.x * voxelRes.y * voxelRes.z;
	std::vector<float> voxelData(voxelCount);

	glGetTexImage(GL_TEXTURE_3D, 0, GL_RED, GL_FLOAT, voxelData.data());

	bool anyNonZero = std::any_of(
		voxelData.begin(), voxelData.end(),
		[](uint32_t val) { return val > glm::epsilon<float>(); }
	);

	FileUtilities::write("output/aabb.cube", voxelData);

	if (anyNonZero)
		std::cout << "Voxel texture contains non-zero data.\n";
	else
		std::cout << "Voxel texture is all zeros.\n"; 

	// Restore OpenGL State - (Good practice)
	glColorMask(true, true, true, true);
	glDepthMask(true);
	glEnable(GL_DEPTH_TEST);
}