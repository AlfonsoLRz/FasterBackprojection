// ReSharper disable All
#include "stdafx.h"
#include "DrawEllipsoids.h"

#include "ChronoUtilities.h"
#include "FileUtilities.h"
#include "progressbar.hpp"
#include "ShaderProgramDB.h"

#include "libmorton/morton.h"

DrawEllipsoids::DrawEllipsoids(NLosData* nlosData): _numInstances(0), _nlosData(nlosData)
{
	_components.resize(2);

	// Multi-instance rendering
	std::vector<glm::vec3> ellipsoidPositions, ellipsoidScale;
	std::vector<float> ellipsoidIntensities;

	size_t maxNumEllipsoids = _nlosData->_data.size();
	ellipsoidPositions.reserve(maxNumEllipsoids / 4);
	ellipsoidScale.reserve(maxNumEllipsoids / 4);
	ellipsoidIntensities.reserve(maxNumEllipsoids / 4);

	AABB relayWall;

	ChronoUtilities::startTimer();

	//for (int t = 0; t < _nlosData->_temporalResolution; ++t)
	//{
	//	float* timeSlice = _nlosData->getTimeSlice(t);

	//	if (_nlosData->_isConfocal)
	//	{
	//		glm::uint pixelIdx = 0;

	//		for (const glm::vec3& pos : _nlosData->_laserGridPositions)
	//		{
	//			if (timeSlice[pixelIdx] > glm::epsilon<float>())
	//			{
	//				const glm::vec3 lPos = _nlosData->_laserGridPositions[pixelIdx];
	//				float traversalDistance = static_cast<float>(t) * _nlosData->_deltaT + _nlosData->_t0;
	//				//if (_nlosData-> == 0)		// Measurements did not discard first and last bounces
	//				traversalDistance -= (glm::distance(lPos, _nlosData->_laserPosition) + glm::distance(lPos, _nlosData->_cameraPosition));
	//				//traversalDistance /= 2.0f;		// Disabled because radius is 0.5f and not 1.0f

	//				ellipsoidPositions.push_back(pos);
	//				ellipsoidScale.emplace_back(traversalDistance); // Uniform cale for confocal ellipsoids
	//				ellipsoidIntensities.push_back(timeSlice[pixelIdx]);
	//				relayWall.update(pos);
	//			}

	//			++pixelIdx;
	//		}
	//	}
	//}

	this->reorder(ellipsoidPositions, ellipsoidScale, ellipsoidIntensities, relayWall);

	std::cout << "Ellipsoids created in " << ChronoUtilities::getElapsedTime() << " ms.\n";

	// Half ellipsoid
	Component* ellipsoid = &_components.front();

	createHalfEllipsoid(ellipsoid, 256, 256);
	ellipsoid->buildVao();

	ellipsoid->_vao->createMultiInstanceVBO(Vao::TRANSLATION, glm::vec3(.0f), .0f, GL_FLOAT);
	ellipsoid->_vao->setVBOData(Vao::TRANSLATION, ellipsoidPositions.data(), static_cast<GLsizei>(ellipsoidPositions.size()));

	ellipsoid->_vao->createMultiInstanceVBO(Vao::SCALE, glm::vec3(.0f), .0f, GL_FLOAT);
	ellipsoid->_vao->setVBOData(Vao::SCALE, ellipsoidScale.data(), static_cast<GLsizei>(ellipsoidScale.size()));

	ellipsoid->_vao->createMultiInstanceVBO(Vao::INTENSITY, .0f, .0f, GL_FLOAT);
	ellipsoid->_vao->setVBOData(Vao::INTENSITY, ellipsoidIntensities.data(), static_cast<GLsizei>(ellipsoidIntensities.size()));

	// Mesh in the bounding box of the hidden geometry
	Component* hiddenBox = &_components.back();

	glm::vec3 hiddenBoxMin = _nlosData->_hiddenGeometry.minPoint(), hiddenBoxSize = _nlosData->_hiddenGeometry.size();
	glm::vec3 stepSize = hiddenBoxSize / glm::vec3(256);
	glm::vec3 initialPoint = hiddenBoxMin;

	for (int x = 0; x < 256; ++x)
		for (int z = 0; z < 256; ++z)
			hiddenBox->_vertices.push_back({ ._position = initialPoint + glm::vec3(x, 0.0f, z) * stepSize });

	hiddenBox->generatePointCloud();
	hiddenBox->buildVao();

	hiddenBox->_vao->createMultiInstanceVBO(Vao::TRANSLATION, glm::vec3(.0f), .0f, GL_FLOAT);
	hiddenBox->_vao->setVBOData(Vao::TRANSLATION, ellipsoidPositions.data(), static_cast<GLsizei>(ellipsoidPositions.size()));

	hiddenBox->_vao->createMultiInstanceVBO(Vao::SCALE, glm::vec3(.0f), .0f, GL_FLOAT);
	hiddenBox->_vao->setVBOData(Vao::SCALE, ellipsoidScale.data(), static_cast<GLsizei>(ellipsoidScale.size()));

	hiddenBox->_vao->createMultiInstanceVBO(Vao::INTENSITY, .0f, .0f, GL_FLOAT);
	hiddenBox->_vao->setVBOData(Vao::INTENSITY, ellipsoidIntensities.data(), static_cast<GLsizei>(ellipsoidIntensities.size()));

	// Set up component properties
	_numInstances = static_cast<GLuint>(ellipsoidPositions.size());

	// Change model matrix according to difference between half ellipsoid and wall normal
	const glm::vec3 halfEllipsoid = glm::vec3(.0f, .0f, 1.0f);
	const glm::vec3 wallNormal = glm::vec3(.0f, .0f, -1.0f);
	const float angle = glm::acos(glm::dot(halfEllipsoid, wallNormal));
	_modelMatrix = glm::rotate(glm::mat4(1.0f), glm::pi<float>() / 256 / 2.0f, wallNormal) * glm::rotate(glm::mat4(1.0f), angle, glm::vec3(1.0f, 0.0f, 0.0f));

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

	//if (appState->_primitiveEnabled[Vao::TRIANGLE] && component->_enabled && component->_vao)
	//{
	//	_multiInstanceTriangleShader->use();
	//	_multiInstanceTriangleShader->setUniform("mModelViewProj", viewProjectionMatrix);
	//	_multiInstanceTriangleShader->applyActiveSubroutines();
	//	component->_vao->drawObject(
	//		Vao::TRIANGLE, GL_TRIANGLES,
	//		static_cast<GLsizei>(component->_indices[Vao::TRIANGLE].size()), static_cast<GLsizei>(glm::min(1u, _numInstances)));
	//}

	if (appState->_primitiveEnabled[Vao::LINE] && component->_enabled && component->_vao)
	{
		_lineShader->use();
		_lineShader->setUniform("mModelViewProj", viewProjectionMatrix);
		_lineShader->setUniform("lineColor", component->_material._lineColor);
		_lineShader->applyActiveSubroutines();
		component->_vao->drawObject(
			Vao::LINE, GL_LINES,
			static_cast<GLsizei>(component->_indices[Vao::LINE].size()), static_cast<GLsizei>(glm::min(1u, _numInstances)));
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
	component->generateWireframe();
}

static unsigned int expandBits(unsigned int v)
{
	v = (v | (v << 16)) & 0x030000FF;
	v = (v | (v << 8)) & 0x0300F00F;
	v = (v | (v << 4)) & 0x030C30C3;
	v = (v | (v << 2)) & 0x09249249;
	return v;
}

static unsigned int morton3D(float x, float y, float z)
{
	unsigned int xx = std::min(1023u, static_cast<unsigned int>(x * 1024.0f));
	unsigned int yy = std::min(1023u, static_cast<unsigned int>(y * 1024.0f));
	unsigned int zz = std::min(1023u, static_cast<unsigned int>(z * 1024.0f));
	return (expandBits(xx) << 0) | (expandBits(yy) << 1) | (expandBits(zz) << 2);
}

void DrawEllipsoids::reorder(std::vector<glm::vec3>& ellipsoidPositions, std::vector<glm::vec3>& ellipsoidScale, std::vector<float>& ellipsoidIntensities, const AABB& relayWall)
{
	const glm::vec3 size = relayWall.size(), aabbMin = relayWall.minPoint();
	std::vector<std::pair<unsigned int, unsigned int>> indexMorton(ellipsoidPositions.size());
	float extent = std::max({ size.x, size.y, size.z });
	glm::uint numEllipsoids = static_cast<glm::uint>(ellipsoidPositions.size());

	for (unsigned int i = 0; i < numEllipsoids; i++)
	{
		const auto pos = (ellipsoidPositions[i] - aabbMin) / extent;
		indexMorton[i] = std::make_pair(i, morton3D(pos.x, pos.y, pos.z));
	}

	std::sort(std::execution::par_unseq, indexMorton.begin(), indexMorton.end(), [](const auto& a, const auto& b) {
		return a.second < b.second;
	});

	std::vector<glm::vec3> tempPositions(numEllipsoids);
	std::vector<float> tempIntensities(numEllipsoids);
	std::vector<glm::vec3> tempScales(numEllipsoids);

	#pragma omp parallel for
	for (int i = 0; i < numEllipsoids; ++i)
	{
		glm::uint index = indexMorton[i].first;
		tempPositions[i] = ellipsoidPositions[index];
		tempIntensities[i] = ellipsoidIntensities[index];
		tempScales[i] = ellipsoidScale[index];
	}

	ellipsoidPositions = std::move(tempPositions);
	ellipsoidIntensities = std::move(tempIntensities);
	ellipsoidScale = std::move(tempScales);
}

void DrawEllipsoids::solveBackprojection()
{
	Component* component = &_components.front();

	// 
	const GLsizei BATCH_SIZE = 65536; 
	const glm::ivec3 voxelRes(256);
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
	float clearValue = 0.0f; 
	glClearTexImage(voxelTex, 0, GL_RED, GL_FLOAT, &clearValue);

	// Set up rendering state
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_MULTISAMPLE);
	glDisable(GL_POLYGON_OFFSET_FILL);

	// Shader Uniforms - (All correct uniforms now passed to the shader)
	_backprojectionShader->use();
	_backprojectionShader->setUniform("voxelMin", minBounds);
	_backprojectionShader->setUniform("voxelSize", maxBounds - minBounds);
	_backprojectionShader->setUniform("voxelRes", voxelRes);
	_backprojectionShader->setUniform("voxelModelMatrix", this->_modelMatrix);
	_backprojectionShader->setUniform("voxelProjectionMatrix", voxelProjectionMatrix); 

	ChronoUtilities::startTimer();

	// Draw Call - (Standard instanced draw)
	glBindVertexArray(component->_vao->_vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, component->_vao->_ibo[Vao::TRIANGLE]);

	progressbar bar(numEllipsoids / BATCH_SIZE + 1, true);

	for (GLsizei i = 0; i < numEllipsoids; i += BATCH_SIZE) {
		GLsizei currentBatchSize = glm::min(BATCH_SIZE, numEllipsoids - i);
		GLuint offset = i; 

		glDrawElementsInstancedBaseInstance(
			GL_TRIANGLES,
			static_cast<GLsizei>(component->_indices[Vao::TRIANGLE].size()),
			GL_UNSIGNED_INT,
			nullptr, 
			currentBatchSize,
			offset);

		glFinish();

		//bar.update();
		std::cout << "DrawEllipsoids::solveBackprojection: Drawn batch " << (i / BATCH_SIZE) + 1 << " of " << (numEllipsoids + BATCH_SIZE - 1) / BATCH_SIZE << "\n";
	}

	// Synchronize
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

void DrawEllipsoids::solveBackprojection2()
{
	Component* component = &_components.back();

	// 
	const GLsizei BATCH_SIZE = 1;
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
	float clearValue = 0.0f;
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

	progressbar bar(numEllipsoids / BATCH_SIZE + 1, true);

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
		break;

		//bar.update();
		std::cout << "DrawEllipsoids::solveBackprojection: Drawn batch " << (i / BATCH_SIZE) + 1 << " of " << (numEllipsoids + BATCH_SIZE - 1) / BATCH_SIZE << "\n";
	}

	// Synchronize
	glBindVertexArray(0);
	glUseProgram(0);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glFinish();

	std::cout << "DrawEllipsoids::solveBackprojection: Backprojection completed in " << ChronoUtilities::getElapsedTime() << " milliseconds.\n";

	// Read back 
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
