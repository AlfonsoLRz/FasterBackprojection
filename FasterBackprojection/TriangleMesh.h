#pragma once

#include <assimp/Importer.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Model3D.h"

class TriangleMesh : public Model3D
{
protected:
	Assimp::Importer		_assimpImporter;

protected:
	static glm::mat4 matrixToGLM(const aiMatrix4x4& matrix);
	static Component processMesh(aiMesh* mesh, glm::uint& meshIndex, const aiScene* scene, const std::string& folder, const glm::mat4& transform);
	void processNode(const aiNode* node, glm::uint& meshIndex, const aiScene* scene, const std::string& folder);

public:
	TriangleMesh();
	~TriangleMesh() override;

	bool load(const std::string& filename);
};

