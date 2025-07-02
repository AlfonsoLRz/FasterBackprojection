#include "stdafx.h"
#include "TriangleMesh.h"

// Public methods

TriangleMesh::TriangleMesh()
= default;

TriangleMesh::~TriangleMesh()
{
}

bool TriangleMesh::load(const std::string& filename)
{
    std::string binaryFile = filename.substr(0, filename.find_last_of('.')) + BINARY_EXTENSION;

    if (std::filesystem::exists(binaryFile))
    {
        this->loadModelBinaryFile(binaryFile);
    }
    else
    {
        const aiScene* scene = _assimpImporter.ReadFile(filename, aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            std::cout << "ERROR::ASSIMP::" << _assimpImporter.GetErrorString() << std::endl;
            return false;
        }

        std::string shortName = aiScene::GetShortFilename(filename.c_str());
        std::string folder = filename.substr(0, filename.length() - shortName.length());
        glm::uint meshIndex = 0;

        this->processNode(scene->mRootNode, meshIndex, scene, folder);

		std::ranges::sort(_components, [](const Component& a, const Component& b) { return a._name < b._name; });

        this->writeBinaryFile(binaryFile);
    }

    this->calculateAABB();

    return true;
}

// Protected methods

glm::mat4 TriangleMesh::matrixToGLM(const aiMatrix4x4& matrix)
{
    glm::mat4 to;

    to[0][0] = static_cast<GLfloat>(matrix.a1); to[0][1] = static_cast<GLfloat>(matrix.b1);  to[0][2] = static_cast<GLfloat>(matrix.c1); to[0][3] = static_cast<GLfloat>(matrix.d1);
    to[1][0] = static_cast<GLfloat>(matrix.a2); to[1][1] = static_cast<GLfloat>(matrix.b2);  to[1][2] = static_cast<GLfloat>(matrix.c2); to[1][3] = static_cast<GLfloat>(matrix.d2);
    to[2][0] = static_cast<GLfloat>(matrix.a3); to[2][1] = static_cast<GLfloat>(matrix.b3);  to[2][2] = static_cast<GLfloat>(matrix.c3); to[2][3] = static_cast<GLfloat>(matrix.d3);
    to[3][0] = static_cast<GLfloat>(matrix.a4); to[3][1] = static_cast<GLfloat>(matrix.b4);  to[3][2] = static_cast<GLfloat>(matrix.c4); to[3][3] = static_cast<GLfloat>(matrix.d4);

    return to;
}

Model3D::Component TriangleMesh::processMesh(aiMesh* mesh, glm::uint& meshIndex, const aiScene* scene, const std::string& folder, const glm::mat4& transform)
{
    AABB aabb;
    std::vector<Vao::Vertex> vertices(mesh->mNumVertices);
    std::vector<GLuint> indices(mesh->mNumFaces * 3);
    int numVertices = static_cast<int>(mesh->mNumVertices);

    for (int i = 0; i < numVertices; i++)
    {
        Vao::Vertex vertex;
        vertex._position = transform * glm::vec4(glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z), 1.0f);
        vertex._normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        if (mesh->mTextureCoords[0]) vertex._texCoords = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);

        vertices[i] = vertex;
        aabb.update(vertex._position);
    }

    // Indices
    for (unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++)
            indices[i * 3 + j] = face.mIndices[j];
    }

    aiMaterial* mat = scene->mMaterials[mesh->mMaterialIndex];
    aiString matName;
    mat->Get(AI_MATKEY_NAME, matName);
    std::string meshName = matName.C_Str();
    meshName += " (" + std::to_string(meshIndex++) + ")";

    Component component;
	component._name = meshName;
    component._vertices = std::move(vertices);
    component._indices[Vao::TRIANGLE] = std::move(indices);
    component._aabb = aabb;

    return component;
}

void TriangleMesh::processNode(const aiNode* node, glm::uint& meshIndex, const aiScene* scene, const std::string& folder)
{
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        aiMatrix4x4 transformation = node->mTransformation;

        _components.push_back(processMesh(mesh, meshIndex, scene, folder, matrixToGLM(transformation)));
        _aabb.update(_components.back()._aabb);
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        this->processNode(node->mChildren[i], meshIndex, scene, folder);
    }
}