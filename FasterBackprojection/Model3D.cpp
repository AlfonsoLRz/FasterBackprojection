#include "stdafx.h"
#include "Model3D.h"

#include "RenderingShader.h"

// Static properties

const std::string Model3D::BINARY_EXTENSION = ".binp";
std::unordered_set<std::string> Model3D::USED_NAMES;

// Public methods

Model3D::Model3D() : _modelMatrix(1.0f)
{
    this->overrideModelName();
}

Model3D::~Model3D() = default;

void Model3D::draw(MatrixRenderInformation* matrixInformation, ApplicationState* appState)
{
}

Model3D* Model3D::moveGeometryToOrigin(const glm::mat4& origMatrix, float maxScale)
{
    AABB aabb = this->getAABB();

    glm::vec3 translate = -aabb.center();
    glm::vec3 extent = aabb.extent();
    float maxScaleAABB = std::max({extent.x, extent.y, extent.z});
    glm::vec3 scale = (maxScale < FLT_MAX) ? ((maxScale > maxScaleAABB) ? glm::vec3(1.0f) : glm::vec3(maxScale / maxScaleAABB)) : glm::vec3(1.0f);

    _modelMatrix = glm::scale(glm::mat4(1.0f), scale) * glm::translate(glm::mat4(1.0f), translate) * origMatrix;

    return this;
}

Model3D* Model3D::overrideModelName()
{
    std::string className = typeid(*this).name();
    std::string classTarget = "class ";
    size_t classIndex = className.find(classTarget);
    if (classIndex != std::string::npos)
    {
        className = className.substr(classIndex + classTarget.size(), className.size() - classIndex - classTarget.size());
    }

    unsigned modelIdx = 0;
	for (Component& component : _components)
	{
        bool nameValid = false;
        while (!nameValid)
        {
            component._name = className + " " + std::to_string(modelIdx);
            nameValid = !USED_NAMES.contains(component._name);
            ++modelIdx;
        }

        USED_NAMES.insert(component._name);
	}

    return this;
}

Model3D* Model3D::setLineColor(const glm::vec3& color)
{
    for (auto& component : _components)
        component._material._lineColor = color;

    return this;
}

Model3D* Model3D::setPointColor(const glm::vec3& color)
{
    for (auto& component : _components)
        component._material._pointColor = color;

    return this;
}

Model3D* Model3D::setTriangleColor(const glm::vec4& color)
{
    for (auto& component : _components)
        component._material._kdColor = color;

    return this;
}

Model3D* Model3D::setLineWidth(float width)
{
    for (auto& component : _components)
        component._lineWidth = width;

    return this;
}

Model3D* Model3D::setPointSize(float size)
{
    for (auto& component : _components)
        component._pointSize = size;

    return this;
}

Model3D* Model3D::setTopologyVisibility(Vao::IBO topology, bool visible)
{
    for (auto& component : _components)
        component._activeRendering[topology] = visible;

    return this;
}

// Private methods

void Model3D::calculateAABB()
{
    _aabb = AABB();

    for (auto& component : _components)
        for (Vao::Vertex& vertex : component._vertices)
            _aabb.update(vertex._position);
}

void Model3D::clearData()
{
    for (auto& component : _components)
    {
        component._vertices.clear();
        for (int i = 0; i < 3; ++i)
            component._indices[i].clear();
    }
}

void Model3D::loadModelBinaryFile(const std::string& path)
{
    std::ifstream fin(path, std::ios::in | std::ios::binary);
    if (!fin.is_open())
    {
        std::cout << "Failed to open the binary file " << path << "!" << std::endl;
        return;
    }

    size_t numComponents = _components.size();
    fin.read(reinterpret_cast<char*>(&numComponents), sizeof(size_t));

    for (size_t compIdx = 0; compIdx < numComponents; ++compIdx)
    {
        Component component;
        size_t numVertices, numIndices;

        fin.read(reinterpret_cast<char*>(&numVertices), sizeof(size_t));
        component._vertices.resize(numVertices);
        fin.read(reinterpret_cast<char*>(component._vertices.data()), sizeof(Vao::Vertex) * numVertices);

        for (auto& indices: component._indices)
        {
            fin.read(reinterpret_cast<char*>(&numIndices), sizeof(size_t));
            indices.resize(numIndices);

            if (numIndices)
            {
                indices.resize(numIndices);
                fin.read(reinterpret_cast<char*>(indices.data()), sizeof(GLuint) * numIndices);
            }
        }

        fin.read(reinterpret_cast<char*>(&component._aabb), sizeof(AABB));
        fin.read(reinterpret_cast<char*>(&component._material), sizeof(Material));

        size_t nameLength;
        fin.read(reinterpret_cast<char*>(&nameLength), sizeof(size_t));
        component._name.resize(nameLength);
        fin.read(component._name.data(), nameLength);

        _components.emplace_back(std::move(component));
        _aabb.update(_components[compIdx]._aabb);
    }
}

void Model3D::writeBinaryFile(const std::string& path)
{
    std::ofstream fout(path, std::ios::out | std::ios::binary);
    if (!fout.is_open())
    {
        std::cout << "Failed to write the binary file!" << std::endl;
    }

    size_t numComponents = _components.size();
    fout.write(reinterpret_cast<char*>(&numComponents), sizeof(size_t));

    for (auto& component : _components)
    {
        size_t numVertices = component._vertices.size();

        fout.write(reinterpret_cast<char*>(&numVertices), sizeof(size_t));
        fout.write(reinterpret_cast<char*>(component._vertices.data()), numVertices * sizeof(Vao::Vertex));

        for (auto& indices : component._indices)
        {
            size_t numIndices = indices.size();
            fout.write(reinterpret_cast<char*>(&numIndices), sizeof(size_t));
            if (numIndices)
                fout.write(reinterpret_cast<char*>(indices.data()), numIndices * sizeof(GLuint));
		}

        fout.write(reinterpret_cast<char*>(&component._aabb), sizeof(AABB));
        fout.write(reinterpret_cast<char*>(&component._material), sizeof(Material));

        size_t nameLength = component._name.size();
        fout.write(reinterpret_cast<char*>(&nameLength), sizeof(size_t));
        fout.write(component._name.c_str(), nameLength);
    }

    fout.close();
}

// MatrixRenderInformation methods

Model3D::MatrixRenderInformation::MatrixRenderInformation()
{
    for (glm::mat4& matrix : _matrix)
        matrix = glm::mat4(1.0f);
}

void Model3D::MatrixRenderInformation::undoMatrix(MatrixType type)
{
    if (_heapMatrices[type].empty())
        _matrix[type] = glm::mat4(1.0f);
    else
    {
        _matrix[type] = *(--_heapMatrices[type].end());
        _heapMatrices[type].erase(--_heapMatrices[type].end());
    }
}

// Component methods

void Model3D::Component::buildVao()
{
    Vao* vao = new Vao(true);
    vao->setVBOData(this->_vertices);
    vao->setIBOData(Vao::POINT, this->_indices[Vao::POINT]);
    vao->setIBOData(Vao::LINE, this->_indices[Vao::LINE]);
    vao->setIBOData(Vao::TRIANGLE, this->_indices[Vao::TRIANGLE]);
    this->_vao = vao;
}

void Model3D::Component::completeTopology()
{
    if (!this->_indices[Vao::TRIANGLE].empty() && this->_indices[Vao::LINE].empty())
    {
        this->generateWireframe();
    }

    if (!this->_indices[Vao::LINE].empty() && this->_indices[Vao::POINT].empty())
    {
        this->generatePointCloud();
    }
}

void Model3D::Component::generateWireframe()
{
    std::unordered_map<glm::uint, std::unordered_set<glm::uint>> segmentIncluded;
    auto isIncluded = [&](glm::uint index1, glm::uint index2) -> bool
        {
            std::unordered_map<glm::uint, std::unordered_set<glm::uint>>::iterator it;

            if ((it = segmentIncluded.find(index1)) != segmentIncluded.end())
            {
                if (it->second.contains(index2))
                {
                    return true;
                }
            }

            if ((it = segmentIncluded.find(index2)) != segmentIncluded.end())
            {
                if (it->second.contains(index1))
                {
                    return true;
                }
            }

            return false;
        };

    const size_t numIndices = this->_indices[Vao::TRIANGLE].size();

    for (size_t i = 0; i < numIndices; i += 3)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            if (!isIncluded(this->_indices[Vao::TRIANGLE][i + j], this->_indices[Vao::TRIANGLE][(j + 1) % 3 + i]))
            {
                this->_indices[Vao::LINE].push_back(this->_indices[Vao::TRIANGLE][i + j]);
                this->_indices[Vao::LINE].push_back(this->_indices[Vao::TRIANGLE][(j + 1) % 3 + i]);
                this->_indices[Vao::LINE].push_back(RESTART_PRIMITIVE_INDEX);
            }
        }
    }
}

void Model3D::Component::generatePointCloud()
{
    this->_indices[Vao::POINT].resize(this->_vertices.size());
    std::iota(this->_indices[Vao::POINT].begin(), this->_indices[Vao::POINT].end(), 0);
}