#pragma once

#include "AABB.h"
#include "ApplicationState.h"
#include "RenderingShader.h"
#include "Vao.h"

class RenderingShader;

#define glAssert(x) do { if (!(x)) { std::cerr << "OpenGL error at " << __FILE__ << ":" << __LINE__ << std::endl; } } while (0)

class Model3D
{
	friend class SceneContent;

protected:
	struct Material
	{
		glm::vec4	_kdColor;
		glm::vec3	_ksColor;
		float		_metallic, _roughnessK;
		glm::vec3	_pointColor;
		glm::vec3	_lineColor;

		Material() :
			_kdColor(1.00, 0.81, 0.29, 1.0f), _ksColor(.5f),
			_metallic(.7f), _roughnessK(.3f), _pointColor(.0f), _lineColor(.0f) {}
	};

public:
	class Component
	{
	public:
		bool						_enabled;
		std::string					_name;

		std::vector<Vao::Vertex>	_vertices;
		std::vector<GLuint>			_indices[3];
		Vao*						_vao;
		AABB						_aabb;

		Material					_material;

		float						_lineWidth, _pointSize;
		bool						_activeRendering[3];

		Component(Vao* vao = nullptr) {
			_enabled = true; _vao = vao; _pointSize = 3.0f; _lineWidth = 1.0f;
			for (bool& i : _activeRendering) i = true;
		}
		virtual ~Component() { delete _vao; _vao = nullptr; }

		void buildVao();
		void completeTopology();
		void generateWireframe();
		void generatePointCloud();
	};

	struct MatrixRenderInformation
	{
		enum MatrixType : uint8_t { MODEL, VIEW, VIEW_PROJECTION };

		glm::mat4				_matrix[VIEW_PROJECTION + 1];
		std::vector<glm::mat4>	_heapMatrices[VIEW_PROJECTION + 1];

		MatrixRenderInformation();
		glm::mat4 multiplyMatrix(MatrixType tMatrix, const glm::mat4& matrix) { this->saveMatrix(tMatrix); return _matrix[tMatrix] *= matrix; }
		void saveMatrix(MatrixType tMatrix) { _heapMatrices[tMatrix].push_back(_matrix[tMatrix]); }
		void setMatrix(MatrixType tMatrix, const glm::mat4& matrix) { this->saveMatrix(tMatrix); _matrix[tMatrix] = matrix; }
		void undoMatrix(MatrixType type);
	};

protected:
	static const std::string				BINARY_EXTENSION;
	static std::unordered_set<std::string>	USED_NAMES;

protected:
	AABB						_aabb;
	std::vector<Component>		_components;
	glm::mat4					_modelMatrix;

protected:
	void calculateAABB();
	void clearData();
	void loadModelBinaryFile(const std::string& path);
	void writeBinaryFile(const std::string& path);

public:
	Model3D();
	virtual ~Model3D();

	virtual void draw(MatrixRenderInformation* matrixInformation, ApplicationState* appState);
	virtual AABB getAABB() const { return _aabb; }
	glm::mat4 getModelMatrix() const { return _modelMatrix; }
	Model3D* moveGeometryToOrigin(const glm::mat4& origMatrix = glm::mat4(1.0f), float maxScale = FLT_MAX);
	Model3D* overrideModelName();
	Model3D* setModelMatrix(const glm::mat4& modelMatrix) { _modelMatrix = modelMatrix; return this; }
	Model3D* setLineColor(const glm::vec3& color);
	Model3D* setPointColor(const glm::vec3& color);
	Model3D* setTriangleColor(const glm::vec4& color);
	Model3D* setLineWidth(float width);
	Model3D* setPointSize(float size);
	Model3D* setTopologyVisibility(Vao::IBO topology, bool visible);
};

