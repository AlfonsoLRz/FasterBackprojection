#pragma once

#include "Model3D.h"

class DrawPointCloud : public Model3D 
{
protected:
    RenderingShader* _pointCloudShader;

public:
    DrawPointCloud(const std::vector<glm::vec3>& points);
    DrawPointCloud(const DrawPointCloud& drawPointCloud) = delete;
    ~DrawPointCloud() override = default;

    void draw(MatrixRenderInformation* matrixInformation, ApplicationState* appState) override;
};

