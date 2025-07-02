#version 460
#extension GL_NV_gpu_shader5 : enable


#include <assets/shaders/structs.glsl>

layout(std430, binding = 0) restrict readonly buffer NodeBuffer 
{
    BvhNode nodes[];
};

uniform mat4 mModelViewProj;

vec4[8] computeCorners(uint index)
{
    vec3 vmin = nodes[index].minPoint;
    vec3 vmax = nodes[index].maxPoint;
    vec4 corners[8] = vec4[](
        mModelViewProj * vec4(vmin.x, vmin.y, vmin.z, 1.0),
        mModelViewProj * vec4(vmin.x, vmax.y, vmin.z, 1.0),
        mModelViewProj * vec4(vmin.x, vmin.y, vmax.z, 1.0),
        mModelViewProj * vec4(vmin.x, vmax.y, vmax.z, 1.0),
        mModelViewProj * vec4(vmax.x, vmin.y, vmin.z, 1.0),
        mModelViewProj * vec4(vmax.x, vmax.y, vmin.z, 1.0),
        mModelViewProj * vec4(vmax.x, vmin.y, vmax.z, 1.0),
        mModelViewProj * vec4(vmax.x, vmax.y, vmax.z, 1.0)
    );

    return corners;
}

vec4[24] computeLineVertices(vec4[8] corners)
{
    vec4 vertices[24] = vec4[](
        corners[0], corners[1],
        corners[2], corners[3],
        corners[4], corners[5],
        corners[6], corners[7],

        corners[0], corners[2],
        corners[1], corners[3],
        corners[4], corners[6],
        corners[5], corners[7],

        corners[0], corners[4],
        corners[1], corners[5],
        corners[2], corners[6],
        corners[3], corners[7]
    );

    return vertices;
}

void main() 
{
    uint meshletIndex = gl_VertexID / 24;
    vec4 corners[8] = computeCorners(meshletIndex);
    vec4 vertices[24] = computeLineVertices(corners);

    gl_Position = vertices[gl_VertexID % 24];
}