#version 460

#extension GL_NV_gpu_shader5 : enable
#extension GL_NV_shader_atomic_float : require

layout(early_fragment_tests) in;
layout(r32f, binding = 0) uniform image3D voxelTex;

in vec3 worldPos;
flat in float intensity;

uniform vec3 voxelMin;
uniform vec3 voxelSize;
uniform ivec3 voxelRes;


void main()
{
    vec3 normalized = (worldPos - voxelMin) / voxelSize;
    if (any(lessThan(normalized, vec3(0.0))) || any(greaterThan(normalized, vec3(1.0)))) 
        discard;

    ivec3 voxelCoord = ivec3(normalized * vec3(voxelRes));
    imageAtomicAdd(voxelTex, voxelCoord, intensity);
}