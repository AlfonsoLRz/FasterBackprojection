#version 460

//#extension GL_NV_gpu_shader5 : enable
#extension GL_NV_shader_atomic_float : require

layout(early_fragment_tests) in;
layout(r32f, binding = 0) uniform image3D voxelTex;

in vec3 worldPos;
flat in float intensity;

uniform vec3 voxelMin;
uniform vec3 voxelMax;
uniform ivec3 voxelRes;

void main()
{
    vec3 normalized = (worldPos - voxelMin) / (voxelMax - voxelMin);

    if (any(lessThan(normalized, vec3(0.0))) || any(greaterThanEqual(normalized, vec3(1.0)))) {
        discard;
    }

    ivec3 voxelCoord = ivec3(floor(normalized * vec3(voxelRes)));
    voxelCoord = clamp(voxelCoord, ivec3(0), voxelRes - ivec3(1)); // Safety clamp

    imageAtomicAdd(voxelTex, voxelCoord, intensity);
}