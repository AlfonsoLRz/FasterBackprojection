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

    ivec3 voxelCoord = ivec3(normalized * vec3(voxelRes - 1) + 0.5);
    imageAtomicAdd(voxelTex, voxelCoord, intensity);

    //for (int x = -1; x <= 1; ++x)
    //    for (int y = -1; y <= 1; ++y)
    //        for (int z = -1; z <= 1; ++z)
    //            imageAtomicAdd(voxelTex, voxelCoord + ivec3(x, y, z), intensity * 0.1);
}