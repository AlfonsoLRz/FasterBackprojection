#version 450

layout(r32ui, binding = 0) uniform uimage3D voxelTex; // Correct image binding

in vec3 worldPos;
flat in uint intensity;

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

    imageAtomicAdd(voxelTex, voxelCoord, 0);
}