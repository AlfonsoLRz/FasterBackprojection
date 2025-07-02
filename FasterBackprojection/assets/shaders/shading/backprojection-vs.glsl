#version 450

layout(location = 0) in vec3 vPosition;
layout(location = 3) in vec3 vTranslation;
layout(location = 4) in vec3 vScale;
layout(location = 6) in uint vIntensity;

uniform vec3 voxelMin, voxelMax;
uniform mat4 voxelModelMatrix; 
uniform mat4 voxelProjectionMatrix; 

out vec3 worldPos;
out flat uint intensity;

void main()
{
    mat4 modelMatrix = mat4(
        vec4(vScale.x, 0.0, 0.0, 0.0),
        vec4(0.0, vScale.y, 0.0, 0.0),
        vec4(0.0, 0.0, vScale.z, 0.0),
        vec4(vTranslation.x, vTranslation.y, vTranslation.z, 1.0)
    );

    vec4 world = voxelModelMatrix * modelMatrix * vec4(vPosition, 1.0);
    worldPos = world.xyz;
    intensity = vIntensity;

    gl_Position = voxelProjectionMatrix * world; // Correct transformation to clip space
}