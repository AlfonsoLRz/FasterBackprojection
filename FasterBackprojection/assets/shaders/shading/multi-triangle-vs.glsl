#version 450

layout(location = 0) in vec3 vPosition;
layout(location = 3) in vec3 vTranslation;
layout(location = 4) in vec3 vScale;
layout(location = 6) in uint vIntensity;

uniform mat4 mModelViewProj;

out flat uint intensity;

void main()
{
    mat4 modelMatrix = mat4(
        vec4(vScale.x, 0.0, 0.0, 0.0),
        vec4(0.0, vScale.y, 0.0, 0.0),
        vec4(0.0, 0.0, vScale.z, 0.0),
        vec4(vTranslation.x, vTranslation.y, vTranslation.z, 1.0)
    );
	vec3 newPosition = vec3(modelMatrix * vec4(vPosition, 1.0f));

	intensity = vIntensity;
	gl_Position = mModelViewProj * vec4(newPosition, 1.0f);
}