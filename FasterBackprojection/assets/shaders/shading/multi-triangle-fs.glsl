#version 450

in flat float intensity;

out vec4 fColor;

void main ()
{
	fColor = vec4(vec3(intensity), 1.0f);
}