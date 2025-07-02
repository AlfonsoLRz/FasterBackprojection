#version 450

in flat uint intensity;

out vec4 fColor;

void main ()
{
	fColor = vec4(vec3(float(intensity) / 255.0), 1.0f);
}