#version 450

uniform vec4 Kd;

out vec4 fragmentColor;

void main ()
{
	fragmentColor = Kd;
}