#version 450

// ********** PARAMETERS & VARIABLES ***********

in vec2 textCoord;

uniform sampler2D texSampler;

layout (location = 0) out vec4 fColor;

void main(void)
{
	fColor = vec4(texture(texSampler, textCoord).rgb, 1.0f);
}
