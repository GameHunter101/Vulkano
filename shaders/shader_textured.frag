#version 450

layout (location=0) out vec4 theColor;

layout (location=0) in vec2 uv;

layout (set=1,binding=0) uniform sampler2D texture_sampler;


void main(){
	theColor=texture(texture_sampler, uv);
}