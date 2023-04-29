#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout (location=0) out vec4 theColor;

layout (location=0) in vec2 uv;
layout (location=1) flat in uint texture_id;

layout (set=1,binding=0) uniform sampler2D texture_sampler[];


void main(){
	theColor=texture(texture_sampler[texture_id], uv);
}