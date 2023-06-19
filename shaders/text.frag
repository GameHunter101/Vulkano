#version 450
#extension GL_EXT_nonuniform_qualifier:require
// #extension GL_EXT_debug_printf:enable

layout(location=0)out vec4 theColor;

layout(location=0)in vec2 texcoord;
layout(location=1)in vec3 color;
layout(location=2)flat in uint texture_id;

layout(set=0,binding=0)uniform sampler2D lettertextures[];

void main(){
    // debugPrintfEXT("hello! %f",color[0]);
    theColor=vec4(color,texture(lettertextures[texture_id],texcoord));
}