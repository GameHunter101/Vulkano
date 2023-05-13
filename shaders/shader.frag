#version 450
#extension GL_EXT_debug_printf:enable

layout(location=0)out vec4 theColor;

layout(location=0)in vec3 color_in;
layout(location=1)in vec3 normal;
layout(location=2)in vec3 worldpos;
layout(location=3)in vec3 camera_coordinates;
layout(location=4)in float roughness;
layout(location=5)in float metallic;
layout(location=6)in vec2 dimensions;

// layout(set=0,binding=1)uniform UniformBufferObject{
    //     vec2 window_size;
// }ubo;

float randomValue(inout int state){
    state=state*747796405+2891336453;
    int result=((state>>((state>>28)+4))^state)*277803737;
    return result/4294967295.;// (2^32 - 1)
}

void main(){
    vec2 pixelCoord=gl_FragCoord.xy*dimensions;
    int pixelIndex=int(pixelCoord.y*dimensions.x+pixelCoord.x);
    // debugPrintfEXT("index:%i",pixelIndex);
    
    int randomState=pixelIndex;
    
    float r=randomValue(randomState);
    float g=randomValue(randomState);
    float b=randomValue(randomState);
    theColor=vec4(r,g,b,1);
}