#version 450
#extension GL_EXT_debug_printf:enable

layout(location=0)out vec4 outColor;

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

float sdfSphere(vec3 point,vec3 center,float radius){
    return length(point-center)-radius;
}

float sdfBox(vec3 point,vec3 dim){
    vec3 d = abs(point) - dim;
    return min(max(d.x,max(d.y,d.z)),0.0)+length(max(d,0.0));
}

float map(in vec3 p){
    float displacement=sin(5.*p.x)*cos(5.*p.y)*tan(5.*p.z)*.25;
    float sphere_0=sdfSphere(p,vec3(0.),1.);
    float box_0=sdfBox(p,vec3(1.0,3.0,0.5));
    return max(-sphere_0,box_0);
}

vec3 calculateNormal(in vec3 p){
    const vec3 smallStep=vec3(.001,0.,0.);
    float gradientX=map(p+smallStep.xyy)-map(p-smallStep.xyy);
    float gradientY=map(p+smallStep.yxy)-map(p-smallStep.yxy);
    float gradientZ=map(p+smallStep.yyx)-map(p-smallStep.yyx);
    
    vec3 normal=vec3(gradientX,gradientY,gradientZ);
    
    return normalize(normal);
}

vec3 rayMarch(in vec3 rayOrigin,in vec3 rayDirection){
    float totalDistanceTraveled=0.;
    const int NUMBER_OF_STEPS=32;
    const float MINIUM_HIT_DISTANCE=.001;
    const float MAXIMUM_TRACE_DISTANCE=1000.;
    
    for(int i=0;i<NUMBER_OF_STEPS;i++){
        vec3 currentPosition=rayOrigin+totalDistanceTraveled*rayDirection;
        
        float distanceToClosest=map(currentPosition);
        
        if(distanceToClosest<MINIUM_HIT_DISTANCE){
            vec3 normal=calculateNormal(currentPosition);
            
            vec3 lightPosition=vec3(2.,-5.,3.);
            
            vec3 directionToLight=normalize(currentPosition-lightPosition);
            
            float diffuseIntensity=max(0.,dot(normal,directionToLight));
            
            return vec3(1.,0.,0.)*diffuseIntensity;
        }
        if(totalDistanceTraveled>MAXIMUM_TRACE_DISTANCE){
            break;
        }
        
        totalDistanceTraveled+=distanceToClosest;
    }
    return vec3(0.);
}

void main(){
    
    vec2 uv=(gl_FragCoord.xy/min(dimensions.x,dimensions.y))*2.-1.;
    uv.y=-uv.y;
    
    vec3 cameraPosition=vec3(0.,-0.,-5.);
    vec3 rayOrigin=cameraPosition;
    vec3 rayDirection=vec3(uv,1.);
    
    vec3 shadedColor=rayMarch(rayOrigin,rayDirection);
    
    outColor=vec4(shadedColor,1.);
    // outColor=vec4(uv.x,uv.y,0.,1.);
    
    /* vec2 pixelCoord=gl_FragCoord.xy*dimensions;
    int pixelIndex=int(pixelCoord.y*dimensions.x+pixelCoord.x);
    // debugPrintfEXT("index:%i",pixelIndex);
    
    int randomState=pixelIndex;
    
    float r=randomValue(randomState);
    float g=randomValue(randomState);
    float b=randomValue(randomState);
    theColor=vec4(r,g,b,1); */
}