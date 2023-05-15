#version 450
#extension GL_EXT_debug_printf:enable

layout(location=0)out vec4 outColor;

layout(location=0)in vec3 color_in;
layout(location=1)in vec3 camera_coordinates;
layout(location=2)in vec2 dimensions;
layout(location=3)in mat3 camera_rotation;

const int NUMBER_OF_STEPS=1000;
const float MINIUM_HIT_DISTANCE=.001;
const float MAXIMUM_TRACE_DISTANCE=1000.;
const float PI=3.141592;

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
    vec3 d=abs(point)-dim;
    return min(max(d.x,max(d.y,d.z)),0.)+length(max(d,0.));
}

float sdfPlane(vec3 point,vec4 planeCoefficients){
    return dot(point,planeCoefficients.xyz)+planeCoefficients.w;
}

float map(in vec3 p){
    float sphere_0=sdfSphere(p,vec3(0.),1.);
    float sphere_1=sdfSphere(p,vec3(2.,0.,3.),.8);
    float box_0=sdfBox(p,vec3(1.,3.,.5));
    vec3 point0=vec3(0.,-1.,0.);
    vec3 point1=vec3(1.,-1.,-.5);
    vec3 point2=vec3(-1.,-1.,-.5);
    vec3 n=normalize(cross(point1-point0,point2-point0));
    float plane_0=sdfPlane(p,vec4(n.x,n.y,n.z,-dot(n,point0)));
    return min(min(plane_0,max(-sphere_0,box_0)),sphere_1);
}

vec3 calculateNormal(in vec3 p){
    const vec3 smallStep=vec3(.0001,0.,0.);
    float gradientX=map(p+smallStep.xyy)-map(p-smallStep.xyy);
    float gradientY=map(p+smallStep.yxy)-map(p-smallStep.yxy);
    float gradientZ=map(p+smallStep.yyx)-map(p-smallStep.yyx);
    
    vec3 normal=vec3(gradientX,gradientY,gradientZ);
    
    return normalize(normal);
}

float calculateAmbientOcclusion(in vec3 p,in vec3 normal){
    // TODO: rename t
    float accumultedOcclusion=0;
    const float sampleSpacing=.1;
    const float aoStrength = 2;
    for(int i=1;i<=5;i++){
        accumultedOcclusion+=exp(-i/2)*(sampleSpacing*i-map(p+normal*sampleSpacing*i));
    }
    return 1- aoStrength*accumultedOcclusion;
}

float lightMarch(in vec3 rayOrigin,in vec3 lightPos){
    float totalDistanceTraveled=0.;
    
    float minAngle=PI;
    vec3 rayDirection=normalize(lightPos-rayOrigin);
    for(int i=0;i<NUMBER_OF_STEPS;i++){
        vec3 currentPosition=rayOrigin+totalDistanceTraveled*rayDirection;
        float lightDist=distance(currentPosition,lightPos);
        float worldDist=map(currentPosition);
        float distanceToClosest=min(lightDist,worldDist);
        minAngle=min(minAngle,atan(worldDist/totalDistanceTraveled));
        
        if(distanceToClosest<MINIUM_HIT_DISTANCE){
            if(lightDist<MINIUM_HIT_DISTANCE){
                break;
            }else{
                return 0.;
            }
        }
        if(totalDistanceTraveled>MAXIMUM_TRACE_DISTANCE){
            return 0.;
        }
        
        totalDistanceTraveled+=distanceToClosest;
    }
    return minAngle;
}

vec3 rayMarch(in vec3 rayOrigin,in vec3 rayDirection){
    float totalDistanceTraveled=0.;
    
    for(int i=0;i<NUMBER_OF_STEPS;i++){
        vec3 currentPosition=rayOrigin+totalDistanceTraveled*rayDirection;
        
        float distanceToClosest=map(currentPosition);
        
        if(distanceToClosest<MINIUM_HIT_DISTANCE){
            vec3 normal=calculateNormal(currentPosition);
            
            vec3 lightPosition=vec3(2.,1.,3.);
            
            vec3 directionToLight=normalize(lightPosition-currentPosition);
            
            float diffuseIntensity=max(0.,dot(normal,directionToLight));
            
            float minAngle=lightMarch(currentPosition+normal*MINIUM_HIT_DISTANCE*10,lightPosition);
            
            // float ambientOcclusion=1-float(i)/(NUMBER_OF_STEPS-1);
            
            return vec3(1.,0.,0.)*diffuseIntensity*(min(1.,minAngle/(PI/100.)))*calculateAmbientOcclusion(currentPosition,normal);
        }
        if(totalDistanceTraveled>MAXIMUM_TRACE_DISTANCE){
            break;
        }
        
        totalDistanceTraveled+=distanceToClosest;
    }
    return vec3(0.,0.,.1);
}

void main(){
    
    vec2 uv=(gl_FragCoord.xy/min(dimensions.x,dimensions.y))*2.-1.;
    uv.y=-uv.y;
    
    vec3 cameraPosition=camera_coordinates;
    vec3 rayOrigin=cameraPosition;
    vec3 rayDirection=normalize(camera_rotation*vec3(uv,sqrt(3)));
    
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