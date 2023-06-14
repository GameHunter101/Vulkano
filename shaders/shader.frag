#version 450
#extension GL_EXT_debug_printf:enable
#extension GL_ARB_shader_clock:enable

layout(location=0)out vec4 outColor;

layout(location=0)in vec3 color_in;
layout(location=1)in vec3 camera_coordinates;
layout(location=2)in vec2 dimensions;
layout(location=3)in mat3 camera_rotation;

const int NUMBER_OF_STEPS=500;
const float MINIUM_HIT_DISTANCE=.001;
const float MAXIMUM_TRACE_DISTANCE=1000.;
const float PI=3.141592;
const uint MAXIMUM_OBJECTS_COUNT=100;

// layout(set=0,binding=1)uniform UniformBufferObject{
    //     vec2 window_size;
// }ubo;

struct LightData{
    vec3 position;
    vec3 color;
};

struct Surface{
    float signedDistance;
    vec3 color;
};

float randomValue(inout int state){
    state=state*747796405+2891336453;
    int result=((state>>((state>>28)+4))^state)*277803737;
    return result/4294967295.;// (2^32 - 1)
}

Surface sdfSphere(vec3 point,vec3 center,float radius, vec3 color){
    return Surface(length(point-center)-radius,color);
}

float sdfBox(vec3 point,vec3 dim){
    vec3 d=abs(point)-dim;
    return min(max(d.x,max(d.y,d.z)),0.)+length(max(d,0.));
}

Surface sdfPlane(vec3 point,vec4 planeCoefficients,vec3 color){
    float signedDistance=dot(point,planeCoefficients.xyz)+planeCoefficients.w;
    return Surface(signedDistance,color);
}

Surface sdfMandelbulb(vec3 pos,out int steps,vec3 color){
    float Power=mod(float(clockARB())/1000000000,20);
    vec3 z=pos;
    float dr=1.;
    float r=0.;
    for(int i=0;i<20;i++){
        r=length(z);
        steps=i;
        if(r>4.)break;
        
        // convert to polar coordinates
        float theta=acos(z.z/r);
        float phi=atan(z.y,z.x);
        dr=pow(r,Power-1.)*Power*dr+1.;
        
        // scale and rotate the point
        float zr=pow(r,Power);
        theta=theta*Power;
        phi=phi*Power;
        
        // convert back to cartesian coordinates
        z=zr*vec3(sin(theta)*cos(phi),sin(phi)*sin(theta),cos(theta));
        z+=pos;
    }
    
    float signedDistance=.5*log(r)*r/dr;
    return Surface(signedDistance,color);
}

vec4 betterMin(Surface sdfArray[MAXIMUM_OBJECTS_COUNT],uint len){
    float min=sdfArray[0].signedDistance;
    vec3 color=sdfArray[0].color;
    for(int i=1;i<len;i++){
        if(sdfArray[i].signedDistance<min){
            min=sdfArray[i].signedDistance;
            color=sdfArray[i].color;
        }
    }
    return vec4(color,min);
}

vec4 map(in vec3 p){
    Surface sphere_0=sdfSphere(p,vec3(0.),1.,vec3(0,1,0));
    Surface sphere_1= sdfSphere(p,vec3(2.,0.,3.),.8,vec3(0.8706, 0.4078, 0.9137));
    float box_0=sdfBox(p,vec3(1.,3.,.5));
    vec3 point0=vec3(0.,-1.,0.);
    vec3 point1=vec3(1.,-1.,-.5);
    vec3 point2=vec3(-1.,-1.,-.5);
    vec3 n=normalize(cross(point1-point0,point2-point0));
    Surface plane_0=sdfPlane(p,vec4(n.x,n.y,n.z,-dot(n,point0)),vec3(.13,.6,.79));
    int steps=4;
    Surface mandelbulb=sdfMandelbulb(p,steps,vec3(.92,.58,.86));
    
    Surface sdfs[MAXIMUM_OBJECTS_COUNT];

    sdfs[0] = plane_0;
    sdfs[1] = sphere_0;
    // sdfs[1] = mandelbulb;
    
    // return min(min(plane_0,max(-sphere_0,box_0)),sphere_1);
    return betterMin(sdfs,2);
}

vec3 calculateNormal(in vec3 p){
    const vec3 smallStep=vec3(.0001,0.,0.);
    float gradientX=map(p+smallStep.xyy).w-map(p-smallStep.xyy).w;
    float gradientY=map(p+smallStep.yxy).w-map(p-smallStep.yxy).w;
    float gradientZ=map(p+smallStep.yyx).w-map(p-smallStep.yyx).w;
    
    vec3 normal=vec3(gradientX,gradientY,gradientZ);
    
    return normalize(normal);
}

float calculateAmbientOcclusion(in vec3 p,in vec3 normal){
    // TODO: rename t
    float accumultedOcclusion=0;
    const float sampleSpacing=.1;
    const float aoStrength=2;
    for(int i=1;i<=5;i++){
        accumultedOcclusion+=exp(-i/2)*(sampleSpacing*i-map(p+normal*sampleSpacing*i)).w;
    }
    return 1-aoStrength*accumultedOcclusion;
}

float lightMarch(in vec3 rayOrigin,in vec3 lightPos){
    float totalDistanceTraveled=0.;
    
    float minAngle=PI;
    vec3 rayDirection=normalize(lightPos-rayOrigin);
    for(int i=0;i<NUMBER_OF_STEPS;i++){
        vec3 currentPosition=rayOrigin+totalDistanceTraveled*rayDirection;
        float lightDist=distance(currentPosition,lightPos);
        float worldDist=map(currentPosition).w;
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

vec3 calculateLight(in vec3 rayOrigin,in vec3 normal,in LightData light){
    
    float minAngle=lightMarch(rayOrigin+normal*MINIUM_HIT_DISTANCE*10,light.position);
    vec3 directionToLight=normalize(light.position-rayOrigin);
    float diffuseIntensity=max(0.,dot(normal,directionToLight));
    return light.color*diffuseIntensity*(min(1.,minAngle/(PI/100.)));
}

vec3 rayMarch(in vec3 rayOrigin,in vec3 rayDirection){
    float totalDistanceTraveled=0.;
    
    for(int i=0;i<NUMBER_OF_STEPS;i++){
        vec3 currentPosition=rayOrigin+totalDistanceTraveled*rayDirection;
        
        vec4 mapResult=map(currentPosition);
        float distanceToClosest=mapResult.w;
        vec3 colorOfClosest= vec3(mapResult);
        
        if(distanceToClosest<MINIUM_HIT_DISTANCE){
            vec3 normal=calculateNormal(currentPosition);
            
            vec3 lightPosition=vec3(2.,1.,3.);
            
            // vec3 directionToLight=normalize(lightPosition-currentPosition);
            
            // float ambientOcclusion=1-float(i)/(NUMBER_OF_STEPS-1);
            
            LightData light=LightData(vec3(2.,1.,3.),vec3(1.0));
            LightData light2=LightData(vec3(-2.,1.,-3.),vec3(0.77f, 0.69f, 0.58f));
            
            vec3 lightAccumulated=colorOfClosest+calculateLight(currentPosition,normal,light)+calculateLight(currentPosition,normal,light2);
            
            return lightAccumulated*calculateAmbientOcclusion(currentPosition,normal);
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
    // debugPrintfEXT("time: %u",clockARB());
    
    /* vec2 pixelCoord=gl_FragCoord.xy*dimensions;
    int pixelIndex=int(pixelCoord.y*dimensions.x+pixelCoord.x);
    debugPrintfEXT("index:%i",pixelIndex);
    
    int randomState=pixelIndex;
    
    float r=randomValue(randomState);
    float g=randomValue(randomState);
    float b=randomValue(randomState);
    theColor=vec4(r,g,b,1); */
}