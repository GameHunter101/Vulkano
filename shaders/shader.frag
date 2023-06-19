#version 450
#pragma shader_stage(fragment)
#pragma spirv(ShaderClockKHR)
// #extension GL_EXT_debug_printf:enable
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
const float FOV=60*PI/180;
const float PHI=1.61803398874989484820459;
const int MAXIMUM_LIGHT_BOUNCES=2;
const bool global_illumination=true;

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

LightData light=LightData(vec3(2.,1.,3.),vec3(1.));
LightData light2=LightData(vec3(-2.,1.,-3.),vec3(.7f,.14f,.42f));

float randomValue(in vec2 xy,in float seed){
    return fract(tan(distance(xy*PHI,xy)*seed)*xy.x);
}

Surface sdfSphere(vec3 point,vec3 center,float radius,vec3 color){
    return Surface(length(point-center)-radius,color);
}

Surface sdfBox(vec3 point,vec3 dim,vec3 color){
    vec3 d=abs(point)-dim;
    float signedDistance=min(max(d.x,max(d.y,d.z)),0.)+length(max(d,0.));
    return Surface(signedDistance,color);
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

Surface objectIntersect(Surface a, Surface b) {
    float intersect = a.signedDistance;
    vec3 color = a.color;
    if (b.signedDistance > a.signedDistance) {
        intersect = b.signedDistance;
        color = b.color;
    }
    return Surface(intersect, color);
}

Surface objectDifference(Surface a,Surface b){
    float difference = objectIntersect(a,Surface(-b.signedDistance,b.color)).signedDistance;
    return Surface(difference,a.color);
}

vec4 map(in vec3 p){
    Surface sphere_0=sdfSphere(p,vec3(0.),1.,vec3(0,1,0));
    Surface sphere_1=sdfSphere(p,vec3(2.,0.,3.),.8,vec3(.8706,.4078,.9137));
    Surface box_0=sdfBox(p,vec3(1.,3.,.5),vec3(.71f,.16f,.25f));
    vec3 point0=vec3(0.,-1.,0.);
    vec3 point1=vec3(1.,-1.,-.5);
    vec3 point2=vec3(-1.,-1.,-.5);
    vec3 n=normalize(cross(point1-point0,point2-point0));
    Surface plane_0=sdfPlane(p,vec4(n.x,n.y,n.z,-dot(n,point0)),vec3(.13,.6,.79));
    int steps=4;
    Surface mandelbulb=sdfMandelbulb(p,steps,vec3(.92,.58,.86));
    
    Surface sdfs[MAXIMUM_OBJECTS_COUNT];
    
    sdfs[0]=plane_0;
    sdfs[1]=objectDifference(box_0,sphere_0);
    sdfs[2]=sphere_1;
    // sdfs[3] = box_0;
    // sdfs[1] = mandelbulb;
    
    // return min(min(plane_0,max(-sphere_0,box_0)),sphere_1);
    return betterMin(sdfs,3);
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

vec3 calculateLight(vec3 rayOrigin,vec3 normal,LightData light){
    
    float minAngle=lightMarch(rayOrigin+normal*MINIUM_HIT_DISTANCE*10,light.position);
    vec3 directionToLight=normalize(light.position-rayOrigin);
    float diffuseIntensity=max(0.,dot(normal,directionToLight));
    
    return light.color*diffuseIntensity*(min(1.,minAngle/(PI/100.)));
}

vec3 lerp(vec3 a,vec3 b,float t){
    return(1-t)*a+t*b;
}

vec3 globalIllumination(vec3 rayOrigin,vec3 bounceNormal,vec3 originalDirection,float roughness){
    vec3 accumulatedColor=vec3(1.);
    
    vec3 bounceOrigin=rayOrigin;
    
    for(int i=0;i<MAXIMUM_LIGHT_BOUNCES;i++){
        vec2 pixelCoord=gl_FragCoord.xy*dimensions;
        int pixelIndex=int(pixelCoord.y*dimensions.x+pixelCoord.x);
        
        float seed=fract(float(clockARB()/1000000000));
        float rand1=randomValue(gl_FragCoord.xy,seed+mod(float(clockARB())/1000000000,20));
        float rand2=randomValue(gl_FragCoord.xy,rand1);
        float azimuthalAngle=rand1*2*PI;
        float polarAngle=acos(2*rand2-1);
        vec3 diffuseDirection=vec3(sin(polarAngle)*cos(azimuthalAngle),sin(polarAngle)*sin(azimuthalAngle),cos(polarAngle));
        vec3 smoothDirection=originalDirection-2.*dot(originalDirection,bounceNormal)*bounceNormal;
        float isAligned=sign(dot(diffuseDirection,bounceNormal));
        diffuseDirection*=isAligned;
        vec3 rayDirecton=lerp(diffuseDirection,smoothDirection,max(roughness,.005));
        
        float totalDistanceTraveled=0.;
        
        for(int i=0;i<NUMBER_OF_STEPS/10;i++){
            vec3 currentPosition=bounceOrigin+totalDistanceTraveled*rayDirecton+rayDirecton*.5;
            vec4 mapResult=map(currentPosition);
            float distanceToClosest=mapResult.w;
            vec3 colorOfClosest=vec3(mapResult);
            
            if(distanceToClosest<MINIUM_HIT_DISTANCE){
                vec3 normal=calculateNormal(currentPosition);
                vec3 lightAccumulated=colorOfClosest+calculateLight(currentPosition,normal,light)+calculateLight(currentPosition,normal,light2);
                vec3 color=lightAccumulated;
                accumulatedColor*=color;
                // accumulatedColor=vec3(1.);
                bounceOrigin=currentPosition;
                // break;
            }
            
            if(totalDistanceTraveled>MAXIMUM_TRACE_DISTANCE/20){
                accumulatedColor=vec3(0.);
                break;
            }
            
            totalDistanceTraveled+=distanceToClosest;
        }
        
    }
    return accumulatedColor;
}

vec3 rayMarch(vec3 rayOrigin,vec3 rayDirection){
    float totalDistanceTraveled=0.;
    float anglePerPixel=FOV/dimensions.y;
    vec3 color=vec3(0.,0.,0.);
    float alpha=0.;
    
    float tracePrev1=100;
    float tracePrev2=100;
    
    for(int i=0;i<NUMBER_OF_STEPS;i++){
        vec3 currentPosition=rayOrigin+totalDistanceTraveled*rayDirection;
        
        vec4 mapResult=map(currentPosition);
        float distanceToClosest=mapResult.w;
        vec3 colorOfClosest=vec3(mapResult);
        float currentTraceAngle=atan(distanceToClosest/totalDistanceTraveled);
        // blending coefficient is wrong
        float blendingCoefficient=(1-min(1,currentTraceAngle/anglePerPixel))*(1-alpha);
        
        // Accumulate light and color data
        if(tracePrev2>tracePrev1&&tracePrev1<currentTraceAngle&&currentTraceAngle<anglePerPixel||distanceToClosest<MINIUM_HIT_DISTANCE){
            if(distanceToClosest<MINIUM_HIT_DISTANCE){
                blendingCoefficient=1-alpha;
            }
            vec3 normal=calculateNormal(currentPosition);
            
            vec3 lightAccumulated=colorOfClosest+calculateLight(currentPosition,normal,light)+calculateLight(currentPosition,normal,light2);
            
            if(global_illumination){
                lightAccumulated+=globalIllumination(currentPosition,normal,rayDirection,0.7);
            }
            
            // color=globalIllumination(currentPosition,normal,rayDirection,0.);
            color=lerp(color,(lightAccumulated)*calculateAmbientOcclusion(currentPosition,normal),blendingCoefficient);
            alpha+=blendingCoefficient;
        }
        
        // Final pixel return color
        if(distanceToClosest<MINIUM_HIT_DISTANCE){
            return color;
        }
        if(totalDistanceTraveled>MAXIMUM_TRACE_DISTANCE){
            break;
        }
        
        tracePrev2=tracePrev1;
        tracePrev1=currentTraceAngle;
        
        totalDistanceTraveled+=distanceToClosest;
    }
    // blend with alpha
    return lerp(color,vec3(0.,0.,.1),1-alpha);
}

void main(){
    
    // TODO: fragcoord doesn't like stretching on the x axis
    vec2 uv=(gl_FragCoord.xy/min(dimensions.x,dimensions.y))*2.-1.;
    uv.y=-uv.y;
    
    vec3 cameraPosition=camera_coordinates;
    vec3 rayOrigin=cameraPosition;
    vec3 rayDirection=normalize(camera_rotation*vec3(uv,1/tan(FOV/2)));
    
    vec3 shadedColor=rayMarch(rayOrigin,rayDirection);
    
    outColor=vec4(shadedColor,1.);
    
}