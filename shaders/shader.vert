#version 450
#extension GL_EXT_debug_printf:enable

layout(location=0)in vec3 position;
layout(location=1)in vec3 normal;
layout(location=2)in mat4 model_matrix;
layout(location=6)in mat4 inverse_model_matrix;
layout(location=10)in vec3 color;
layout(location=11)in float metallic_in;
layout(location=12)in float roughness_in;

layout(set=0,binding=0)uniform UniformBufferObject{
    mat4 view_matrix;
    // mat4 projection_matrix;
    mat4 window;
}ubo;

layout(location=0)out vec3 color_in;
layout(location=1)out vec3 camera_coordinates;
layout(location=2)out vec2 dimensions;
layout(location=3)out mat3 camera_rotation;

void main(){
    
    camera_coordinates=vec3(ubo.view_matrix[3]);
    camera_rotation = mat3(ubo.view_matrix);
    
    dimensions=vec2(ubo.window[0][0],ubo.window[0][1]);
    
    gl_Position=vec4(position,1.);
    
    color_in=color;
    // out_normal=transpose(mat3(inverse_model_matrix))*normal;
}