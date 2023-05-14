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

layout(location=0)out vec3 color_data_for_the_fragment_shader;
layout(location=1)out vec3 out_normal;
layout(location=2)out vec4 worldpos;
layout(location=3)out vec3 camera_coordinates;
layout(location=4)out float metallic;
layout(location=5)out float roughness;
layout(location=6)out vec2 dimensions;
layout(location=7)out mat3 camera_rotation;

void main(){
    metallic=metallic_in;
    roughness=roughness_in;
    
    camera_coordinates=vec3(ubo.view_matrix[3]);
    camera_rotation = mat3(ubo.view_matrix);
    /* -ubo.view_matrix[3][0]*vec3(ubo.view_matrix[0][0],ubo.view_matrix[1][0],ubo.view_matrix[2][0])
    -ubo.view_matrix[3][1]*vec3(ubo.view_matrix[0][1],ubo.view_matrix[1][1],ubo.view_matrix[2][1])
    -ubo.view_matrix[3][2]*vec3(ubo.view_matrix[0][2],ubo.view_matrix[1][2],ubo.view_matrix[2][2]); */
    
    dimensions=vec2(ubo.window[0][0],ubo.window[0][1]);
    
    worldpos=model_matrix*vec4(position,1.);
    gl_Position=vec4(position,1.);
    // debugPrintfEXT("position: %f, %f, %f, %f",position);
    // debugPrintfEXT("gl_pos: %f, %f, %f, %f",gl_Position[0],gl_Position[1],gl_Position[2],gl_Position[3]);
    color_data_for_the_fragment_shader=color;
    out_normal=transpose(mat3(inverse_model_matrix))*normal;
}