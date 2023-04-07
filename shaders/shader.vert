#version 450

layout(location = 0) in vec4 position;
layout(location = 1) in float size;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 color_data_for_the_fragment_shader;

void main() {
    gl_PointSize = size;
    gl_Position = position;
    color_data_for_the_fragment_shader = color;
}