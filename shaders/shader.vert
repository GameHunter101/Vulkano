#version 450

layout (location=0) in vec4 position;

layout(location = 0) out vec4 color_data_for_the_fragment_shader;

void main() {
    gl_PointSize = 10.0;
    gl_Position = position;
    color_data_for_the_fragment_shader = vec4(0.0, 0.6, 1.0, 1.0);
}