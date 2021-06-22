import numpy as np 
from .mesh import mesh
from gym_fish.envs.entities.underwater_agent import underwater_agent
from .light import pointLight
from .camera import camera
from PIL import Image
import moderngl
class renderer:
    def __init__(self,camera:camera) -> None:
        self.camera =camera
        self.meshes = []
        self.light = pointLight((0,0,0),(1,1,1,0.25))
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(
            vertex_shader='''
            #version 330 core
            layout (location = 0) in vec3 in_pos;
            layout (location = 1) in vec3 in_normal;
            layout (location = 2) in vec2 in_uv;

            out vec3 v_vert;
            out vec3 v_norm;
            out vec2 v_text;

            uniform mat4 mvp;

            void main()
            {
                v_vert = in_pos;
                v_norm = in_normal;  
                v_text = in_uv;
                gl_Position = mvp * vec4(in_pos, 1.0);
            }   
            ''',
    fragment_shader='''
                uniform vec3 obj_color;
                uniform vec4 light_color;
                uniform vec3 light_pos;
                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;
                out vec4 f_color;
                void main() {
                    float lum = dot(normalize(v_norm), normalize(v_vert - light_pos));
                    lum = acos(lum) / 3.14159265;
                    lum = clamp(lum, 0.0, 1.0);
                    lum = lum * lum;
                    lum = smoothstep(0.0, 1.0, lum);
                    lum *= smoothstep(0.0, 80.0, v_vert.z) * 0.3 + 0.7;
                    lum = lum * 0.8 + 0.2;
                    vec3 color = obj_color;
                    color = color * (1.0 - light_color.a) + light_color.rgb * light_color.a;
                    f_color = vec4(color * lum, 1.0);
                }
    ''',
        )        

        self.fbo = self.ctx.simple_framebuffer(self.camera.window_size)

    def add_mesh(self,agent:underwater_agent):
        self.meshes.append(mesh(agent))
    def add_light(self,light:pointLight):
        self.light = light
    def render(self):
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)

        self.prog['light_pos'].value = self.light.pos
        self.prog['light_color'] = self.light.color
        self.prog['obj_color'] = (1,0,0)
        self.prog['mvp'].write(self.camera.mat_projection.astype('f4'))

        for mesh in self.meshes:
            mesh.vao.render(moderngl.TRIANGLES)
        Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1).show()
    
    
