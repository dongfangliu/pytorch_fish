import numpy as np
from gym_fish.envs.entities.underwater_agent import underwater_agent
import moderngl
class mesh:
    def __init__(self,agent:underwater_agent,ctx:moderngl.Context,prog:moderngl.Program) -> None:
        self.set_agent(agent)
        self.update_mesh_data()
        self.vbo = ctx.buffer(self.get_vertices_data().astype('f4').tobytes(),dynamic=True)
        self.index_buffer = ctx.buffer(self.indices.astype('int').tobytes())
        self.vao = ctx.simple_vertex_array(program=prog, buffer = self.vbo,index_buffer=self.index_buffer, attributes=['in_pos', 'in_normal','in_uv'])
    def set_agent(self,agent:underwater_agent)->None:
        self._agent = agent
    def update_mesh_data(self):
        if self._agent!=None:
            data = self._agent._dynamics.getRenderData()
            self.pos = data.pos
            self.normal = data.normal
            self.uv = data.uv
            self.indices = data.indices
    def get_vertices_data(self):
        return np.dstack([self.pos[:,0],self.pos[:,1],self.pos[:,2], self.normal[:,0],self.normal[:,1],self.normal[:,2],self.uv[:,0],self.uv[:,1]])