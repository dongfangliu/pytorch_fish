#!/usr/bin/env python3


import ctypes
import logging
import math
import os.path

import numpy as np
import OpenGL.GL as gl
import pyrr
from PyQt5.QtWidgets import QApplication,QTextEdit
from PyQt5.QtGui import QOpenGLWindow

from gym_fish.envs.visualization.camera import  camera
from  gym_fish.envs.lib import pyflare as fl

class visualObject:
    def __init__(self):
        self.color = (1,1,1,1)
        self.data = {"position":[],"normal":[],"indices":[]}
    def update(self):
        pass
class visualSphere(visualObject):
    def __init__(self):
        super(visualSphere, self).__init__()
        x_segements = 50
        y_segements=50
        for y in range(y_segements+1):
            for x in range(x_segements+1):
                xSeg = float(x)/x_segements
                ySeg = float(y)/y_segements
                xPos = math.cos(xSeg*2*math.pi)*math.sin(ySeg*math.pi)
                yPos = math.cos(ySeg*math.pi)
                zPos = math.sin(xSeg*2*math.pi)*math.sin(ySeg*math.pi)
                self.data["position"].append((xPos,yPos,zPos))
                self.data["normal"].append((xPos,yPos,zPos))

        for i in range(y_segements):
            for j in range(x_segements):
                self.data['indices'].append(i * (x_segements + 1) + j)
                self.data['indices'].append((i+1) * (x_segements + 1) + j)
                self.data['indices'].append((i+1)  * (x_segements + 1) + (j+1) )
                self.data['indices'].append(i * (x_segements + 1) + j)
                self.data['indices'].append((i+1) * (x_segements + 1) + (j+1))
                self.data['indices'].append(i * (x_segements + 1) + (j+1) )

        self.data['position'] =  np.array(self.data['position']).astype('float32')
        self.data["normal"] =  np.array(self.data["normal"]).astype('float32')
        self.data["indices"] =  np.array(self.data["indices"]).astype('uint32')


    def update(self):
        pass
class visualMesh(visualObject):
    def __init__(self,meshData:fl.RenderData):
        super(visualMesh, self).__init__()
        self.update(meshData)

    def update(self, meshData):
        self.data['position'] = np.array(meshData.pos).astype('float32')
        self.data['normal'] = np.array(meshData.normal).astype('float32')
        self.data['indices'] = np.array(meshData.indices).astype('uint32')
class bufferForVisualObject:
    def __init__(self,obj:visualObject):
        self.init_buffer(obj)

    def init_buffer(self,obj):
        self.VAO = gl.glGenVertexArrays(1);
        self.PVBO = gl.glGenBuffers(1)
        self.NVBO = gl.glGenBuffers(1)
        self.EBO = gl.glGenBuffers(1)
        self.buffer_data(obj)

    def buffer_data(self,obj):
        gl.glBindVertexArray(self.VAO);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.PVBO);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, obj.data["position"].nbytes, obj.data["position"], gl.GL_DYNAMIC_DRAW);
        gl.glEnableVertexAttribArray(0);
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, obj.data["position"].dtype.itemsize*3, ctypes.c_void_p(0));

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.NVBO);
        gl.glBufferData(gl.GL_ARRAY_BUFFER, obj.data["normal"].nbytes, obj.data["normal"], gl.GL_DYNAMIC_DRAW);
        gl.glEnableVertexAttribArray(1);
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, obj.data["normal"].dtype.itemsize*3, ctypes.c_void_p(0));

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.EBO);
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, obj.data["indices"].nbytes, obj.data["indices"],  gl.GL_DYNAMIC_DRAW);
        gl.glBindVertexArray(0);



class CameraWidget:
    pass


class MinimalGLWidget(QOpenGLWindow):
    vertex_code = '''
        attribute vec3 position;
        attribute vec3 normal;
        uniform mat4 model;
        uniform mat3 normal_mat;
        uniform mat4 projection_view;
        uniform vec4 color;

        varying vec4 v_color;
        varying vec3 v_normal;

    void main()
    {

        v_normal =  normal_mat* normal;
        v_color = color;

       gl_Position = projection_view * model*vec4(position,1.0);
    }
    '''

    fragment_code = '''
    varying vec4 v_color;
    varying vec3 v_normal;
    // entry point
    void main()
    {
        gl_FragColor =   v_color;
    }
    '''

    def __init__(self,skeleton_file:str):
        super(MinimalGLWidget, self).__init__()
        self.cam = camera()
        self.cam.center=(0.0,0.0,1)
        self.skeleton = fl.skeletonFromJson(skeleton_file, 0)

    def initializeGL(self):
        gl.glEnable(gl.GL_ALPHA_TEST)
        gl.glEnable(gl.GL_BLEND);
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
        self.program = self.initShaderProgram()

        gl.glUseProgram(self.program)

        self.sphere = visualSphere()
        self.sphere_buffer = bufferForVisualObject(self.sphere)
        mesh = visualMesh(meshData=self.skeleton.getRenderData())
        self.mesh_buffer = bufferForVisualObject(mesh)

    def initShaderProgram(self):
        program = gl.glCreateProgram()
        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        # Set shaders source
        gl.glShaderSource(vertex, self.vertex_code)
        gl.glShaderSource(fragment, self.fragment_code)
        # Compile shaders
        gl.glCompileShader(vertex)
        if not gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(vertex).decode()
            raise RuntimeError("Vertex shader compilation error: %s", error)
        gl.glCompileShader(fragment)
        if not gl.glGetShaderiv(fragment, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(fragment).decode()
            print(error)
            raise RuntimeError("Fragment shader compilation error")
        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            print(gl.glGetProgramInfoLog(program))
            raise RuntimeError('Linking error')
        gl.glDetachShader(program, vertex)
        gl.glDetachShader(program, fragment)
        return program

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        loc = gl.glGetUniformLocation(self.program, "projection_view")
        gl.glUniformMatrix4fv(loc, 1, False, self.cam.viewProejction.astype('f4'))

        self.skeleton.update(True,True)
        for joint in self.skeleton.joints:
            # model_mat = pyrr.matrix44.create_from_translation((1,0,0))
            model_mat = joint.node.getWorldTrans()
            model_mat[0,0]= 0.01
            model_mat[1, 1] = 0.01
            model_mat[2, 2] = 0.01
            color = (0, 0.4, 0.2, 0.4)
            self.draw(model_mat, color, self.sphere_buffer.VAO, len(self.sphere.data['indices']))
        for link in self.skeleton.links:
            model_mat = link.getWorldTrans()
            model_mat[:,0] = model_mat[:,0]*link.size[0]
            model_mat[:,1] = model_mat[:,1]*link.size[1]
            model_mat[:,2] = model_mat[:,2]*link.size[2]
            color = (0, 0.2, 0.4, 0.4)
            self.draw(model_mat,color,self.sphere_buffer.VAO,len(self.sphere.data['indices']))

        mesh = visualMesh(meshData=self.skeleton.getRenderData())
        self.mesh_buffer.buffer_data(mesh)
        self.draw(np.eye(4),(0.5,0.5,0.5,0.5),self.mesh_buffer.VAO,len(mesh.data['indices']))

    def draw(self, model_mat,color,VAO,indice_num):
        loc = gl.glGetUniformLocation(self.program, "normal_mat")
        gl.glUniformMatrix3fv(loc, 1, False, np.transpose(np.transpose(np.linalg.pinv(model_mat)))[0:3, 0:3])
        model_mat = np.transpose(model_mat)
        loc = gl.glGetUniformLocation(self.program, "model")
        gl.glUniformMatrix4fv(loc, 1, False, model_mat)
        loc = gl.glGetUniformLocation(self.program, "color")
        gl.glUniform4f(loc, *color)
        gl.glBindVertexArray(VAO)
        gl.glDrawElements(gl.GL_TRIANGLES, indice_num, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)

if __name__ == '__main__':
    skeleton_file = os.path.abspath( __file__+'/../assets/agents/koi_all_fins.json')
    app = QApplication([])
    widget = MinimalGLWidget(skeleton_file=skeleton_file)
    widget.show()
    app.exec_()