#!/usr/bin/env python3

"""

Code from http://www.labri.fr/perso/nrougier/python-opengl/#the-hard-way

"""

import ctypes
import logging

import numpy as np
import OpenGL.GL as gl
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QOpenGLWindow

from gym_fish.envs.visualization.camera import  camera

vertex_code = '''
uniform float scale;
    uniform mat4 matCam;
    attribute vec4 color;
    attribute vec3 position;
    varying vec4 v_color;
    void main()
    {
        gl_Position = matCam*vec4(scale*position, 1.0);
        v_color = color;
    } 
'''


fragment_code = '''
varying vec4 v_color;
    void main()
    {
        gl_FragColor = v_color;
    } 
'''

class MinimalGLWidget(QOpenGLWindow):
    def __init__(self):
        super(MinimalGLWidget, self).__init__()
        self.cam = camera()
        self.cam.center=(2,2,2)

    def initializeGL(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        self.program = self.initShaderProgram()
        # Build data
        # --------------------------------------
        self.data = np.zeros(8, [("position", np.float32, 3),
                            ("color", np.float32, 4)])

        self.data['color'] = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1),
                         (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]

        self.data['position'] = [(-1, -1, 1),
                            (1, -1, 1),
                            (1, 1, 1),
                            (-1, 1, 1),
                            (-1, -1, -1),
                            (1, -1, -1),
                            (1, 1, -1),
                            (-1, 1, -1)]

        self.index = np.array([0, 1, 2,
                          2, 3, 0,
                          1, 5, 6,
                          6, 2, 1,
                          7, 6, 5,
                          5, 4, 7,
                          4, 0, 3,
                          3, 7, 4,
                          4, 5, 1,
                          1, 0, 4,
                          3, 2, 6,
                          6, 7, 3],dtype=np.uint32)

        # Build buffer
        # --------------------------------------

        # Request a buffer slot from GPU
        buffer = gl.glGenBuffers(1)

        # Make this buffer the default one
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)

        # Upload data
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.data.nbytes, self.data, gl.GL_DYNAMIC_DRAW)

        # same for index buffer
        buffer_index = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, buffer_index)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.index.nbytes, self.index, gl.GL_STATIC_DRAW)

        # Bind attributes
        # --------------------------------------
        stride = self.data.strides[0]
        offset = ctypes.c_void_p(0)
        loc = gl.glGetAttribLocation(self.program, "position")
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

        offset = ctypes.c_void_p(self.data.dtype["position"].itemsize)
        loc = gl.glGetAttribLocation(self.program, "color")
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, buffer_index)

        # Bind uniforms
        # --------------------------------------
        loc = gl.glGetUniformLocation(self.program, "scale")
        gl.glUniform1f(loc, 1)
        clock = 0

        loc = gl.glGetUniformLocation(self.program, "matCam")
        # gl.glUniformMatrix4fv(loc, 1, False, np.eye(4))
        gl.glUniformMatrix4fv(loc, 1, False, self.cam.viewProejction.astype('f4'))

    def initShaderProgram(self):
        program = gl.glCreateProgram()
        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        # Set shaders source
        gl.glShaderSource(vertex, vertex_code)
        gl.glShaderSource(fragment, fragment_code)
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
        gl.glUseProgram(program)
        return program

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.index), gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))


if __name__ == '__main__':
    app = QApplication([])
    widget = MinimalGLWidget()
    widget.show()
    app.exec_()