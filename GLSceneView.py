from __future__ import annotations

from PySide6 import QtWidgets, QtCore
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
from OpenGL.GLUT import glutInit, glutSolidSphere
import numpy as np

import AMUtilities


class GLSceneView(QtWidgets.QOpenGLWidget):
    """Simple OpenGL viewer for calibration results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: list[np.ndarray] = []
        self.errors: dict[str, float] = {}
        self.cameras = []
        self.active_cam = 0
        self.image_width = 640
        glutInit()

    def set_scene(self, calibrator, errors: dict[str, float] | None = None):
        self.points = calibrator.get_points_3d()
        self.cameras = calibrator.get_camera_parameters()
        self.errors = errors or {}
        self.update()

    def set_active_camera(self, index: int):
        if 0 <= index < len(self.cameras):
            self.active_cam = index
            self.update()

    # --- Qt OpenGL callbacks -------------------------------------------------
    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, max(h, 1))

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        w, h = self.width(), max(self.height(), 1)

        # --- Setup camera ----------------------------------------------------
        aspect = w / h
        fov = 60.0
        eye = np.array([0.0, 0.0, 5.0])
        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        if self.cameras:
            cam = self.cameras[self.active_cam]
            K = np.array(cam.intrinsics, dtype=float)
            fov = np.degrees(2 * np.arctan(self.image_width / (2 * K[0, 0])))
            R = np.array(cam.rotation, dtype=float)
            t = np.array(cam.translation, dtype=float).reshape(3)
            eye = -R.T @ t
            center = eye + R.T @ np.array([0, 0, 1])
            up = R.T @ np.array([0, 1, 0])

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(float(fov), float(aspect), 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            float(eye[0]), float(eye[1]), float(eye[2]),
            float(center[0]), float(center[1]), float(center[2]),
            float(up[0]), float(up[1]), float(up[2])
        )

        self._draw_points()
        self._draw_cameras()
        self._draw_axis_indicator(w, h)

    # --- Drawing helpers ----------------------------------------------------
    def _draw_points(self):
        glEnable(GL_DEPTH_TEST)
        for idx, pt in enumerate(self.points):
            err = self.errors.get(str(idx), 0.0)
            color = AMUtilities.error_to_color(err)
            glColor3f(color.redF(), color.greenF(), color.blueF())
            glPushMatrix()
            glTranslatef(float(pt[0]), float(pt[1]), float(pt[2]))
            glutSolidSphere(0.02, 8, 8)
            glPopMatrix()

    def _draw_cameras(self):
        glColor4f(1.0, 1.0, 1.0, 0.4)
        size = 0.05
        for cam in self.cameras:
            R = np.array(cam.rotation, dtype=float)
            t = np.array(cam.translation, dtype=float).reshape(3)
            c = -R.T @ t
            corners = [
                np.array([ size,  size,  size]),
                np.array([-size,  size,  size]),
                np.array([-size, -size,  size]),
                np.array([ size, -size,  size])
            ]
            pts = [c + R.T @ p for p in corners]
            glBegin(GL_LINE_LOOP)
            for p in pts:
                glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()
            for p in pts:
                glBegin(GL_LINES)
                glVertex3f(float(c[0]), float(c[1]), float(c[2]))
                glVertex3f(float(p[0]), float(p[1]), float(p[2]))
                glEnd()

    def _draw_axis_indicator(self, w: int, h: int):
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        origin = QtCore.QPointF(w - 50, 50)
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex2f(origin.x(), origin.y())
        glVertex2f(origin.x() + 30, origin.y())
        glColor3f(0.0, 1.0, 0.0)
        glVertex2f(origin.x(), origin.y())
        glVertex2f(origin.x(), origin.y() + 30)
        glColor3f(0.0, 0.0, 1.0)
        glVertex2f(origin.x(), origin.y())
        glVertex2f(origin.x() - 21, origin.y() + 21)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

