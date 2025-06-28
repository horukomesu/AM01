from __future__ import annotations

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide6 import QtWidgets, QtCore, QtGui

from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt, gluNewQuadric, gluSphere
from OpenGL.GLUT import (
    glutInit,
    glutSolidSphere,
    glutBitmapCharacter,
    GLUT_BITMAP_HELVETICA_12,
)
import numpy as np

import AMUtilities


class GLSceneView(QtWidgets.QOpenGLWidget):
    """Simple OpenGL viewer for calibration results."""

    locator_clicked = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: list[np.ndarray] = []
        self.errors: dict[str, float] = {}
        self.cameras = []
        self.active_cam = 0
        self.image_width = 640
        self.image_height = 480
        self.texture_ids: dict[int, int] = {}
        self.locator_names: list[str] = []
        self.axis_points: dict[str, list[str]] = {}
        self.scale_pair: tuple[str, str] | None = None
        self.origin_locator_name: str | None = None
        self.calibrator = None
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        glutInit()

    def load_images(self, images: list[QtGui.QImage]):
        self.texture_ids.clear()
        for idx, img in enumerate(images):
            if img.isNull():
                continue
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            fmt = QtGui.QImage.Format_RGBA8888
            conv = img.convertToFormat(fmt)
            self.image_width = conv.width()
            self.image_height = conv.height()
            ptr = conv.bits()
            buf = memoryview(ptr)[:conv.byteCount()]
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                conv.width(),
                conv.height(),
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                buf,
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glBindTexture(GL_TEXTURE_2D, 0)
            self.texture_ids[idx] = tex_id

    def set_scene(
        self,
        calibrator,
        errors: dict[str, float] | None = None,
        locator_names: list[str] | None = None,
    ):
        self.calibrator = calibrator
        self.points = calibrator.get_points_3d()
        self.cameras = calibrator.get_camera_parameters()
        self.errors = errors or {}
        self.locator_names = list(locator_names or [])
        self.update()

    def set_active_camera(self, index: int):
        if 0 <= index < len(self.cameras):
            self.active_cam = index
            self.update()

    def set_overlay_data(
        self,
        axis_points: dict[str, list[str]] | None = None,
        scale_pair: tuple[str, str] | None = None,
        origin_name: str | None = None,
    ) -> None:
        self.axis_points = axis_points or {}
        self.scale_pair = scale_pair
        self.origin_locator_name = origin_name
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
        self._draw_image_plane()
        self._draw_points()
        self._draw_cameras()
        self._draw_axis_indicator(w, h)
        self._draw_overlay_elements()

    # --- Drawing helpers ----------------------------------------------------
    def _draw_points(self):
        glEnable(GL_DEPTH_TEST)
        quadric = gluNewQuadric()
        for idx, pt in enumerate(self.points):
            err = self.errors.get(str(idx), 0.0)
            color = AMUtilities.error_to_color(err)
            glColor3f(color.redF(), color.greenF(), color.blueF())
            glPushMatrix()
            glTranslatef(float(pt[0]), float(pt[1]), float(pt[2]))
            gluSphere(quadric, 0.02, 8, 8)
            glPopMatrix()
            if idx < len(self.locator_names):
                name = self.locator_names[idx]
                glColor3f(color.redF(), color.greenF(), color.blueF())
                glRasterPos3f(float(pt[0]), float(pt[1]), float(pt[2]))
                for ch in name:
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))

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

    def _draw_image_plane(self):
        tex_id = self.texture_ids.get(self.active_cam)
        if not tex_id:
            return
        cam = None
        if self.cameras:
            cam = self.cameras[self.active_cam]
        if not cam:
            return
        K = np.array(cam.intrinsics, dtype=float)
        f = K[0, 0]
        width = self.image_width
        height = self.image_height
        plane_w = width / f
        plane_h = height / f
        half_w = plane_w / 2.0
        half_h = plane_h / 2.0

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-half_w, -half_h, 1.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(half_w, -half_h, 1.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(half_w, half_h, 1.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-half_w, half_h, 1.0)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

    def _draw_overlay_elements(self):
        if not self.points:
            return
        name_to_index = {n: i for i, n in enumerate(self.locator_names)}

        def loc_3d(name: str):
            idx = name_to_index.get(name)
            if idx is not None and idx < len(self.points):
                return self.points[idx]
            return None

        colors = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
        for axis, pair in self.axis_points.items():
            if len(pair) != 2:
                continue
            p1 = loc_3d(pair[0])
            p2 = loc_3d(pair[1])
            if p1 is None or p2 is None:
                continue
            glColor3f(*colors.get(axis, (1.0, 1.0, 1.0)))
            glBegin(GL_LINES)
            glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
            glVertex3f(float(p2[0]), float(p2[1]), float(p2[2]))
            glEnd()
            mid = (np.array(p1) + np.array(p2)) / 2.0
            glRasterPos3f(float(mid[0]), float(mid[1]), float(mid[2]))
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(axis))

        if self.scale_pair and len(self.scale_pair) == 2:
            p1 = loc_3d(self.scale_pair[0])
            p2 = loc_3d(self.scale_pair[1])
            if p1 is not None and p2 is not None:
                glColor3f(1.0, 1.0, 0.0)
                glBegin(GL_LINES)
                glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
                glVertex3f(float(p2[0]), float(p2[1]), float(p2[2]))
                glEnd()

        if self.origin_locator_name:
            p = loc_3d(self.origin_locator_name)
            if p is not None:
                glColor3f(1.0, 1.0, 1.0)
                glRasterPos3f(float(p[0]), float(p[1]), float(p[2]))
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord("O"))

    # --- Interaction helpers -------------------------------------------------
    def _generate_ray(self, pos: QtCore.QPoint) -> tuple[np.ndarray, np.ndarray]:
        w, h = max(self.width(), 1), max(self.height(), 1)
        u = pos.x() / w * self.image_width
        v = pos.y() / h * self.image_height

        if self.cameras:
            cam = self.cameras[self.active_cam]
            K = np.array(cam.intrinsics, dtype=float)
            R = np.array(cam.rotation, dtype=float)
            t = np.array(cam.translation, dtype=float).reshape(3)
            x = (u - K[0, 2]) / K[0, 0]
            y = (v - K[1, 2]) / K[1, 1]
            ray_cam = np.array([x, y, 1.0])
            ray_world = R.T @ ray_cam
            ray_world /= np.linalg.norm(ray_world)
            origin = -R.T @ t
            return origin, ray_world

        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        return origin, direction

    @staticmethod
    def _ray_hits_sphere(
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> bool:
        v = center - origin
        t = np.dot(v, direction)
        if t < 0:
            return False
        closest = origin + t * direction
        dist = np.linalg.norm(center - closest)
        return dist <= radius

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return super().mousePressEvent(event)
        origin, direction = self._generate_ray(event.pos())
        for idx, pt in enumerate(self.points):
            if self._ray_hits_sphere(origin, direction, np.array(pt, dtype=float), radius=0.02):
                if idx < len(self.locator_names):
                    self.locator_clicked.emit(self.locator_names[idx])
                break
        else:
            super().mousePressEvent(event)

