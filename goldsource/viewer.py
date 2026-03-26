"""
3-D SMD Viewer panel for the GoldSource Model Merger.

Provides:
  - Model / SMD drop-downs for selecting what to display
  - QTreeWidget showing bone hierarchy with hover tooltips and rename support
  - QOpenGLWidget (OpenGL 1.x immediate-mode) with:
      - semi-transparent mesh triangles + wireframe overlay
      - bone skeleton lines and joints
      - hovered / selected bone highlighting
  - Status bar below viewport: bone name + world-space position on hover
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QCheckBox, QComboBox, QLabel, QPushButton, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QInputDialog,
)

try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL.GL import (
        glClear, glClearColor, glEnable, glDisable,
        glBegin, glEnd, glVertex3f, glColor4f, glPointSize, glLineWidth,
        glMatrixMode, glLoadIdentity, glViewport,
        glGetDoublev, glGetIntegerv,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST, GL_BLEND, GL_POINT_SMOOTH, GL_LINE_SMOOTH,
        GL_BLEND_SRC, GL_BLEND_DST,
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        GL_PROJECTION, GL_MODELVIEW,
        GL_TRIANGLES, GL_LINES, GL_POINTS,
        GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT,
        glBlendFunc,
        GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
        GL_LINEAR, GL_RGBA, GL_UNSIGNED_BYTE,
        glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
        glDeleteTextures, glTexCoord2f,
    )
    from OpenGL.GLU import gluPerspective, gluLookAt, gluProject
    _GL_OK = True
except Exception:
    _GL_OK = False

from PyQt6.QtWidgets import QMessageBox

from goldsource.smd import SMD


# ---------------------------------------------------------------------------
# Bone-editing helpers (operate on SMD in-place)
# ---------------------------------------------------------------------------

def _delete_bone(smd: SMD, bone_name: str) -> bool:
    """
    Remove the bone with *bone_name* from *smd* in-place.

    - Child bones are re-parented to the deleted bone's parent.
    - Vertices referencing the bone are reassigned to the parent bone
      (or to bone 0 when the deleted bone is a root).
    - All skeleton frame transforms for the bone are removed.

    Returns True if the bone was found and removed, False otherwise.
    """
    node = next((n for n in smd.nodes if n.name == bone_name), None)
    if node is None:
        return False

    bid            = node.id
    parent_bid     = node.parent_id  # -1 for root bones

    # Choose the bone id to reassign orphaned vertices to
    if parent_bid != -1:
        reassign_to = parent_bid
    else:
        # Deleted bone was a root — find another root or fall back to 0
        other_root = next(
            (n.id for n in smd.nodes if n.parent_id == -1 and n.id != bid),
            0,
        )
        reassign_to = other_root

    # Re-parent direct children
    for n in smd.nodes:
        if n.parent_id == bid:
            n.parent_id = parent_bid

    # Re-assign vertices
    for tri in smd.triangles:
        for v in (tri.v0, tri.v1, tri.v2):
            if v.bone_id == bid:
                v.bone_id = reassign_to

    # Remove skeleton transforms
    for frame in smd.skeleton:
        frame.bones = [b for b in frame.bones if b.bone_id != bid]

    # Remove node
    smd.nodes = [n for n in smd.nodes if n.id != bid]
    return True


# ---------------------------------------------------------------------------
# Bone world-transform maths
# ---------------------------------------------------------------------------

def _euler_mat4(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    4×4 matrix from ZYX Euler angles (radians) — GoldSource SMD convention.
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [ cy*cz,  cz*sx*sy - cx*sz,  cx*cz*sy + sx*sz,  0.0],
        [ cy*sz,  cx*cz + sx*sy*sz,  cx*sy*sz - cz*sx,  0.0],
        [-sy,     cy*sx,             cx*cy,             0.0],
        [ 0.0,    0.0,               0.0,               1.0],
    ], dtype=np.float64)


def compute_world_transforms(smd: SMD) -> dict[int, np.ndarray]:
    """
    Returns {bone_id: 4×4 world-space transform} using skeleton frame 0.
    The translation column ([:3, 3]) gives the world-space bone origin.
    """
    if not smd.skeleton:
        return {n.id: np.eye(4) for n in smd.nodes}

    frame = smd.skeleton[0]
    bt_map = {bt.bone_id: bt for bt in frame.bones}
    id_to_node = {n.id: n for n in smd.nodes}
    cache: dict[int, np.ndarray] = {}

    def _get(bid: int) -> np.ndarray:
        if bid in cache:
            return cache[bid]
        bt = bt_map.get(bid)
        if bt is None:
            cache[bid] = np.eye(4)
            return cache[bid]
        local = _euler_mat4(bt.rx, bt.ry, bt.rz)
        local[0, 3], local[1, 3], local[2, 3] = bt.tx, bt.ty, bt.tz
        node = id_to_node.get(bid)
        if node is None or node.parent_id == -1:
            cache[bid] = local
        else:
            cache[bid] = _get(node.parent_id) @ local
        return cache[bid]

    for node in smd.nodes:
        _get(node.id)
    return cache


# ---------------------------------------------------------------------------
# OpenGL viewport
# ---------------------------------------------------------------------------

if _GL_OK:
    class _SMDViewport(QOpenGLWidget):
        """Orbit 3-D viewport rendering an SMD mesh + skeleton."""

        boneHovered   = pyqtSignal(str, float, float, float)  # name, wx, wy, wz
        boneUnhovered = pyqtSignal()

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._smd: SMD | None = None

            # Geometry
            self._verts:       list[tuple[float, float, float]] = []
            self._tris:        list[tuple[int, int, int]]       = []
            # mat -> list of ((x0,y0,z0,u0,v0),(x1,...),(x2,...))
            self._mat_tris:    dict[str, list] = {}
            self._bone_pos:    dict[int, tuple[float, float, float]] = {}
            self._bone_lines:  list[tuple[int, int]]            = []

            # Textures
            self._tex_dir:     str  = ""
            self._textures:    dict[str, int] = {}   # material -> GL texture id
            self._show_textures: bool = False
            self._tex_dirty:   bool  = True

            # Orbit camera
            self._azimuth   = 45.0   # degrees
            self._elevation = 20.0   # degrees
            self._distance  = 50.0
            self._target    = np.zeros(3, dtype=float)

            self._last_mouse: QPoint | None     = None
            self._drag_btn:   Qt.MouseButton | None = None

            self._hovered_bone:  int | None = None
            self._selected_bone: int | None = None

            self.setMouseTracking(True)
            self.setMinimumSize(300, 300)

        # ── Public API ───────────────────────────────────────────────────

        def set_smd(self, smd: SMD | None) -> None:
            self._smd = smd
            self._build_geometry()
            self._auto_frame()
            self._tex_dirty = True
            self.update()

        def set_texture_dir(self, directory: str) -> None:
            if self._tex_dir != directory:
                self._tex_dir  = directory
                self._tex_dirty = True
                self.update()

        def set_textures_visible(self, visible: bool) -> None:
            self._show_textures = visible
            if visible:
                self._tex_dirty = True
            self.update()

        def set_selected_bone(self, bone_id: int | None) -> None:
            self._selected_bone = bone_id
            self.update()

        def highlight_bone(self, bone_id: int | None) -> None:
            if self._hovered_bone != bone_id:
                self._hovered_bone = bone_id
                self.update()

        # ── Geometry ─────────────────────────────────────────────────────

        def _build_geometry(self) -> None:
            self._verts, self._tris = [], []
            self._mat_tris = {}
            self._bone_pos, self._bone_lines = {}, []

            if self._smd is None:
                return

            # Mesh
            for tri in self._smd.triangles:
                base = len(self._verts)
                for v in (tri.v0, tri.v1, tri.v2):
                    self._verts.append((v.x, v.y, v.z))
                self._tris.append((base, base + 1, base + 2))
                # Also store UV-aware data grouped by material
                mat = tri.material
                if mat not in self._mat_tris:
                    self._mat_tris[mat] = []
                self._mat_tris[mat].append((
                    (tri.v0.x, tri.v0.y, tri.v0.z, tri.v0.u, tri.v0.v),
                    (tri.v1.x, tri.v1.y, tri.v1.z, tri.v1.u, tri.v1.v),
                    (tri.v2.x, tri.v2.y, tri.v2.z, tri.v2.u, tri.v2.v),
                ))

            # Skeleton
            world = compute_world_transforms(self._smd)
            for bid, mat in world.items():
                self._bone_pos[bid] = (float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]))
            for node in self._smd.nodes:
                if node.parent_id != -1 and node.parent_id in world:
                    self._bone_lines.append((node.parent_id, node.id))

        def _auto_frame(self) -> None:
            pts = list(self._verts) or list(self._bone_pos.values())
            if not pts:
                return
            arr = np.array(pts, dtype=float)
            self._target    = arr.mean(axis=0)
            extent          = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
            self._distance  = max(extent * 1.2, 1.0)

        # ── Texture management ───────────────────────────────────────────

        def _free_textures(self) -> None:
            for tid in self._textures.values():
                try:
                    glDeleteTextures(1, [tid])
                except Exception:
                    pass
            self._textures.clear()

        def _load_textures(self) -> None:
            self._free_textures()
            if not self._tex_dir or self._smd is None:
                return
            try:
                from PIL import Image
            except ImportError:
                return
            tex_dir = Path(self._tex_dir)
            if not tex_dir.is_dir():
                return
            # Build case-insensitive filename map for the directory
            dir_files: dict[str, Path] = {}
            for f in tex_dir.iterdir():
                dir_files[f.name.lower()] = f

            for mat in self._mat_tris:
                fname = Path(mat).name
                if not fname.lower().endswith(".bmp"):
                    fname = fname + ".bmp"
                fpath = dir_files.get(fname.lower())
                if fpath is None:
                    continue
                try:
                    img  = Image.open(fpath).convert("RGBA")
                    img  = img.transpose(Image.FLIP_TOP_BOTTOM)
                    data = img.tobytes()
                    w, h = img.size
                    tid  = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, tid)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
                    glBindTexture(GL_TEXTURE_2D, 0)
                    self._textures[mat] = tid
                except Exception:
                    pass

        # ── OpenGL callbacks ─────────────────────────────────────────────

        def initializeGL(self) -> None:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POINT_SMOOTH)
            glEnable(GL_LINE_SMOOTH)
            glClearColor(0.14, 0.14, 0.17, 1.0)

        def resizeGL(self, w: int, h: int) -> None:
            glViewport(0, 0, w, h)

        def paintGL(self) -> None:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            w, h = self.width(), self.height()
            if w == 0 or h == 0:
                return

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            near = max(self._distance * 0.001, 0.01)
            far  = self._distance * 20.0
            gluPerspective(45.0, w / h, near, far)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            eye = self._eye()
            # Use Z as up; fall back to Y when looking straight down
            gluLookAt(
                eye[0], eye[1], eye[2],
                self._target[0], self._target[1], self._target[2],
                0.0, 0.0, 1.0,
            )

            self._draw_grid()
            self._draw_mesh()
            self._draw_skeleton()

        # ── Camera helpers ───────────────────────────────────────────────

        def _eye(self) -> np.ndarray:
            az = math.radians(self._azimuth)
            el = math.radians(max(-88.0, min(88.0, self._elevation)))
            return self._target + self._distance * np.array([
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el),
            ])

        # ── Drawing helpers ──────────────────────────────────────────────

        def _draw_grid(self) -> None:
            step = max(self._distance / 12.0, 0.01)
            n    = 10
            glLineWidth(1.0)
            glColor4f(0.28, 0.28, 0.32, 0.6)
            glBegin(GL_LINES)
            for i in range(-n, n + 1):
                glVertex3f(i * step, -n * step, 0.0)
                glVertex3f(i * step,  n * step, 0.0)
                glVertex3f(-n * step, i * step, 0.0)
                glVertex3f( n * step, i * step, 0.0)
            glEnd()

        def _draw_mesh(self) -> None:
            if not self._tris:
                return

            if self._show_textures and self._mat_tris:
                if self._tex_dirty:
                    self._load_textures()
                    self._tex_dirty = False
                glEnable(GL_TEXTURE_2D)
                for mat, tris_data in self._mat_tris.items():
                    tid = self._textures.get(mat)
                    if tid:
                        glBindTexture(GL_TEXTURE_2D, tid)
                        glColor4f(1.0, 1.0, 1.0, 1.0)
                    else:
                        glBindTexture(GL_TEXTURE_2D, 0)
                        glColor4f(0.55, 0.65, 0.80, 0.80)
                    glBegin(GL_TRIANGLES)
                    for (x0, y0, z0, u0, v0), (x1, y1, z1, u1, v1), (x2, y2, z2, u2, v2) in tris_data:
                        glTexCoord2f(u0, v0); glVertex3f(x0, y0, z0)
                        glTexCoord2f(u1, v1); glVertex3f(x1, y1, z1)
                        glTexCoord2f(u2, v2); glVertex3f(x2, y2, z2)
                    glEnd()
                glBindTexture(GL_TEXTURE_2D, 0)
                glDisable(GL_TEXTURE_2D)
            else:
                # Solid fill (semi-transparent)
                glColor4f(0.55, 0.65, 0.80, 0.20)
                glBegin(GL_TRIANGLES)
                for a, b, c in self._tris:
                    for idx in (a, b, c):
                        x, y, z = self._verts[idx]
                        glVertex3f(x, y, z)
                glEnd()
                # Wireframe overlay
                glColor4f(0.45, 0.55, 0.70, 0.35)
                glLineWidth(0.6)
                glBegin(GL_LINES)
                for a, b, c in self._tris:
                    for u, v in ((a, b), (b, c), (c, a)):
                        x1, y1, z1 = self._verts[u]
                        x2, y2, z2 = self._verts[v]
                        glVertex3f(x1, y1, z1)
                        glVertex3f(x2, y2, z2)
                glEnd()

        def _draw_skeleton(self) -> None:
            if not self._bone_pos:
                return

            # Bone lines
            glLineWidth(2.0)
            glColor4f(1.0, 0.75, 0.15, 1.0)
            glBegin(GL_LINES)
            for pid, cid in self._bone_lines:
                if pid in self._bone_pos and cid in self._bone_pos:
                    px, py, pz = self._bone_pos[pid]
                    cx, cy, cz = self._bone_pos[cid]
                    glVertex3f(px, py, pz)
                    glVertex3f(cx, cy, cz)
            glEnd()

            # Bone dots
            glPointSize(9.0)
            glBegin(GL_POINTS)
            for bid, (x, y, z) in self._bone_pos.items():
                if bid == self._selected_bone:
                    glColor4f(1.0, 0.30, 0.25, 1.0)  # red
                elif bid == self._hovered_bone:
                    glColor4f(0.25, 1.0, 0.40, 1.0)  # green
                else:
                    glColor4f(0.10, 0.75, 1.00, 1.0)  # cyan
                glVertex3f(x, y, z)
            glEnd()

        # ── Mouse interaction ────────────────────────────────────────────

        def mousePressEvent(self, event) -> None:
            self._last_mouse = event.pos()
            self._drag_btn   = event.button()

        def mouseReleaseEvent(self, event) -> None:
            self._last_mouse = None
            self._drag_btn   = None

        def mouseMoveEvent(self, event) -> None:
            pos = event.pos()

            if self._last_mouse is not None and self._drag_btn is not None:
                dx = pos.x() - self._last_mouse.x()
                dy = pos.y() - self._last_mouse.y()

                if self._drag_btn == Qt.MouseButton.LeftButton:
                    self._azimuth   -= dx * 0.5
                    self._elevation  = max(-88.0, min(88.0, self._elevation + dy * 0.5))
                    self.update()

                elif self._drag_btn in (
                    Qt.MouseButton.RightButton,
                    Qt.MouseButton.MiddleButton,
                ):
                    az    = math.radians(self._azimuth)
                    right = np.array([-math.sin(az), math.cos(az), 0.0])
                    fwd   = self._eye() - self._target
                    up    = np.cross(right, fwd)
                    n     = np.linalg.norm(up)
                    if n > 1e-9:
                        up /= n
                    else:
                        up = np.array([0.0, 0.0, 1.0])
                    scale          = self._distance * 0.0018
                    self._target  -= right * dx * scale
                    self._target  += up    * dy * scale
                    self.update()

            self._last_mouse = pos
            self._check_hover(pos)

        def wheelEvent(self, event) -> None:
            factor        = 0.88 if event.angleDelta().y() > 0 else 1.14
            self._distance = max(0.2, self._distance * factor)
            self.update()

        # ── Hover detection ──────────────────────────────────────────────

        def _check_hover(self, mouse: QPoint) -> None:
            if not self._bone_pos:
                return
            self.makeCurrent()
            try:
                from OpenGL.GL import GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT
                from OpenGL.raw.GL.VERSION.GL_1_0 import glGetDoublev as _getd
                from OpenGL.raw.GL.VERSION.GL_1_0 import glGetIntegerv as _geti
                from OpenGL.arrays import vbo as _vbo
                import ctypes

                mv   = (ctypes.c_double * 16)()
                proj = (ctypes.c_double * 16)()
                vp   = (ctypes.c_int    *  4)()
                glGetDoublev(GL_MODELVIEW_MATRIX,  mv)
                glGetDoublev(GL_PROJECTION_MATRIX, proj)
                glGetIntegerv(GL_VIEWPORT, vp)
            except Exception:
                return

            THRESHOLD = 14.0
            best_bid  = None
            best_dist = THRESHOLD

            for bid, (wx, wy, wz) in self._bone_pos.items():
                try:
                    sx, sy, _sz = gluProject(wx, wy, wz, mv, proj, vp)
                except Exception:
                    continue
                sy_qt = self.height() - sy  # flip Y
                d = math.hypot(sx - mouse.x(), sy_qt - mouse.y())
                if d < best_dist:
                    best_dist = d
                    best_bid  = bid

            if best_bid != self._hovered_bone:
                self._hovered_bone = best_bid
                self.update()
                if best_bid is not None and self._smd:
                    node = next((n for n in self._smd.nodes if n.id == best_bid), None)
                    if node:
                        x, y, z = self._bone_pos[best_bid]
                        self.boneHovered.emit(node.name, x, y, z)
                else:
                    self.boneUnhovered.emit()

else:
    # Fallback when OpenGL is not available
    class _SMDViewport(QLabel):  # type: ignore[no-redef]
        boneHovered   = pyqtSignal(str, float, float, float)
        boneUnhovered = pyqtSignal()

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(
                "OpenGL not available.\nInstall PyOpenGL:  pip install PyOpenGL",
                parent,
            )
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        def set_smd(self, smd: SMD | None) -> None:
            pass

        def set_selected_bone(self, bone_id: int | None) -> None:
            pass

        def highlight_bone(self, bone_id: int | None) -> None:
            pass


# ---------------------------------------------------------------------------
# Viewer panel (left controls + right viewport)
# ---------------------------------------------------------------------------

class ViewerPanel(QWidget):
    """
    Full viewer tab: SMD selector on the left, 3-D viewport on the right.
    Emits *bonesRenamed* after any in-place bone rename so the main window
    can trigger re-analysis.
    """

    bonesRenamed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._models:  list  = []          # list[ModelInput]
        self._dirs:    dict[str, str] = {}
        self._cur_smd: SMD | None = None
        self._cur_model_name: str = ""
        self._setup_ui()

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel ───────────────────────────────────────────────────
        left = QWidget()
        ll   = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        ll.setSpacing(4)

        ll.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        ll.addWidget(self._model_combo)

        ll.addWidget(QLabel("SMD:"))
        self._smd_combo = QComboBox()
        self._smd_combo.currentIndexChanged.connect(self._on_smd_changed)
        ll.addWidget(self._smd_combo)

        ll.addWidget(QLabel("Bones:"))
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "ID", "Parent"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(lambda item, _: self._rename(item))
        ll.addWidget(self._tree, 1)

        bone_btns = QVBoxLayout()
        bone_btns.setSpacing(3)

        row1 = QHBoxLayout()
        self._btn_rename = QPushButton("Rename")
        self._btn_rename.setEnabled(False)
        self._btn_rename.clicked.connect(self._on_rename_clicked)
        row1.addWidget(self._btn_rename)
        row1.addStretch()
        bone_btns.addLayout(row1)

        row2 = QHBoxLayout()
        self._btn_delete = QPushButton("Delete Bone")
        self._btn_delete.setEnabled(False)
        self._btn_delete.setStyleSheet("color: #e05555;")
        self._btn_delete.clicked.connect(self._on_delete_clicked)
        row2.addWidget(self._btn_delete)
        row2.addStretch()
        bone_btns.addLayout(row2)

        row3 = QHBoxLayout()
        self._btn_delete_all = QPushButton("Delete from All SMDs")
        self._btn_delete_all.setEnabled(False)
        self._btn_delete_all.setStyleSheet("color: #e05555;")
        self._btn_delete_all.clicked.connect(self._on_delete_all_clicked)
        row3.addWidget(self._btn_delete_all)
        row3.addStretch()
        bone_btns.addLayout(row3)

        row4 = QHBoxLayout()
        self._btn_export = QPushButton("Export SMD")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export_clicked)
        row4.addWidget(self._btn_export)
        row4.addStretch()
        bone_btns.addLayout(row4)

        ll.addLayout(bone_btns)

        left.setMinimumWidth(220)
        left.setMaximumWidth(380)
        splitter.addWidget(left)

        # ── Right panel ──────────────────────────────────────────────────
        right  = QWidget()
        rl     = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # Toolbar above viewport
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        self._chk_textures = QCheckBox("Show Textures")
        self._chk_textures.setChecked(False)
        self._chk_textures.toggled.connect(self._on_textures_toggled)
        toolbar.addWidget(self._chk_textures)
        toolbar.addStretch()
        rl.addLayout(toolbar)

        self._viewport = _SMDViewport()
        self._viewport.boneHovered.connect(self._on_bone_hovered)
        self._viewport.boneUnhovered.connect(self._on_bone_unhovered)
        rl.addWidget(self._viewport, 1)

        self._info_label = QLabel("  Left-drag: orbit   Right-drag: pan   Scroll: zoom   Hover bone to inspect")
        self._info_label.setStyleSheet("color: #888; padding: 3px 6px; font-size: 11px;")
        rl.addWidget(self._info_label)

        splitter.addWidget(right)
        splitter.setSizes([260, 700])
        root.addWidget(splitter)

    # ── Public API ────────────────────────────────────────────────────────

    def update_models(self, models: list, dirs: dict[str, str] | None = None) -> None:
        self._models = models
        if dirs:
            self._dirs.update(dirs)

        prev = self._model_combo.currentText()
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in models:
            self._model_combo.addItem(m.name)
        idx = self._model_combo.findText(prev)
        self._model_combo.setCurrentIndex(max(0, idx))
        self._model_combo.blockSignals(False)
        self._on_model_changed()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_model_changed(self) -> None:
        name = self._model_combo.currentText()
        self._cur_model_name = name
        model = next((m for m in self._models if m.name == name), None)

        prev = self._smd_combo.currentText()
        self._smd_combo.blockSignals(True)
        self._smd_combo.clear()
        if model:
            for key in sorted(model.smds.keys()):
                self._smd_combo.addItem(key)
        idx = self._smd_combo.findText(prev)
        self._smd_combo.setCurrentIndex(max(0, idx))
        self._smd_combo.blockSignals(False)
        self._on_smd_changed()

    def _on_smd_changed(self) -> None:
        key   = self._smd_combo.currentText()
        model = next((m for m in self._models if m.name == self._cur_model_name), None)
        self._cur_smd = model.smds.get(key) if model else None
        self._rebuild_tree()
        self._viewport.set_smd(self._cur_smd)
        tex_dir = self._dirs.get(self._cur_model_name, "")
        self._viewport.set_texture_dir(tex_dir)
        self._btn_export.setEnabled(self._cur_smd is not None)

    def _on_textures_toggled(self, checked: bool) -> None:
        self._viewport.set_textures_visible(checked)

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        has_sel = bool(items)
        has_smd = self._cur_smd is not None
        self._btn_rename.setEnabled(has_sel)
        self._btn_delete.setEnabled(has_sel)
        self._btn_delete_all.setEnabled(has_sel)
        self._btn_export.setEnabled(has_smd)
        if items:
            bid = items[0].data(0, Qt.ItemDataRole.UserRole)
            self._viewport.set_selected_bone(bid)
            self._viewport.highlight_bone(bid)

    def _on_rename_clicked(self) -> None:
        items = self._tree.selectedItems()
        if items:
            self._rename(items[0])

    def _on_delete_clicked(self) -> None:
        items = self._tree.selectedItems()
        if not items or self._cur_smd is None:
            return
        bone_name = items[0].text(0)
        ans = QMessageBox.question(
            self, "Delete Bone",
            f"Delete bone '{bone_name}' from the current SMD?\n\n"
            "Its vertices will be reassigned to the parent bone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        if _delete_bone(self._cur_smd, bone_name):
            self._rebuild_tree()
            self._viewport.set_smd(self._cur_smd)
            self.bonesRenamed.emit()

    def _on_delete_all_clicked(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        bone_name = items[0].text(0)
        model = next((m for m in self._models if m.name == self._cur_model_name), None)
        if model is None:
            return
        ans = QMessageBox.question(
            self, "Delete from All SMDs",
            f"Delete bone '{bone_name}' from ALL {len(model.smds)} SMD(s) "
            f"in model '{self._cur_model_name}'?\n\n"
            "Its vertices will be reassigned to the parent bone in each file.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        count = sum(1 for smd in model.smds.values() if _delete_bone(smd, bone_name))
        self._rebuild_tree()
        self._viewport.set_smd(self._cur_smd)
        self.bonesRenamed.emit()
        QMessageBox.information(
            self, "Done",
            f"Bone '{bone_name}' deleted from {count} SMD(s).",
        )

    def _on_export_clicked(self) -> None:
        if self._cur_smd is None:
            return
        smd_key = self._smd_combo.currentText()
        model_dir = self._dirs.get(self._cur_model_name, "")
        if not model_dir:
            QMessageBox.warning(self, "Export SMD", "Model directory is unknown.")
            return
        out_path = Path(model_dir) / (smd_key + ".smd")
        try:
            self._cur_smd.save(out_path)
            QMessageBox.information(self, "Export SMD", f"Saved:\n{out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export SMD", f"Failed to save:\n{exc}")

    def _on_bone_hovered(self, name: str, x: float, y: float, z: float) -> None:
        self._info_label.setText(
            f"  Bone: {name}   pos ({x:.3f},  {y:.3f},  {z:.3f})"
        )
        # Mirror selection in tree
        matches = self._tree.findItems(
            name,
            Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive,
            0,
        )
        if matches:
            self._tree.scrollToItem(matches[0])

    def _on_bone_unhovered(self) -> None:
        self._info_label.setText(
            "  Left-drag: orbit   Right-drag: pan   Scroll: zoom   Hover bone to inspect"
        )

    # ── Bone tree ─────────────────────────────────────────────────────────

    def _rebuild_tree(self) -> None:
        self._tree.clear()
        if self._cur_smd is None:
            return

        smd          = self._cur_smd
        id_to_node   = {n.id: n for n in smd.nodes}
        world        = compute_world_transforms(smd)
        items:  dict[int, QTreeWidgetItem] = {}

        for node in sorted(smd.nodes, key=lambda n: n.id):
            parent_name = (
                id_to_node[node.parent_id].name
                if node.parent_id in id_to_node else "-"
            )
            item = QTreeWidgetItem([node.name, str(node.id), parent_name])
            item.setData(0, Qt.ItemDataRole.UserRole, node.id)
            mat = world.get(node.id)
            if mat is not None:
                px, py, pz = float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])
                item.setToolTip(0, f"{node.name}\nID: {node.id}\nWorld pos: ({px:.3f}, {py:.3f}, {pz:.3f})")
            items[node.id] = item

        # Wire parent-child relationships
        for node in sorted(smd.nodes, key=lambda n: n.id):
            item = items[node.id]
            if node.parent_id != -1 and node.parent_id in items:
                items[node.parent_id].addChild(item)
            else:
                self._tree.addTopLevelItem(item)

        self._tree.expandAll()

    # ── Bone rename ───────────────────────────────────────────────────────

    def _rename(self, item: QTreeWidgetItem) -> None:
        old_name = item.text(0)
        new_name, ok = QInputDialog.getText(
            self, "Rename Bone",
            f"New name for '{old_name}':",
            text=old_name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return

        # Apply to every SMD in this model
        model = next((m for m in self._models if m.name == self._cur_model_name), None)
        if model is None:
            return
        for smd in model.smds.values():
            for node in smd.nodes:
                if node.name == old_name:
                    node.name = new_name

        # Refresh tree and emit change signal
        self._rebuild_tree()
        self.bonesRenamed.emit()
