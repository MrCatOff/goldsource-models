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
import re
import shutil
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QCheckBox, QComboBox, QLabel, QPushButton, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QInputDialog, QFileDialog, QGroupBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QSlider,
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

from goldsource.smd import SMD, Node, BoneTransform, SkeletonFrame
from goldsource.qc import QC, BodyGroup, BodyGroupEntry


# ---------------------------------------------------------------------------
# Bone-editing helpers (operate on SMD in-place)
# ---------------------------------------------------------------------------

def _collect_descendants(smd: SMD, bone_name: str) -> list[str]:
    """
    Return all descendant bone names of *bone_name* in leaf-first order
    (so that deleting them in sequence is safe).
    """
    name_to_id  = {n.name: n.id for n in smd.nodes}
    id_to_name  = {n.id: n.name for n in smd.nodes}
    children_of: dict[int, list[int]] = {}
    for n in smd.nodes:
        children_of.setdefault(n.parent_id, []).append(n.id)

    root_id = name_to_id.get(bone_name)
    if root_id is None:
        return []

    # BFS to collect all descendants (excluding the root itself)
    result_ids: list[int] = []
    queue = list(children_of.get(root_id, []))
    while queue:
        bid = queue.pop(0)
        result_ids.append(bid)
        queue.extend(children_of.get(bid, []))

    # Reverse so leaves come first
    result_ids.reverse()
    return [id_to_name[i] for i in result_ids if i in id_to_name]


def _delete_bone(smd: SMD, bone_name: str) -> bool:
    """
    Remove the bone with *bone_name* from *smd* in-place.

    - Triangles that have any vertex referencing the bone are removed.
    - Child bones are re-parented to the deleted bone's parent.
    - All skeleton frame transforms for the bone are removed.

    Returns True if the bone was found and removed, False otherwise.
    """
    node = next((n for n in smd.nodes if n.name == bone_name), None)
    if node is None:
        return False

    bid        = node.id
    parent_bid = node.parent_id  # -1 for root bones

    # Remove any triangle that has at least one vertex on this bone
    smd.triangles = [
        tri for tri in smd.triangles
        if not any(v.bone_id == bid for v in (tri.v0, tri.v1, tri.v2))
    ]

    # Re-parent direct children
    for n in smd.nodes:
        if n.parent_id == bid:
            n.parent_id = parent_bid

    # Remove skeleton transforms
    for frame in smd.skeleton:
        frame.bones = [b for b in frame.bones if b.bone_id != bid]

    # Remove node
    smd.nodes = [n for n in smd.nodes if n.id != bid]
    return True


# ---------------------------------------------------------------------------
# Hand-replacement algorithm
# ---------------------------------------------------------------------------

def _apply_hand_replacement(
    smd: SMD,
    bone_map: dict[str, str],   # weapon_bone_name → hand_bone_name
    hands_smd: SMD,
) -> SMD:
    """
    Return a *new* SMD where the bones named as keys in *bone_map* have been
    replaced by the corresponding bones from *hands_smd*, while weapon-specific
    bones (not in *bone_map*) are renumbered to IDs above the max hand ID.

    For reference SMDs the master-hand transforms are used for hand bones.
    For animation SMDs the original per-frame transforms are remapped.
    """
    import copy

    hand_by_name = {n.name: n for n in hands_smd.nodes}
    reverse_map  = {v: k for k, v in bone_map.items()}   # hand_name → weapon_name
    max_hand_id  = max((n.id for n in hands_smd.nodes), default=-1)

    # ── Build old-weapon-ID → new-ID mapping ─────────────────────────────
    old_to_new: dict[int, int] = {}
    for wp_name, hp_name in bone_map.items():
        wp_node = next((n for n in smd.nodes if n.name == wp_name), None)
        hp_node = hand_by_name.get(hp_name)
        if wp_node and hp_node:
            old_to_new[wp_node.id] = hp_node.id

    # Topological sort of weapon-specific bones (parents before children)
    weapon_specific = [n for n in smd.nodes if n.name not in bone_map]
    wp_by_old = {n.id: n for n in weapon_specific}
    weapon_ordered: list[Node] = []
    visited: set[int] = set()

    def _visit(n: Node) -> None:
        if n.id in visited:
            return
        visited.add(n.id)
        if n.parent_id in wp_by_old:
            _visit(wp_by_old[n.parent_id])
        weapon_ordered.append(n)

    for n in weapon_specific:
        _visit(n)

    next_id = max_hand_id + 1
    for n in weapon_ordered:
        old_to_new[n.id] = next_id
        next_id += 1

    # ── New node list ─────────────────────────────────────────────────────
    new_nodes: list[Node] = [
        Node(id=hn.id, name=hn.name, parent_id=hn.parent_id)
        for hn in hands_smd.nodes
    ]
    for n in weapon_ordered:
        new_parent = old_to_new.get(n.parent_id, -1) if n.parent_id != -1 else -1
        new_nodes.append(Node(id=old_to_new[n.id], name=n.name, parent_id=new_parent))

    # ── Skeleton frames ───────────────────────────────────────────────────
    # Pre-build a name→old-weapon-node lookup for hand bones
    wp_node_by_hand_name: dict[str, "Node"] = {}
    for hn in hands_smd.nodes:
        wp_name = reverse_map.get(hn.name, "")
        wp_node = next((n for n in smd.nodes if n.name == wp_name), None)
        if wp_node:
            wp_node_by_hand_name[hn.name] = wp_node

    new_frames: list[SkeletonFrame] = []
    for frame in (smd.skeleton or []):
        old_bt = {bt.bone_id: bt for bt in frame.bones}
        new_bones: list[BoneTransform] = []

        # Hand bones — always use the weapon's own skeleton data (just re-ID'd).
        # The master hands file only contributes node IDs/names/hierarchy,
        # never transforms, so animation is preserved correctly.
        for hn in hands_smd.nodes:
            wp_node = wp_node_by_hand_name.get(hn.name)
            src_bt  = old_bt.get(wp_node.id) if wp_node else None
            if src_bt:
                new_bones.append(BoneTransform(
                    bone_id=hn.id,
                    tx=src_bt.tx, ty=src_bt.ty, tz=src_bt.tz,
                    rx=src_bt.rx, ry=src_bt.ry, rz=src_bt.rz,
                ))
            else:
                new_bones.append(BoneTransform(
                    bone_id=hn.id, tx=0, ty=0, tz=0, rx=0, ry=0, rz=0,
                ))

        # Weapon-specific bones
        for n in weapon_ordered:
            src_bt = old_bt.get(n.id)
            if src_bt:
                new_bones.append(BoneTransform(
                    bone_id=old_to_new[n.id],
                    tx=src_bt.tx, ty=src_bt.ty, tz=src_bt.tz,
                    rx=src_bt.rx, ry=src_bt.ry, rz=src_bt.rz,
                ))

        new_frames.append(SkeletonFrame(time=frame.time, bones=new_bones))

    # ── Triangles (remap vertex bone_ids) ────────────────────────────────
    new_triangles = []
    for tri in smd.triangles:
        new_v = []
        for v in tri.vertices:
            nv = copy.copy(v)
            nv.bone_id = old_to_new.get(v.bone_id, v.bone_id)
            new_v.append(nv)
        nt = copy.copy(tri)
        nt.v0, nt.v1, nt.v2 = new_v[0], new_v[1], new_v[2]
        new_triangles.append(nt)

    result = copy.copy(smd)
    result.nodes     = new_nodes
    result.skeleton  = new_frames
    result.triangles = new_triangles
    return result


def _build_hands_body_smd(
    hands_smd: SMD,
    weapon_ref_smd: SMD,
    bone_map: dict[str, str],   # weapon_name → hand_name
) -> SMD:
    """
    Build a combined reference SMD suitable for a '$bodygroup hands' entry.

    The result contains:
    - The hand mesh triangles from *hands_smd*
    - Hand bone nodes + bind-pose skeleton from *hands_smd*
    - Weapon-specific bones (those not in *bone_map*) appended with the same
      new IDs that _apply_hand_replacement would assign, and bind-pose
      transforms taken from *weapon_ref_smd* frame 0.

    This ensures the compiler sees a consistent bone hierarchy across all
    SMDs in the merged model.
    """
    import copy

    hand_by_name = {n.name: n for n in hands_smd.nodes}
    max_hand_id  = max((n.id for n in hands_smd.nodes), default=-1)

    # Map weapon-bone old-ID → new-ID for the matched (hand) bones
    old_to_new: dict[int, int] = {}
    for wp_name, hp_name in bone_map.items():
        wp_node = next((n for n in weapon_ref_smd.nodes if n.name == wp_name), None)
        hp_node = hand_by_name.get(hp_name)
        if wp_node and hp_node:
            old_to_new[wp_node.id] = hp_node.id

    # Topological sort of weapon-specific bones
    weapon_specific = [n for n in weapon_ref_smd.nodes if n.name not in bone_map]
    wp_by_old       = {n.id: n for n in weapon_specific}
    weapon_ordered: list[Node] = []
    visited: set[int] = set()

    def _visit(n: Node) -> None:
        if n.id in visited:
            return
        visited.add(n.id)
        if n.parent_id in wp_by_old:
            _visit(wp_by_old[n.parent_id])
        weapon_ordered.append(n)

    for n in weapon_specific:
        _visit(n)

    next_id = max_hand_id + 1
    for n in weapon_ordered:
        old_to_new[n.id] = next_id
        next_id += 1

    # ── Nodes: hand bones first, weapon-specific after ────────────────
    new_nodes: list[Node] = [
        Node(id=hn.id, name=hn.name, parent_id=hn.parent_id)
        for hn in hands_smd.nodes
    ]
    for n in weapon_ordered:
        new_parent = old_to_new.get(n.parent_id, -1) if n.parent_id != -1 else -1
        new_nodes.append(Node(id=old_to_new[n.id], name=n.name, parent_id=new_parent))

    # ── Skeleton frame 0 ─────────────────────────────────────────────
    hand_bt  = {bt.bone_id: bt for bt in hands_smd.skeleton[0].bones} \
               if hands_smd.skeleton else {}
    wp_bt    = {bt.bone_id: bt for bt in weapon_ref_smd.skeleton[0].bones} \
               if weapon_ref_smd.skeleton else {}

    new_bones: list[BoneTransform] = []
    for hn in hands_smd.nodes:
        bt = hand_bt.get(hn.id)
        if bt:
            new_bones.append(BoneTransform(
                bone_id=hn.id,
                tx=bt.tx, ty=bt.ty, tz=bt.tz,
                rx=bt.rx, ry=bt.ry, rz=bt.rz,
            ))
        else:
            new_bones.append(BoneTransform(
                bone_id=hn.id, tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0,
            ))
    for n in weapon_ordered:
        bt = wp_bt.get(n.id)
        nid = old_to_new[n.id]
        if bt:
            new_bones.append(BoneTransform(
                bone_id=nid,
                tx=bt.tx, ty=bt.ty, tz=bt.tz,
                rx=bt.rx, ry=bt.ry, rz=bt.rz,
            ))
        else:
            new_bones.append(BoneTransform(
                bone_id=nid, tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0,
            ))

    result           = copy.copy(hands_smd)
    result.nodes     = new_nodes
    result.skeleton  = [SkeletonFrame(time=0, bones=new_bones)]
    result.triangles = list(hands_smd.triangles)
    return result


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


def compute_world_transforms(smd: SMD, frame_idx: int = 0) -> dict[int, np.ndarray]:
    """
    Returns {bone_id: 4×4 world-space transform} for the given skeleton frame.
    The translation column ([:3, 3]) gives the world-space bone origin.
    """
    if not smd.skeleton:
        return {n.id: np.eye(4) for n in smd.nodes}

    frame = smd.skeleton[min(frame_idx, len(smd.skeleton) - 1)]
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
            self._smd:          SMD | None = None   # reference / texture SMD
            self._anim_smd:     SMD | None = None   # animation SMD
            self._frame_idx:    int        = 0
            self._skinned_verts:    list | None = None  # per-frame skinned positions (flat)
            self._skinned_mat_tris: dict | None = None  # per-frame skinned mat_tris for textured draw

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
            self._show_mesh:     bool = True
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
            """Set the reference (texture/mesh) SMD."""
            self._smd = smd
            self._anim_smd = None
            self._skinned_verts = None
            self._skinned_mat_tris = None
            self._frame_idx = 0
            self._build_mesh()
            self._build_skeleton(0)
            self._auto_frame()
            self._tex_dirty = True
            self.update()

        def set_anim_smd(self, smd: SMD | None) -> None:
            """Set the animation SMD to play over the reference mesh."""
            self._anim_smd = smd
            self._frame_idx = 0
            self._build_skeleton(0)
            self._compute_skinned_verts(0)
            self.update()

        def set_frame(self, idx: int) -> None:
            """Seek to *idx* and redraw skeleton + skinned mesh."""
            self._frame_idx = idx
            self._build_skeleton(idx)
            self._compute_skinned_verts(idx)
            self.update()

        def set_texture_dir(self, directory: str) -> None:
            if self._tex_dir != directory:
                self._tex_dir  = directory
                self._tex_dirty = True
                self.update()

        def set_mesh_visible(self, visible: bool) -> None:
            self._show_mesh = visible
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

        def _build_mesh(self) -> None:
            """Build static mesh geometry (triangles). Call once per SMD load."""
            self._verts, self._tris = [], []
            self._mat_tris = {}

            if self._smd is None:
                return

            for tri in self._smd.triangles:
                base = len(self._verts)
                for v in (tri.v0, tri.v1, tri.v2):
                    self._verts.append((v.x, v.y, v.z))
                self._tris.append((base, base + 1, base + 2))
                mat = tri.material
                if mat not in self._mat_tris:
                    self._mat_tris[mat] = []
                self._mat_tris[mat].append((
                    (tri.v0.x, tri.v0.y, tri.v0.z, tri.v0.u, tri.v0.v),
                    (tri.v1.x, tri.v1.y, tri.v1.z, tri.v1.u, tri.v1.v),
                    (tri.v2.x, tri.v2.y, tri.v2.z, tri.v2.u, tri.v2.v),
                ))

        def _build_skeleton(self, frame_idx: int) -> None:
            """
            Recompute bone positions for *frame_idx*.
            When an animation SMD is set, bones are positioned using the anim
            transforms (matched to ref bones by name); otherwise uses the ref SMD.
            """
            self._bone_pos, self._bone_lines = {}, []

            if self._smd is None:
                return

            if self._anim_smd is not None:
                anim_world  = compute_world_transforms(self._anim_smd, frame_idx)
                anim_by_name = {n.name: n.id for n in self._anim_smd.nodes}
                ref_bind    = compute_world_transforms(self._smd, 0)
                for ref_node in self._smd.nodes:
                    anim_bid = anim_by_name.get(ref_node.name)
                    mat = anim_world.get(anim_bid) if anim_bid is not None else None
                    if mat is None:
                        mat = ref_bind.get(ref_node.id, np.eye(4))
                    self._bone_pos[ref_node.id] = (
                        float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])
                    )
            else:
                world = compute_world_transforms(self._smd, frame_idx)
                for bid, mat in world.items():
                    self._bone_pos[bid] = (float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]))

            for node in self._smd.nodes:
                if node.parent_id != -1 and node.parent_id in self._bone_pos:
                    self._bone_lines.append((node.parent_id, node.id))

        def _compute_skinned_verts(self, frame_idx: int) -> None:
            """
            Compute skinned vertex positions using linear blend skinning.
            Stores result in *_skinned_verts* (same length as *_verts*).
            When no animation SMD is set, clears *_skinned_verts* so the bind
            pose *_verts* are used directly.
            """
            if self._smd is None or self._anim_smd is None:
                self._skinned_verts = None
                return

            ref_world   = compute_world_transforms(self._smd, 0)
            anim_world  = compute_world_transforms(self._anim_smd, frame_idx)
            anim_by_name = {n.name: n.id for n in self._anim_smd.nodes}

            # Per-ref-bone skinning matrix: M_anim @ inv(M_bind)
            skin: dict[int, np.ndarray] = {}
            for ref_node in self._smd.nodes:
                M_bind = ref_world.get(ref_node.id, np.eye(4))
                try:
                    M_inv = np.linalg.inv(M_bind)
                except np.linalg.LinAlgError:
                    M_inv = np.eye(4)
                anim_bid = anim_by_name.get(ref_node.name)
                M_anim = anim_world.get(anim_bid, M_bind) if anim_bid is not None else M_bind
                skin[ref_node.id] = M_anim @ M_inv

            skinned: list[tuple[float, float, float]] = []
            skinned_mat_tris: dict[str, list] = {}
            for tri in self._smd.triangles:
                verts_xyz = []
                for v in (tri.v0, tri.v1, tri.v2):
                    M = skin.get(v.bone_id, np.eye(4))
                    p = M @ np.array([v.x, v.y, v.z, 1.0], dtype=np.float64)
                    verts_xyz.append((float(p[0]), float(p[1]), float(p[2])))
                    skinned.append(verts_xyz[-1])
                mat = tri.material
                if mat not in skinned_mat_tris:
                    skinned_mat_tris[mat] = []
                skinned_mat_tris[mat].append((
                    (verts_xyz[0][0], verts_xyz[0][1], verts_xyz[0][2], tri.v0.u, tri.v0.v),
                    (verts_xyz[1][0], verts_xyz[1][1], verts_xyz[1][2], tri.v1.u, tri.v1.v),
                    (verts_xyz[2][0], verts_xyz[2][1], verts_xyz[2][2], tri.v2.u, tri.v2.v),
                ))
            self._skinned_verts    = skinned
            self._skinned_mat_tris = skinned_mat_tris

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
            if self._show_mesh:
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

            active_mat_tris = (
                self._skinned_mat_tris
                if self._skinned_mat_tris is not None
                else self._mat_tris
            )
            if self._show_textures and active_mat_tris:
                if self._tex_dirty:
                    self._load_textures()
                    self._tex_dirty = False
                glEnable(GL_TEXTURE_2D)
                for mat, tris_data in active_mat_tris.items():
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
                active = self._skinned_verts if self._skinned_verts is not None else self._verts
                # Solid fill (semi-transparent)
                glColor4f(0.55, 0.65, 0.80, 0.20)
                glBegin(GL_TRIANGLES)
                for a, b, c in self._tris:
                    for idx in (a, b, c):
                        x, y, z = active[idx]
                        glVertex3f(x, y, z)
                glEnd()
                # Wireframe overlay
                glColor4f(0.45, 0.55, 0.70, 0.35)
                glLineWidth(0.6)
                glBegin(GL_LINES)
                for a, b, c in self._tris:
                    for u, v in ((a, b), (b, c), (c, a)):
                        x1, y1, z1 = active[u]
                        x2, y2, z2 = active[v]
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
# SMD Editor viewport (triangle picking + selection highlight)
# ---------------------------------------------------------------------------

if _GL_OK:
    class _SMDEditorViewport(QOpenGLWidget):
        """Orbit 3-D viewport for the SMD Editor — triangle picking and highlight."""

        triangleSelected       = pyqtSignal(int)  # emits SMD triangle index
        triangleDoubleClicked  = pyqtSignal(int)  # emits SMD triangle index

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._smd:          SMD | None = None
            self._tex_dir:      str        = ""
            # [{idx, mat, v: [(x,y,z,u,v)×3]}]
            self._tris_data:    list       = []
            self._textures:     dict[str, int] = {}
            self._tex_overrides: dict[str, str] = {}  # orig_mat → file_path
            self._tex_dirty:    bool       = True
            self._show_textures: bool      = False
            self._selected_tri: int | None = None

            self._azimuth   = 45.0
            self._elevation = 20.0
            self._distance  = 50.0
            self._target    = np.zeros(3, dtype=float)
            self._last_mouse: QPoint | None     = None
            self._drag_btn:   Qt.MouseButton | None = None

            self.setMinimumSize(300, 300)

        # ── Public API ───────────────────────────────────────────────────

        def set_smd(self, smd: SMD | None, tex_dir: str = "") -> None:
            self._smd          = smd
            self._tex_dir      = tex_dir
            self._selected_tri = None
            self._tris_data    = []
            self._tex_dirty    = True
            self._tex_overrides.clear()
            self._free_textures()
            if smd:
                self._build_tris()
                self._auto_frame()
            self.update()

        def set_textures_visible(self, visible: bool) -> None:
            self._show_textures = visible
            if visible:
                self._tex_dirty = True
            self.update()

        def set_selected(self, tri_idx: int | None) -> None:
            self._selected_tri = tri_idx
            self.update()

        def set_tex_override(self, orig_mat: str, file_path: str) -> None:
            """Display *file_path* wherever *orig_mat* is used."""
            self._tex_overrides[orig_mat] = file_path
            self._tex_dirty = True
            self.update()

        def clear_tex_override(self, orig_mat: str) -> None:
            self._tex_overrides.pop(orig_mat, None)
            self._tex_dirty = True
            self.update()

        def clear_all_tex_overrides(self) -> None:
            self._tex_overrides.clear()
            self._tex_dirty = True
            self.update()

        def materials(self) -> list[str]:
            """Return a sorted, deduplicated list of material names in the current SMD."""
            return sorted({d['mat'] for d in self._tris_data})

        def rebuild_from_smd(self) -> None:
            """Re-read triangle geometry from the current SMD (call after editing)."""
            self._tris_data = []
            self._tex_dirty = True
            if self._smd:
                self._build_tris()
                self._auto_frame()
            self.update()

        # ── Geometry ─────────────────────────────────────────────────────

        def _build_tris(self) -> None:
            if self._smd is None:
                return
            for i, tri in enumerate(self._smd.triangles):
                self._tris_data.append({
                    'idx': i,
                    'mat': tri.material,
                    'v': [
                        (tri.v0.x, tri.v0.y, tri.v0.z, tri.v0.u, tri.v0.v),
                        (tri.v1.x, tri.v1.y, tri.v1.z, tri.v1.u, tri.v1.v),
                        (tri.v2.x, tri.v2.y, tri.v2.z, tri.v2.u, tri.v2.v),
                    ],
                })

        def _auto_frame(self) -> None:
            pts = [(d['v'][j][0], d['v'][j][1], d['v'][j][2])
                   for d in self._tris_data for j in range(3)]
            if not pts:
                return
            arr = np.array(pts, dtype=float)
            self._target   = arr.mean(axis=0)
            extent         = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
            self._distance = max(extent * 1.2, 1.0)

        def _eye(self) -> np.ndarray:
            az = math.radians(self._azimuth)
            el = math.radians(max(-88.0, min(88.0, self._elevation)))
            return self._target + self._distance * np.array([
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el),
            ])

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
            if not self._tris_data:
                return
            try:
                from PIL import Image
            except ImportError:
                return

            tex_dir = Path(self._tex_dir) if self._tex_dir else None
            dir_files: dict[str, Path] = {}
            if tex_dir and tex_dir.is_dir():
                for f in tex_dir.iterdir():
                    dir_files[f.name.lower()] = f

            def _load_img(fpath: Path) -> int | None:
                try:
                    img  = Image.open(fpath).convert("RGBA")
                    img  = img.transpose(Image.FLIP_TOP_BOTTOM)
                    data = img.tobytes()
                    w, h = img.size
                    tid  = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, tid)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                                 GL_RGBA, GL_UNSIGNED_BYTE, data)
                    glBindTexture(GL_TEXTURE_2D, 0)
                    return tid
                except Exception:
                    return None

            mats = {d['mat'] for d in self._tris_data}
            for mat in mats:
                # Check for a user-specified override first
                override = self._tex_overrides.get(mat)
                if override:
                    tid = _load_img(Path(override))
                    if tid is not None:
                        self._textures[mat] = tid
                    continue
                # Fall back to auto-discovery in tex_dir
                fname = Path(mat).name
                if not fname.lower().endswith(".bmp"):
                    fname = fname + ".bmp"
                fpath = dir_files.get(fname.lower())
                if fpath is None:
                    continue
                tid = _load_img(fpath)
                if tid is not None:
                    self._textures[mat] = tid

        # ── OpenGL callbacks ─────────────────────────────────────────────

        def initializeGL(self) -> None:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
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
            gluLookAt(
                eye[0], eye[1], eye[2],
                self._target[0], self._target[1], self._target[2],
                0.0, 0.0, 1.0,
            )

            self._draw_grid()
            self._draw_triangles()

        # ── Drawing ──────────────────────────────────────────────────────

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

        def _draw_triangles(self) -> None:
            if not self._tris_data:
                return

            sel = self._selected_tri

            if self._show_textures:
                if self._tex_dirty:
                    self._load_textures()
                    self._tex_dirty = False
                by_mat: dict[str, list] = {}
                for d in self._tris_data:
                    if d['idx'] == sel:
                        continue
                    by_mat.setdefault(d['mat'], []).append(d)
                glEnable(GL_TEXTURE_2D)
                for mat, tris in by_mat.items():
                    tid = self._textures.get(mat)
                    if tid:
                        glBindTexture(GL_TEXTURE_2D, tid)
                        glColor4f(1.0, 1.0, 1.0, 1.0)
                    else:
                        glBindTexture(GL_TEXTURE_2D, 0)
                        glColor4f(0.55, 0.65, 0.80, 0.80)
                    glBegin(GL_TRIANGLES)
                    for d in tris:
                        for (x, y, z, u, v) in d['v']:
                            glTexCoord2f(u, v)
                            glVertex3f(x, y, z)
                    glEnd()
                glBindTexture(GL_TEXTURE_2D, 0)
                glDisable(GL_TEXTURE_2D)
            else:
                # Solid fill (semi-transparent)
                glColor4f(0.55, 0.65, 0.80, 0.25)
                glBegin(GL_TRIANGLES)
                for d in self._tris_data:
                    if d['idx'] == sel:
                        continue
                    for (x, y, z, u, v) in d['v']:
                        glVertex3f(x, y, z)
                glEnd()

            # Wireframe overlay
            glColor4f(0.45, 0.55, 0.70, 0.50)
            glLineWidth(0.6)
            glBegin(GL_LINES)
            for d in self._tris_data:
                if d['idx'] == sel:
                    continue
                vv = d['v']
                for (a, b) in ((0, 1), (1, 2), (2, 0)):
                    glVertex3f(vv[a][0], vv[a][1], vv[a][2])
                    glVertex3f(vv[b][0], vv[b][1], vv[b][2])
            glEnd()

            # Selected triangle highlight
            if sel is not None:
                sd = next((d for d in self._tris_data if d['idx'] == sel), None)
                if sd:
                    glColor4f(1.0, 0.85, 0.1, 0.85)
                    glBegin(GL_TRIANGLES)
                    for (x, y, z, u, v) in sd['v']:
                        glVertex3f(x, y, z)
                    glEnd()
                    glColor4f(1.0, 0.85, 0.1, 1.0)
                    glLineWidth(2.5)
                    glBegin(GL_LINES)
                    vv = sd['v']
                    for (a, b) in ((0, 1), (1, 2), (2, 0)):
                        glVertex3f(vv[a][0], vv[a][1], vv[a][2])
                        glVertex3f(vv[b][0], vv[b][1], vv[b][2])
                    glEnd()

        # ── Mouse events ─────────────────────────────────────────────────

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton:
                idx = self._pick_triangle(event.pos().x(), event.pos().y())
                if idx is not None:
                    self._selected_tri = idx
                    self.triangleSelected.emit(idx)
                    self.update()
            self._last_mouse = event.pos()
            self._drag_btn   = event.button()

        def mouseDoubleClickEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton:
                idx = self._pick_triangle(event.pos().x(), event.pos().y())
                if idx is not None:
                    self._selected_tri = idx
                    self.triangleDoubleClicked.emit(idx)
                    self.update()

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
                    Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton,
                ):
                    az    = math.radians(self._azimuth)
                    right = np.array([-math.sin(az), math.cos(az), 0.0])
                    fwd   = self._eye() - self._target
                    up    = np.cross(right, fwd)
                    n     = np.linalg.norm(up)
                    up    = up / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
                    scale          = self._distance * 0.0018
                    self._target  -= right * dx * scale
                    self._target  += up    * dy * scale
                    self.update()
            self._last_mouse = pos

        def wheelEvent(self, event) -> None:
            factor        = 0.88 if event.angleDelta().y() > 0 else 1.14
            self._distance = max(0.2, self._distance * factor)
            self.update()

        # ── Triangle picking (Möller–Trumbore) ───────────────────────────

        def _pick_triangle(self, mx: int, my: int) -> int | None:
            """
            Cast a perspective ray from screen pixel (mx, my) and return the
            index of the nearest hit triangle.

            The ray is computed analytically from our camera parameters so
            it is always consistent with what paintGL renders, without relying
            on querying the OpenGL matrix state after the frame.
            """
            if not self._tris_data:
                return None

            w, h = self.width(), self.height()
            if w == 0 or h == 0:
                return None

            # Camera basis vectors (same as paintGL / gluLookAt)
            eye    = self._eye()
            target = self._target
            fwd    = target - eye
            fwd_n  = np.linalg.norm(fwd)
            if fwd_n < 1e-12:
                return None
            fwd /= fwd_n

            world_up = np.array([0.0, 0.0, 1.0])
            right    = np.cross(fwd, world_up)
            r_n      = np.linalg.norm(right)
            if r_n < 1e-9:
                # Looking straight up/down — pick an arbitrary right
                right = np.array([1.0, 0.0, 0.0])
            else:
                right /= r_n
            up = np.cross(right, fwd)   # true up (already unit length)

            # Perspective: fov=45° (same as gluPerspective(45, ...))
            tan_half = math.tan(math.radians(45.0) / 2.0)
            aspect   = w / h

            # NDC coords: pixel (0,0) is top-left; NDC (-1,-1) is bottom-left
            ndc_x =  (2.0 * mx / w) - 1.0
            ndc_y = -(2.0 * my / h) + 1.0   # flip Y

            ray_dir = fwd + right * (ndc_x * tan_half * aspect) + up * (ndc_y * tan_half)
            n_rd    = np.linalg.norm(ray_dir)
            if n_rd < 1e-12:
                return None
            ray_dir /= n_rd

            orig    = eye
            EPSILON = 1e-9
            best_t  = float('inf')
            best_idx = None

            for td in self._tris_data:
                v0    = np.array(td['v'][0][:3], dtype=float)
                v1    = np.array(td['v'][1][:3], dtype=float)
                v2    = np.array(td['v'][2][:3], dtype=float)
                edge1 = v1 - v0
                edge2 = v2 - v0
                hh    = np.cross(ray_dir, edge2)
                a     = np.dot(edge1, hh)
                if abs(a) < EPSILON:
                    continue
                f  = 1.0 / a
                s  = orig - v0
                u  = f * np.dot(s, hh)
                if u < 0.0 or u > 1.0:
                    continue
                q  = np.cross(s, edge1)
                vv = f * np.dot(ray_dir, q)
                if vv < 0.0 or u + vv > 1.0:
                    continue
                t = f * np.dot(edge2, q)
                if t > EPSILON and t < best_t:
                    best_t   = t
                    best_idx = td['idx']

            return best_idx

else:
    class _SMDEditorViewport(QLabel):  # type: ignore[no-redef]
        triangleSelected      = pyqtSignal(int)
        triangleDoubleClicked = pyqtSignal(int)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(
                "OpenGL not available.\nInstall PyOpenGL:  pip install PyOpenGL",
                parent,
            )
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        def set_smd(self, smd: SMD | None, tex_dir: str = "") -> None:
            pass

        def set_textures_visible(self, visible: bool) -> None:
            pass

        def set_tex_override(self, orig_mat: str, file_path: str) -> None:
            pass

        def clear_tex_override(self, orig_mat: str) -> None:
            pass

        def clear_all_tex_overrides(self) -> None:
            pass

        def materials(self) -> list[str]:
            return []

        def rebuild_from_smd(self) -> None:
            pass

        def set_selected(self, tri_idx: int | None) -> None:
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

    bonesRenamed       = pyqtSignal()
    qcModified         = pyqtSignal(str)   # model_name whose QC was changed on disk
    # Emitted after every successful operation: (description, op_type)
    # MainWindow listens to record history.
    operationRecorded  = pyqtSignal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._models:       list  = []          # list[ModelInput]
        self._dirs:         dict[str, str] = {}
        self._cur_smd:      SMD | None = None   # reference SMD
        self._cur_anim_smd: SMD | None = None   # animation SMD
        self._cur_model_name: str = ""
        # Recorded ops not yet propagated to all SMDs.
        # Each entry is {"type": "rename", "old": str, "new": str}
        #             or {"type": "delete", "name": str}
        self._pending_ops: list[dict] = []
        # Reference-hands state
        self._hands_smd:      SMD | None           = None
        self._hands_path:     str                  = ""
        self._bone_map:       dict[str, str]       = {}  # weapon_name → hand_name
        self._hand_combo_map: dict[int, QComboBox] = {}  # hand_bone_id → QComboBox
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

        ll.addWidget(QLabel("Reference SMD (mesh / textures):"))
        self._ref_combo = QComboBox()
        self._ref_combo.currentIndexChanged.connect(self._on_ref_changed)
        ll.addWidget(self._ref_combo)

        ll.addWidget(QLabel("Animation SMD:"))
        self._anim_combo = QComboBox()
        self._anim_combo.currentIndexChanged.connect(self._on_anim_combo_changed)
        ll.addWidget(self._anim_combo)

        ll.addWidget(QLabel("Bones:"))
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "ID", "Parent"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(lambda item, _: self._rename(item))
        ll.addWidget(self._tree, 1)

        bone_btns = QHBoxLayout()
        bone_btns.setSpacing(4)

        self._btn_rename = QPushButton("Rename")
        self._btn_rename.setEnabled(False)
        self._btn_rename.clicked.connect(self._on_rename_clicked)
        bone_btns.addWidget(self._btn_rename)

        self._btn_delete = QPushButton("Delete")
        self._btn_delete.setEnabled(False)
        self._btn_delete.setStyleSheet("color: #e05555;")
        self._btn_delete.setToolTip("Delete selected bone")
        self._btn_delete.clicked.connect(self._on_delete_clicked)
        bone_btns.addWidget(self._btn_delete)

        self._btn_apply_all = QPushButton("Apply to All")
        self._btn_apply_all.setEnabled(False)
        self._btn_apply_all.setToolTip(
            "Replay all pending renames / deletions on every SMD in this model"
        )
        self._btn_apply_all.clicked.connect(self._on_apply_all_clicked)
        bone_btns.addWidget(self._btn_apply_all)

        self._btn_export = QPushButton("Export")
        self._btn_export.setEnabled(False)
        self._btn_export.setToolTip("Export current SMD to disk")
        self._btn_export.clicked.connect(self._on_export_clicked)
        bone_btns.addWidget(self._btn_export)

        self._btn_save_all = QPushButton("Save All")
        self._btn_save_all.setEnabled(False)
        self._btn_save_all.setToolTip("Overwrite all original SMD files for this model")
        self._btn_save_all.setStyleSheet("font-weight: bold;")
        self._btn_save_all.clicked.connect(self._on_save_all_clicked)
        bone_btns.addWidget(self._btn_save_all)

        ll.addLayout(bone_btns)

        # ── Reference Hands group ─────────────────────────────────────────
        hands_box = QGroupBox("Reference Hands")
        hands_box.setCheckable(False)
        hl = QVBoxLayout(hands_box)
        hl.setContentsMargins(6, 6, 6, 6)
        hl.setSpacing(4)

        # Load button + path label
        self._btn_load_hands = QPushButton("Load Hands SMD…")
        self._btn_load_hands.clicked.connect(self._on_load_hands_smd)
        hl.addWidget(self._btn_load_hands)
        self._hands_path_label = QLabel("(none)")
        self._hands_path_label.setStyleSheet("color: #888; font-size: 10px;")
        self._hands_path_label.setWordWrap(True)
        hl.addWidget(self._hands_path_label)

        # Hand-bone tree with per-row weapon-bone dropdowns
        self._hands_tree = QTreeWidget()
        self._hands_tree.setColumnCount(2)
        self._hands_tree.setHeaderLabels(["Hand Bone", "Weapon Bone"])
        self._hands_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._hands_tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._hands_tree.setMinimumHeight(140)
        hl.addWidget(self._hands_tree, 1)

        # Bottom row: status + buttons
        self._hands_status = QLabel("")
        self._hands_status.setStyleSheet("color: #888; font-size: 10px;")
        self._hands_status.setWordWrap(True)
        hl.addWidget(self._hands_status)

        bottom_row = QHBoxLayout()
        self._btn_clear_hands = QPushButton("Clear")
        self._btn_clear_hands.setEnabled(False)
        self._btn_clear_hands.clicked.connect(self._on_clear_hands_map)
        bottom_row.addWidget(self._btn_clear_hands)

        self._btn_apply_hands = QPushButton("Apply Hands to All SMDs")
        self._btn_apply_hands.setEnabled(False)
        self._btn_apply_hands.setStyleSheet(
            "background-color: #2a6099; color: white; font-weight: bold;"
        )
        self._btn_apply_hands.clicked.connect(self._on_apply_hands)
        bottom_row.addWidget(self._btn_apply_hands)
        hl.addLayout(bottom_row)

        ll.addWidget(hands_box)

        left.setMinimumWidth(240)
        splitter.addWidget(left)

        # ── Right panel ──────────────────────────────────────────────────
        right  = QWidget()
        rl     = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # Toolbar above viewport
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        self._chk_mesh = QCheckBox("Show Mesh")
        self._chk_mesh.setChecked(True)
        self._chk_mesh.toggled.connect(self._on_mesh_toggled)
        toolbar.addWidget(self._chk_mesh)

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

        # ── Animation playback bar (hidden for reference SMDs) ───────────
        self._anim_bar = QWidget()
        abl = QHBoxLayout(self._anim_bar)
        abl.setContentsMargins(4, 2, 4, 2)
        abl.setSpacing(4)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(32)
        self._btn_play.setToolTip("Play / Pause")
        self._btn_play.clicked.connect(self._on_play_pause)
        abl.addWidget(self._btn_play)

        self._anim_slider = QSlider(Qt.Orientation.Horizontal)
        self._anim_slider.setMinimum(0)
        self._anim_slider.setValue(0)
        self._anim_slider.valueChanged.connect(self._on_frame_slider)
        abl.addWidget(self._anim_slider, 1)

        self._anim_label = QLabel("0 / 0")
        self._anim_label.setFixedWidth(70)
        abl.addWidget(self._anim_label)

        self._anim_bar.setVisible(False)
        rl.addWidget(self._anim_bar)

        self._info_label = QLabel("  Left-drag: orbit   Right-drag: pan   Scroll: zoom   Hover bone to inspect")
        self._info_label.setStyleSheet("color: #888; padding: 3px 6px; font-size: 11px;")
        rl.addWidget(self._info_label)

        # Playback timer
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)   # ~30 fps
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_playing = False

        splitter.addWidget(right)
        splitter.setSizes([260, 700])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)
        # Enforce 50 % maximum on the side panel once the widget has a real size.
        splitter.splitterMoved.connect(
            lambda pos, idx, s=splitter: s.moveSplitter(
                min(pos, s.width() // 2), idx
            ) if pos > s.width() // 2 else None
        )

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

    def get_current_model_smds(self) -> tuple[str, dict]:
        """Return (model_name, smds_dict) for the currently selected model."""
        name  = self._cur_model_name
        model = next((m for m in self._models if m.name == name), None)
        return name, dict(model.smds) if model else {}

    def restore_model_smds(self, model_name: str, smds: dict) -> None:
        """Replace in-memory SMDs for *model_name* and refresh the viewer."""
        model = next((m for m in self._models if m.name == model_name), None)
        if model is None:
            return
        model.smds.clear()
        model.smds.update(smds)
        # If this is the currently displayed model, reload the view.
        if model_name == self._cur_model_name:
            self._on_model_changed()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_model_changed(self) -> None:
        self._anim_timer.stop()
        self._anim_playing = False
        self._btn_play.setText("▶")
        name = self._model_combo.currentText()
        self._cur_model_name = name
        self._pending_ops.clear()
        self._update_apply_button()
        model = next((m for m in self._models if m.name == name), None)
        keys  = sorted(model.smds.keys()) if model else []

        prev_ref  = self._ref_combo.currentText()
        prev_anim = self._anim_combo.currentText()

        self._ref_combo.blockSignals(True)
        self._anim_combo.blockSignals(True)
        self._ref_combo.clear()
        self._anim_combo.clear()
        self._anim_combo.addItem("(none)")
        for key in keys:
            self._ref_combo.addItem(key)
            self._anim_combo.addItem(key)
        self._ref_combo.setCurrentIndex(max(0, self._ref_combo.findText(prev_ref)))
        anim_idx = self._anim_combo.findText(prev_anim)
        self._anim_combo.setCurrentIndex(max(0, anim_idx))
        self._ref_combo.blockSignals(False)
        self._anim_combo.blockSignals(False)

        self._on_ref_changed()

    def _on_ref_changed(self) -> None:
        """Called when the reference (mesh/texture) SMD selection changes."""
        key   = self._ref_combo.currentText()
        model = next((m for m in self._models if m.name == self._cur_model_name), None)
        self._cur_smd = model.smds.get(key) if model else None
        self._rebuild_tree()
        self._viewport.set_smd(self._cur_smd)
        tex_dir = self._dirs.get(self._cur_model_name, "")
        self._viewport.set_texture_dir(tex_dir)
        has_model_dir = bool(self._dirs.get(self._cur_model_name, ""))
        self._btn_export.setEnabled(self._cur_smd is not None)
        self._btn_save_all.setEnabled(has_model_dir and model is not None)
        self._rebuild_hands_tree()
        # Re-apply the current animation selection on top of the new ref SMD
        self._on_anim_combo_changed()

    def _on_anim_combo_changed(self) -> None:
        """Called when the animation SMD selection changes."""
        self._anim_timer.stop()
        self._anim_playing = False
        self._btn_play.setText("▶")

        key   = self._anim_combo.currentText()
        model = next((m for m in self._models if m.name == self._cur_model_name), None)
        self._cur_anim_smd = (
            model.smds.get(key) if model and key != "(none)" else None
        )
        self._viewport.set_anim_smd(self._cur_anim_smd)
        self._setup_anim_bar()

    def _setup_anim_bar(self) -> None:
        """Show / hide the animation playback bar based on the animation SMD."""
        smd      = self._cur_anim_smd
        n_frames = len(smd.skeleton) if smd else 0
        is_anim  = n_frames > 1

        self._anim_bar.setVisible(is_anim)
        if is_anim:
            self._anim_slider.blockSignals(True)
            self._anim_slider.setMaximum(n_frames - 1)
            self._anim_slider.setValue(0)
            self._anim_slider.blockSignals(False)
            self._anim_label.setText(f"0 / {n_frames - 1}")

    def _on_play_pause(self) -> None:
        if self._cur_smd is None:
            return
        self._anim_playing = not self._anim_playing
        if self._anim_playing:
            self._btn_play.setText("⏸")
            self._anim_timer.start()
        else:
            self._btn_play.setText("▶")
            self._anim_timer.stop()

    def _on_anim_tick(self) -> None:
        smd = self._cur_anim_smd
        if smd is None:
            return
        n_frames = len(smd.skeleton)
        cur = self._anim_slider.value()
        nxt = (cur + 1) % n_frames
        self._anim_slider.blockSignals(True)
        self._anim_slider.setValue(nxt)
        self._anim_slider.blockSignals(False)
        self._anim_label.setText(f"{nxt} / {n_frames - 1}")
        self._viewport.set_frame(nxt)

    def _on_frame_slider(self, value: int) -> None:
        smd = self._cur_anim_smd
        if smd is None:
            return
        n_frames = len(smd.skeleton)
        self._anim_label.setText(f"{value} / {n_frames - 1}")
        self._viewport.set_frame(value)

    def _on_mesh_toggled(self, checked: bool) -> None:
        self._viewport.set_mesh_visible(checked)

    def _on_textures_toggled(self, checked: bool) -> None:
        self._viewport.set_textures_visible(checked)

    def _on_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        has_sel = bool(items)
        has_smd = self._cur_smd is not None
        self._btn_rename.setEnabled(has_sel)
        self._btn_delete.setEnabled(has_sel)
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
        bone_name   = items[0].text(0)
        descendants = _collect_descendants(self._cur_smd, bone_name)

        # Build confirmation message
        if descendants:
            detail = "Also deleting children (leaf-first):\n" + "\n".join(
                f"  • {n}" for n in descendants
            )
            msg = (
                f"Delete bone '{bone_name}' and its {len(descendants)} "
                f"descendant(s) from the current SMD?\n\n"
                "Use 'Apply to All' to propagate to all SMDs in the model."
            )
        else:
            detail = None
            msg = (
                f"Delete bone '{bone_name}' from the current SMD?\n\n"
                "Use 'Apply to All' to propagate this to all SMDs in the model."
            )

        mb = QMessageBox(self)
        mb.setWindowTitle("Delete Bone")
        mb.setIcon(QMessageBox.Icon.Question)
        mb.setText(msg)
        if detail:
            mb.setDetailedText(detail)
        mb.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if mb.exec() != QMessageBox.StandardButton.Yes:
            return

        # Delete descendants first (leaf-first order), then the selected bone
        deleted: list[str] = []
        for name in descendants:
            if _delete_bone(self._cur_smd, name):
                self._pending_ops.append({"type": "delete", "name": name})
                deleted.append(name)
        if _delete_bone(self._cur_smd, bone_name):
            self._pending_ops.append({"type": "delete", "name": bone_name})
            deleted.append(bone_name)

        if deleted:
            self._update_apply_button()
            self._rebuild_tree()
            self._viewport.set_smd(self._cur_smd)
            self.bonesRenamed.emit()
            desc = (
                f"Deleted bone '{bone_name}' + {len(deleted)-1} child(ren)"
                if len(deleted) > 1 else f"Deleted bone '{bone_name}'"
            )
            self.operationRecorded.emit(
                f"{desc} in {self._ref_combo.currentText()}",
                "delete",
            )

    def _on_apply_all_clicked(self) -> None:
        if not self._pending_ops:
            return
        model = next((m for m in self._models if m.name == self._cur_model_name), None)
        if model is None:
            return

        # Build human-readable summary
        lines = []
        for op in self._pending_ops:
            if op["type"] == "rename":
                lines.append(f"  Rename  '{op['old']}'  →  '{op['new']}'")
            else:
                lines.append(f"  Delete  '{op['name']}'")
        summary = "\n".join(lines)
        other_count = len(model.smds) - 1  # current SMD already has the changes

        mb = QMessageBox(self)
        mb.setWindowTitle("Apply to All SMDs")
        mb.setText(
            f"Apply {len(self._pending_ops)} pending change(s) to the other "
            f"{other_count} SMD(s) in '{self._cur_model_name}'?"
        )
        mb.setDetailedText(summary)
        mb.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        ans = mb.exec()
        if ans != QMessageBox.StandardButton.Yes:
            return

        cur_key = self._ref_combo.currentText()
        applied = 0
        for key, smd in model.smds.items():
            if key == cur_key:
                continue  # already done
            for op in self._pending_ops:
                if op["type"] == "rename":
                    for node in smd.nodes:
                        if node.name == op["old"]:
                            node.name = op["new"]
                else:
                    _delete_bone(smd, op["name"])
            applied += 1

        n_ops = len(self._pending_ops)
        self._pending_ops.clear()
        self._update_apply_button()
        self.bonesRenamed.emit()
        self.operationRecorded.emit(
            f"Applied {n_ops} op(s) to {applied} SMD(s) in '{self._cur_model_name}'",
            "apply_all",
        )
        QMessageBox.information(
            self, "Done",
            f"Changes applied to {applied} SMD(s).",
        )

    def _update_apply_button(self) -> None:
        n = len(self._pending_ops)
        if n == 0:
            self._btn_apply_all.setText("Apply to All")
            self._btn_apply_all.setEnabled(False)
        else:
            self._btn_apply_all.setText(f"Apply to All  ({n} pending)")
            self._btn_apply_all.setEnabled(True)

    def _on_export_clicked(self) -> None:
        if self._cur_smd is None:
            return
        smd_key = self._ref_combo.currentText()
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

    def _on_save_all_clicked(self) -> None:
        model = next((m for m in self._models if m.name == self._cur_model_name), None)
        if model is None:
            return
        model_dir = self._dirs.get(self._cur_model_name, "")
        if not model_dir:
            QMessageBox.warning(self, "Save All", "Model directory is unknown.")
            return

        n = len(model.smds)
        ans = QMessageBox.question(
            self, "Save All SMDs",
            f"Overwrite all {n} original SMD file(s) in:\n{model_dir}\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        errors: list[str] = []
        saved = 0
        for key, smd in model.smds.items():
            try:
                smd.save(Path(model_dir) / (key + ".smd"))
                saved += 1
            except Exception as exc:
                errors.append(f"{key}: {exc}")

        if errors:
            mb = QMessageBox(self)
            mb.setWindowTitle("Save All")
            mb.setIcon(QMessageBox.Icon.Warning)
            mb.setText(f"Saved {saved}/{n} file(s). {len(errors)} error(s) occurred.")
            mb.setDetailedText("\n".join(errors))
            mb.exec()
        else:
            QMessageBox.information(
                self, "Save All",
                f"Saved {saved} SMD file(s) to:\n{model_dir}",
            )
            self.operationRecorded.emit(
                f"Saved all {saved} SMD(s) to disk for '{self._cur_model_name}'",
                "save_all",
            )

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

        # Apply to current SMD only; record op so Apply-to-All can propagate it
        if self._cur_smd is not None:
            for node in self._cur_smd.nodes:
                if node.name == old_name:
                    node.name = new_name
        self._pending_ops.append({"type": "rename", "old": old_name, "new": new_name})
        self._update_apply_button()
        self._rebuild_tree()
        self.bonesRenamed.emit()
        self.operationRecorded.emit(
            f"Renamed bone '{old_name}' -> '{new_name}' in {self._ref_combo.currentText()}",
            "rename",
        )

    # ── Reference Hands ───────────────────────────────────────────────────

    def _on_load_hands_smd(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference Hands SMD", "", "SMD Files (*.smd);;All files (*)"
        )
        if not path:
            return
        try:
            self._hands_smd  = SMD.from_file(path)
            self._hands_path = path
            self._hands_path_label.setText(Path(path).name)
            self._rebuild_hands_tree()
        except Exception as exc:
            QMessageBox.critical(self, "Load Hands SMD", str(exc))

    def _rebuild_hands_tree(self) -> None:
        """
        Rebuild the hand-bone tree widget.  Each row shows a hand bone in
        column 0 and a QComboBox of weapon-bone names (+ '(none)') in column 1.
        Selecting a weapon bone auto-assigns children by name recursively.
        """
        self._hands_tree.clear()
        self._hand_combo_map.clear()
        self._bone_map.clear()
        self._hands_status.setText("")
        self._btn_apply_hands.setEnabled(False)
        self._btn_clear_hands.setEnabled(False)

        if self._hands_smd is None:
            return

        weapon_names = ["(none)"]
        if self._cur_smd:
            weapon_names += sorted(n.name for n in self._cur_smd.nodes)

        # Build tree items (need parents before children → sort by id)
        hand_items: dict[int, QTreeWidgetItem] = {}
        for node in sorted(self._hands_smd.nodes, key=lambda n: n.id):
            item = QTreeWidgetItem([node.name, ""])
            combo = QComboBox()
            combo.addItems(weapon_names)
            combo.currentTextChanged.connect(
                lambda text, n=node: self._on_hand_combo_changed(n, text)
            )
            self._hand_combo_map[node.id] = combo
            hand_items[node.id] = item

            if node.parent_id != -1 and node.parent_id in hand_items:
                hand_items[node.parent_id].addChild(item)
            else:
                self._hands_tree.addTopLevelItem(item)

            self._hands_tree.setItemWidget(item, 1, combo)

        self._hands_tree.expandAll()

    def _on_hand_combo_changed(self, hand_node: "Node", weapon_name: str) -> None:
        """
        Called when the user picks a weapon bone for *hand_node*.
        Updates bone_map and auto-assigns children by matching names.
        """
        # Remove any old mapping for this hand bone
        self._bone_map = {
            wp: hp for wp, hp in self._bone_map.items()
            if hp != hand_node.name
        }

        if weapon_name and weapon_name != "(none)":
            self._bone_map[weapon_name] = hand_node.name
            if self._cur_smd and self._hands_smd:
                # Walk up both parent chains (handles non-matching names)
                self._auto_assign_parents(hand_node, weapon_name)
                # Walk down children by name (same-name rigs)
                self._auto_assign_children(hand_node, weapon_name)

        n = len(self._bone_map)
        total_weapon = len(self._cur_smd.nodes) if self._cur_smd else 0
        weapon_specific = total_weapon - n
        if n:
            self._hands_status.setText(
                f"{n} mapped · {weapon_specific} weapon-specific"
            )
        else:
            self._hands_status.setText("")

        self._btn_apply_hands.setEnabled(bool(self._bone_map))
        self._btn_clear_hands.setEnabled(bool(self._bone_map))
        self._refresh_all_combos()

    def _auto_assign_parents(self, hand_node: "Node", weapon_name: str) -> None:
        """
        Walk one level up both the hand-bone tree and the weapon-bone tree and
        auto-select the weapon parent in the hand-parent's combo box (if it is
        still '(none)').  Setting that combo triggers its own signal, which
        calls this method again for the next level — propagating all the way to
        the root without an explicit loop.
        """
        if self._cur_smd is None or self._hands_smd is None:
            return

        hand_by_id   = {n.id: n for n in self._hands_smd.nodes}
        weapon_by_id = {n.id: n for n in self._cur_smd.nodes}
        weapon_by_name = {n.name: n for n in self._cur_smd.nodes}

        hand_parent   = hand_by_id.get(hand_node.parent_id)
        weapon_node   = weapon_by_name.get(weapon_name)
        if hand_parent is None or weapon_node is None:
            return

        weapon_parent = weapon_by_id.get(weapon_node.parent_id)
        if weapon_parent is None:
            return

        combo = self._hand_combo_map.get(hand_parent.id)
        if combo is not None and combo.currentText() == "(none)":
            idx = combo.findText(weapon_parent.name)
            if idx >= 0:
                combo.setCurrentIndex(idx)   # triggers signal → next level

    def _auto_assign_children(self, hand_node: "Node", weapon_name: str) -> None:
        """
        For the hand bone → weapon bone pair, walk their children in parallel
        and auto-select matching weapon children (by name) in the tree combos.
        Only fills combos that are still '(none)' to avoid overwriting user choices.
        """
        if self._cur_smd is None or self._hands_smd is None:
            return

        weapon_node = next(
            (n for n in self._cur_smd.nodes if n.name == weapon_name), None
        )
        if weapon_node is None:
            return

        hand_children   = [n for n in self._hands_smd.nodes if n.parent_id == hand_node.id]
        weapon_children = {
            n.name: n for n in self._cur_smd.nodes if n.parent_id == weapon_node.id
        }

        for hc in hand_children:
            combo = self._hand_combo_map.get(hc.id)
            if combo is None or combo.currentText() != "(none)":
                continue
            if hc.name in weapon_children:
                idx = combo.findText(weapon_children[hc.name].name)
                if idx >= 0:
                    combo.setCurrentIndex(idx)   # triggers signal → recurse

    def _refresh_all_combos(self) -> None:
        """Rebuild every combo's item list so already-used weapon bones
        only appear in the combo that has them selected."""
        if self._cur_smd is None:
            return
        all_weapon = sorted(n.name for n in self._cur_smd.nodes)
        used = set(self._bone_map.keys())  # weapon names currently assigned

        for combo in self._hand_combo_map.values():
            current = combo.currentText()
            available = ["(none)"] + [
                w for w in all_weapon if w not in used or w == current
            ]
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(available)
            idx = combo.findText(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def _on_clear_hands_map(self) -> None:
        """Reset all combo boxes to '(none)'."""
        self._bone_map.clear()
        self._refresh_all_combos()
        for combo in self._hand_combo_map.values():
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._hands_status.setText("")
        self._btn_apply_hands.setEnabled(False)
        self._btn_clear_hands.setEnabled(False)

    def _on_apply_hands(self) -> None:
        if not self._bone_map or self._hands_smd is None:
            return
        model = next(
            (m for m in self._models if m.name == self._cur_model_name), None
        )
        if model is None:
            return

        # Capture original reference SMD BEFORE any modification
        original_ref_smd = self._cur_smd

        n_smds     = len(model.smds)
        matched    = len(self._bone_map)
        model_dir  = self._dirs.get(self._cur_model_name, "")

        mb = QMessageBox(self)
        mb.setWindowTitle("Apply Reference Hands")
        mb.setText(
            f"Replace {matched} hand bone(s) across all {n_smds} SMD(s) in "
            f"'{self._cur_model_name}'?"
        )
        mb.setInformativeText(
            "Weapon SMDs are updated in memory — use 'Save All' to write them.\n"
            "The model directory will also be updated on disk (see details)."
        )
        mb.setDetailedText(
            "Weapon-specific bones will be renumbered to IDs above the last hand bone ID.\n\n"
            "Disk operations:\n"
            "  • Hands SMD (with weapon bones) copied to model directory\n"
            "  • Hand textures copied to model directory\n"
            "  • 'hand*' bodygroup(s) removed from QC + old SMD/texture files deleted\n"
            "  • New 'hands' bodygroup added to QC"
        )
        mb.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        ans = mb.exec()
        if ans != QMessageBox.StandardButton.Yes:
            return

        errors: list[str] = []
        for key, smd in list(model.smds.items()):
            try:
                model.smds[key] = _apply_hand_replacement(
                    smd, self._bone_map, self._hands_smd
                )
            except Exception as exc:
                errors.append(f"{key}: {exc}")

        # Post-apply file operations (only when no per-SMD errors)
        disk_msg    = ""
        qc_modified = False
        if model_dir and original_ref_smd is not None and not errors:
            try:
                disk_msg    = self._post_apply_hands(model_dir, original_ref_smd)
                qc_modified = True
            except Exception as exc:
                errors.append(f"Post-apply disk ops: {exc}")

        self._on_ref_changed()
        self._pending_ops.clear()
        self._update_apply_button()
        self.bonesRenamed.emit()
        if not errors:
            self.operationRecorded.emit(
                f"Applied hands ({matched} bone(s) mapped) to all {n_smds} SMD(s) "
                f"in '{self._cur_model_name}'",
                "apply_hands",
            )
        if qc_modified:
            self.qcModified.emit(self._cur_model_name)

        if errors:
            mb = QMessageBox(self)
            mb.setWindowTitle("Apply Hands — Errors")
            mb.setIcon(QMessageBox.Icon.Warning)
            mb.setText(f"{len(errors)} error(s) occurred.")
            mb.setDetailedText("\n".join(errors))
            mb.exec()
        else:
            mb = QMessageBox(self)
            mb.setWindowTitle("Apply Hands")
            mb.setIcon(QMessageBox.Icon.Information)
            mb.setText(f"Done. {n_smds} SMD(s) updated in memory.")
            mb.setInformativeText("Use 'Save All' to write the modified weapon SMDs to disk.")
            if disk_msg:
                mb.setDetailedText(disk_msg)
            mb.exec()

    def _post_apply_hands(self, model_dir: str, original_ref_smd: SMD) -> str:
        """
        Perform disk-side operations after hand replacement:
        1. Build and write combined hands body SMD to model directory
        2. Copy hand mesh textures to model directory
        3. Update QC: remove 'hand*' bodygroups (deleting their files),
           add new 'hands' bodygroup
        Returns a short summary string.
        """
        model_path  = Path(model_dir)
        hands_path  = Path(self._hands_path)
        hands_stem  = hands_path.stem        # e.g. "male", "default"
        hands_dir   = hands_path.parent

        # 1. Build combined hands body SMD and save
        hands_body     = _build_hands_body_smd(
            self._hands_smd, original_ref_smd, self._bone_map
        )
        dest_smd_path  = model_path / f"{hands_stem}.smd"
        hands_body.save(dest_smd_path)

        # 2. Copy hand textures (BMP files used by the hands mesh)
        hand_mats = {Path(tri.material).name for tri in self._hands_smd.triangles}
        tex_copied: list[str] = []
        for mat_name in hand_mats:
            stem = Path(mat_name).stem
            for candidate in [mat_name, stem + ".bmp", stem + ".BMP"]:
                src = hands_dir / candidate
                if src.exists():
                    dest = model_path / candidate
                    if not dest.exists() or src.read_bytes() != dest.read_bytes():
                        shutil.copy2(str(src), str(dest))
                        tex_copied.append(candidate)
                    break

        # 3. Update QC
        qc_files = list(model_path.glob("*.qc"))
        if not qc_files:
            return (
                f"Hands SMD written: {dest_smd_path.name}\n"
                f"Textures copied: {len(tex_copied)}\n"
                "No QC file found in model directory."
            )

        qc_path = qc_files[0]
        qc      = QC.from_file(str(qc_path))

        _HAND_RE = re.compile(r'^hands?$', re.IGNORECASE)
        old_bgs  = [bg for bg in qc.bodygroups if _HAND_RE.match(bg.name)]

        deleted_smds: list[str] = []
        deleted_texs: list[str] = []

        for bg in old_bgs:
            for entry in bg.entries:
                if not entry.smd:
                    continue
                old_smd_path = model_path / f"{entry.smd}.smd"
                if old_smd_path.exists():
                    # Collect textures before deleting
                    try:
                        old_smd = SMD.from_file(str(old_smd_path))
                        for tri in old_smd.triangles:
                            tex_stem = Path(tri.material).stem
                            for fname in [tri.material, tex_stem + ".bmp", tex_stem + ".BMP"]:
                                tex_p = model_path / Path(fname).name
                                if tex_p.exists():
                                    tex_p.unlink()
                                    deleted_texs.append(tex_p.name)
                    except Exception:
                        pass
                    old_smd_path.unlink()
                    deleted_smds.append(old_smd_path.name)

        # Remove old hand bodygroups, append new one
        qc.bodygroups = [bg for bg in qc.bodygroups if not _HAND_RE.match(bg.name)]
        qc.bodygroups.append(BodyGroup(
            name="hands",
            entries=[BodyGroupEntry(smd=hands_stem)],
        ))

        # Update $attachment and $hbox bone names: weapon bone → hand bone
        att_renamed: list[str] = []
        hb_renamed:  list[str] = []
        for att in qc.attachments:
            if att.bone in self._bone_map:
                new_name  = self._bone_map[att.bone]
                att_renamed.append(f"{att.bone} → {new_name}")
                att.bone  = new_name
        for hb in qc.hboxes:
            if hb.bone in self._bone_map:
                new_name = self._bone_map[hb.bone]
                hb_renamed.append(f"{hb.bone} → {new_name}")
                hb.bone  = new_name

        qc.save(str(qc_path))

        lines = [
            f"Hands SMD written: {dest_smd_path.name}",
            f"Textures copied:   {', '.join(tex_copied) if tex_copied else 'none'}",
            f"QC updated ({qc_path.name}):",
            f"  Removed bodygroup(s): {[bg.name for bg in old_bgs] or 'none'}",
            f"  Deleted SMDs:  {deleted_smds or 'none'}",
            f"  Deleted textures: {deleted_texs or 'none'}",
            f"  Added bodygroup 'hands' → {hands_stem}.smd",
            f"  $attachment bones renamed: {att_renamed or 'none'}",
            f"  $hbox bones renamed: {hb_renamed or 'none'}",
        ]
        return "\n".join(lines)
