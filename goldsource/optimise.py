"""
Bone optimisation for GoldSource SMD files.

Two passes are applied per-SMD:
  1. Dead-leaf removal  — a bone with no mesh vertices and no children can be
     deleted outright (it contributes nothing to the model).
  2. Pass-through collapse — a bone with no mesh vertices and exactly one child
     can be folded into that child.  The child's local transform is replaced by
     B_local @ C_local so that the child's world-space position/orientation is
     unchanged for every animation frame.

Both operations are performed on *all* SMD files in the directory
(reference + every animation SMD found in sub-directories).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from goldsource.smd import SMD, Node, BoneTransform


# ---------------------------------------------------------------------------
# Matrix helpers (same convention as viewer.py)
# ---------------------------------------------------------------------------

def _euler_mat4(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [ cy*cz,  cz*sx*sy - cx*sz,  cx*cz*sy + sx*sz,  0.0],
        [ cy*sz,  cx*cz + sx*sy*sz,  cx*sy*sz - cz*sx,  0.0],
        [-sy,     cy*sx,             cx*cy,             0.0],
        [ 0.0,    0.0,               0.0,               1.0],
    ], dtype=np.float64)


# Below this the Y rotation is at gimbal lock: R[2,1] and R[2,2] are pure
# rounding noise, so X and Z can no longer be separated and Z is folded into X.
#
# Keep it as small as possible.  ``atan2`` is scale-invariant and stays accurate
# for tiny-but-real arguments, whereas the degenerate branch *discards* the Z
# rotation — an error of order ``cy``.  A threshold of 1e-6 therefore throws
# away up to 1e-6 rad on orientations that were perfectly representable, and
# folding a long bone chain amplifies that by each joint's lever arm (a 3e-7
# error at the shoulder becomes 1e-5 units at the fingertips).
_GIMBAL_EPSILON = 1e-13


def _extract_euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    sy = -R[2, 0]
    cy = math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)
    if cy > _GIMBAL_EPSILON:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(sy, cy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(sy, cy)
        rz = 0.0
    return rx, ry, rz


def _mat4_from_bt(bt: BoneTransform) -> np.ndarray:
    m = _euler_mat4(bt.rx, bt.ry, bt.rz)
    m[0, 3], m[1, 3], m[2, 3] = bt.tx, bt.ty, bt.tz
    return m


def _bt_from_mat4(bone_id: int, m: np.ndarray) -> BoneTransform:
    rx, ry, rz = _extract_euler_zyx(m[:3, :3])
    return BoneTransform(
        bone_id=bone_id,
        tx=float(m[0, 3]), ty=float(m[1, 3]), tz=float(m[2, 3]),
        rx=rx, ry=ry, rz=rz,
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class OptimisationReport:
    """Candidates found across all SMDs in a directory."""
    # bone_name → set of SMD filenames where it was found as dead leaf
    dead_leaves: dict[str, set[str]] = field(default_factory=dict)
    # bone_name → set of SMD filenames where it was found as collapsible
    collapsible: dict[str, set[str]] = field(default_factory=dict)


def _collect_smd_files(directory: str) -> list[Path]:
    """Return all .smd files under *directory* (recursive)."""
    root = Path(directory)
    return sorted(root.rglob("*.smd"))


def _child_map(smd: SMD) -> dict[int, list[int]]:
    cm: dict[int, list[int]] = {n.id: [] for n in smd.nodes}
    for n in smd.nodes:
        if n.parent_id != -1 and n.parent_id in cm:
            cm[n.parent_id].append(n.id)
    return cm


# Biped hand/finger/forearm bones that must never be removed or collapsed.
_HAND_BONE_RE = re.compile(
    r"^Bip\d*_[LR]_(Forearm|Hand|Finger\d*|Thumb\d*)$",
    re.IGNORECASE,
)


def _is_hand_bone(name: str) -> bool:
    return bool(_HAND_BONE_RE.match(name))


def _vertex_bone_ids(smd: SMD) -> set[int]:
    ids: set[int] = set()
    for tri in smd.triangles:
        for v in tri.vertices:
            ids.add(v.bone_id)
    return ids


def analyse_directory(directory: str) -> OptimisationReport:
    """
    Scan all SMD files under *directory* and return an OptimisationReport
    listing dead-leaf and collapsible bones.

    Reference SMDs (those that contain triangles/textures) are used to
    determine which bones carry mesh vertices.  Animation SMDs share the
    same bone hierarchy but have no triangles, so their vertex ownership
    is inherited from the reference SMDs rather than evaluated independently.
    """
    report = OptimisationReport()
    smd_files = _collect_smd_files(directory)

    # ── Pass 1: load all SMDs, separate ref from anim ────────────────────
    loaded: list[tuple[Path, SMD]] = []
    for path in smd_files:
        try:
            smd = SMD.from_file(path)
            loaded.append((path, smd))
        except Exception:
            continue

    # Collect bone *names* that carry vertices in any reference SMD.
    # Animation SMDs have no triangles and must not override this.
    vertex_bone_names: set[str] = set()
    for _path, smd in loaded:
        if smd.triangles:  # reference SMD
            for bid in _vertex_bone_ids(smd):
                node = smd.node_by_id(bid)
                if node is not None:
                    vertex_bone_names.add(node.name)

    # ── Pass 2: find candidates in every SMD ─────────────────────────────
    for path, smd in loaded:
        fname = path.name
        cm = _child_map(smd)

        for node in smd.nodes:
            children = cm.get(node.id, [])
            if node.name in vertex_bone_names:
                continue  # carries geometry in the reference SMD
            if _is_hand_bone(node.name):
                continue  # hand/finger bones are preserved regardless

            if not children:
                report.dead_leaves.setdefault(node.name, set()).add(fname)
            elif len(children) == 1:
                report.collapsible.setdefault(node.name, set()).add(fname)

    return report


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _remove_dead_leaf(smd: SMD, bone_name: str) -> bool:
    """
    Remove a dead-leaf bone from *smd* in-place.
    Returns True if the bone was found and removed.
    """
    node = smd.node_by_name(bone_name)
    if node is None:
        return False

    bid = node.id
    cm = _child_map(smd)
    has_verts = _vertex_bone_ids(smd)

    if cm.get(bid):
        return False  # has children — not a leaf
    if bid in has_verts:
        return False  # carries geometry

    # Remove node
    smd.nodes = [n for n in smd.nodes if n.id != bid]

    # Remove from skeleton frames
    for frame in smd.skeleton:
        frame.bones = [b for b in frame.bones if b.bone_id != bid]

    return True


def _collapse_bone(smd: SMD, bone_name: str) -> bool:
    """
    Collapse a pass-through bone B into its single child C.

    For every frame:
        new_C_local = B_local @ C_local

    B is then removed and C's parent_id is set to B's parent_id.
    Returns True if successfully collapsed.
    """
    node_b = smd.node_by_name(bone_name)
    if node_b is None:
        return False

    bid = node_b.id
    cm = _child_map(smd)
    has_verts = _vertex_bone_ids(smd)

    children = cm.get(bid, [])
    if len(children) != 1:
        return False
    if bid in has_verts:
        return False

    cid = children[0]  # the single child

    # Reparent child
    for n in smd.nodes:
        if n.id == cid:
            n.parent_id = node_b.parent_id
            break

    # Fold transform per frame
    for frame in smd.skeleton:
        bt_map = {b.bone_id: b for b in frame.bones}
        bt_b = bt_map.get(bid)
        bt_c = bt_map.get(cid)
        if bt_b is None or bt_c is None:
            continue

        M_b = _mat4_from_bt(bt_b)
        M_c = _mat4_from_bt(bt_c)
        M_new = M_b @ M_c

        new_bt = _bt_from_mat4(cid, M_new)
        # Replace bt_c in-place
        for i, b in enumerate(frame.bones):
            if b.bone_id == cid:
                frame.bones[i] = new_bt
                break

    # Remove bone B from nodes and all skeleton frames
    smd.nodes = [n for n in smd.nodes if n.id != bid]
    for frame in smd.skeleton:
        frame.bones = [b for b in frame.bones if b.bone_id != bid]

    return True


def apply_optimisations(
    directory: str,
    dead_leaves: list[str],
    to_collapse: list[str],
) -> tuple[int, list[str]]:
    """
    Apply selected optimisations to all SMD files under *directory*.

    Returns (files_modified, error_list).
    """
    smd_files = _collect_smd_files(directory)
    modified = 0
    errors: list[str] = []

    for path in smd_files:
        try:
            smd = SMD.from_file(path)
        except Exception as exc:
            errors.append(f"{path.name}: load error — {exc}")
            continue

        changed = False

        # Collapse pass-through bones first (dead leaves may be created by this)
        for name in to_collapse:
            if _collapse_bone(smd, name):
                changed = True

        # Remove dead leaves
        for name in dead_leaves:
            if _remove_dead_leaf(smd, name):
                changed = True

        if changed:
            try:
                smd.save(path)
                modified += 1
            except Exception as exc:
                errors.append(f"{path.name}: save error — {exc}")

    return modified, errors
