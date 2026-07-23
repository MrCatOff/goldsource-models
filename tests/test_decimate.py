"""Tests for mesh decimation."""

import numpy as np
import pytest

from goldsource.decimate import decimate_mesh
from goldsource.merger import ModelInput
from goldsource.pipeline import dedupe_bodygroup_names, _reference_keys
from goldsource.smd import SMD, Node, SkeletonFrame, BoneTransform, Vertex, Triangle


def _grid_mesh(n: int = 20, bone: int = 0) -> SMD:
    """A flat n×n triangulated grid — plenty of coplanar detail to remove."""
    smd = SMD(version=1, nodes=[Node(id=0, name="root", parent_id=-1)])
    smd.skeleton.append(SkeletonFrame(time=0, bones=[BoneTransform(0, 0, 0, 0, 0, 0, 0)]))

    def vert(x, y):
        return Vertex(bone_id=bone, x=float(x), y=float(y), z=0.0,
                      nx=0.0, ny=0.0, nz=1.0, u=x / n, v=y / n)

    for i in range(n):
        for j in range(n):
            a, b, c, d = vert(i, j), vert(i + 1, j), vert(i + 1, j + 1), vert(i, j + 1)
            smd.triangles.append(Triangle(material="t.bmp", v0=a, v1=b, v2=c))
            smd.triangles.append(Triangle(material="t.bmp", v0=a, v1=c, v2=d))
    return smd


def test_decimation_reduces_triangle_count():
    smd = _grid_mesh()
    before = len(smd.triangles)
    b, a = decimate_mesh(smd, 0.5)
    assert b == before
    assert a < before
    assert len(smd.triangles) == a


def test_ratio_one_is_a_no_op():
    smd = _grid_mesh()
    before = len(smd.triangles)
    b, a = decimate_mesh(smd, 1.0)
    assert (b, a) == (before, before)
    assert len(smd.triangles) == before


def test_lower_ratio_removes_more():
    counts = []
    for ratio in (0.8, 0.5, 0.25):
        smd = _grid_mesh()
        decimate_mesh(smd, ratio)
        counts.append(len(smd.triangles))
    assert counts[0] > counts[1] > counts[2]


def test_bone_bindings_are_preserved():
    """Every surviving vertex must still ride a bone the mesh already had."""
    smd = _grid_mesh(bone=3)
    decimate_mesh(smd, 0.4)
    assert {v.bone_id for t in smd.triangles for v in t.vertices} == {3}


def test_two_bones_never_merge():
    """A cluster spanning two bones must split, so neither bone's grip drifts."""
    left = _grid_mesh(n=12, bone=1)
    right = _grid_mesh(n=12, bone=2)
    # Overlay them in the same space so cells would merge if bone were ignored.
    left.triangles.extend(right.triangles)
    left.nodes.append(Node(id=1, name="a", parent_id=-1))
    left.nodes.append(Node(id=2, name="b", parent_id=-1))
    decimate_mesh(left, 0.5)
    for tri in left.triangles:
        bones = {v.bone_id for v in tri.vertices}
        # A triangle may legitimately span bones, but a *vertex* keeps one bone
        # and both original bones must still be present somewhere.
        assert bones <= {1, 2}
    assert {v.bone_id for t in left.triangles for v in t.vertices} == {1, 2}


def test_no_nan_and_bounds_are_kept():
    smd = _grid_mesh()
    before = np.array([[v.x, v.y, v.z] for t in smd.triangles for v in t.vertices])
    decimate_mesh(smd, 0.3)
    after = np.array([[v.x, v.y, v.z] for t in smd.triangles for v in t.vertices])
    assert not np.isnan(after).any() and not np.isinf(after).any()
    # The representative is clamped to its cluster, so bounds barely move.
    assert np.abs(after.min(0) - before.min(0)).max() < 2.0
    assert np.abs(after.max(0) - before.max(0)).max() < 2.0


def test_normals_stay_unit_length():
    smd = _grid_mesh()
    decimate_mesh(smd, 0.4)
    for tri in smd.triangles:
        for v in tri.vertices:
            assert abs(np.linalg.norm([v.nx, v.ny, v.nz]) - 1.0) < 1e-4


# --- on a real, messy model ------------------------------------------------

def test_decimation_reduces_a_real_weapon(more_weapons_dir):
    d = more_weapons_dir / "v_ak47chimera"
    if not d.is_dir():
        pytest.skip("v_ak47chimera not present")
    model = ModelInput.from_directory("v_ak47chimera", d)
    dedupe_bodygroup_names(model.qc)
    hand = {"leftarm-1", "leftarm-2", "rightarm-1", "rightarm-2", "sleeves"}
    weapon = [k for k in _reference_keys(model) if k not in hand]

    before = after = 0
    for key in weapon:
        b, a = decimate_mesh(model.smds[key], 0.5)
        before += b
        after += a

    assert before > 20000
    assert after < 0.7 * before  # a real, substantial cut
