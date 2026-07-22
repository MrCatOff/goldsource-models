import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from goldsource.optimise import _mat4_from_bt  # noqa: E402
from goldsource.smd import SMD, Node, BoneTransform, SkeletonFrame, Vertex, Triangle  # noqa: E402


PISTOLS = PROJECT_ROOT / "storage" / "decompiled" / "pistols"
DEFAULT_HAND = PROJECT_ROOT / "storage" / "hands" / "default_hand.smd"


@pytest.fixture(scope="session")
def pistols_dir() -> Path:
    if not PISTOLS.is_dir():
        pytest.skip(f"sample models not present at {PISTOLS}")
    return PISTOLS


@pytest.fixture(scope="session")
def default_hand_path() -> Path:
    if not DEFAULT_HAND.is_file():
        pytest.skip(f"reference hand not present at {DEFAULT_HAND}")
    return DEFAULT_HAND


def world_at(smd: SMD, frame_index: int = 0) -> dict[str, "object"]:
    """World-space matrix per bone name — the reference implementation used by tests."""
    frame = smd.skeleton[frame_index]
    local = {b.bone_id: _mat4_from_bt(b) for b in frame.bones}
    by_id = {n.id: n for n in smd.nodes}
    cache: dict[int, object] = {}

    def resolve(bone_id):
        if bone_id in cache:
            return cache[bone_id]
        node, matrix = by_id.get(bone_id), local.get(bone_id)
        if node is None or matrix is None:
            return None
        if node.parent_id != -1:
            parent = resolve(node.parent_id)
            if parent is not None:
                matrix = parent @ matrix
        cache[bone_id] = matrix
        return matrix

    return {
        node.name: matrix
        for node in smd.nodes
        if (matrix := resolve(node.id)) is not None
    }


def make_chain_smd(frames: int = 3) -> SMD:
    """A→B→C chain plus a second child D of B, with non-trivial transforms."""
    smd = SMD(
        version=1,
        nodes=[
            Node(id=0, name="A", parent_id=-1),
            Node(id=1, name="B", parent_id=0),
            Node(id=2, name="C", parent_id=1),
            Node(id=3, name="D", parent_id=1),
        ],
    )
    for t in range(frames):
        smd.skeleton.append(SkeletonFrame(time=t, bones=[
            BoneTransform(0, 1.0 + t, 2.0, 3.0, 0.1, 0.2 * t, 0.3),
            BoneTransform(1, 4.0, 5.0 - t, 6.0, 0.4 * t, 0.5, 0.6),
            BoneTransform(2, 7.0, 8.0, 9.0 + t, 0.7, 0.8, 0.9 * t),
            BoneTransform(3, -1.0, -2.0, -3.0, 0.2, -0.3, 0.4),
        ]))
    # Geometry only on C, so A/B/D are prunable.
    vertex = Vertex(bone_id=2, x=1.0, y=1.0, z=1.0, nx=0.0, ny=0.0, nz=1.0, u=0.5, v=0.5)
    smd.triangles.append(Triangle(material="tex.bmp", v0=vertex, v1=vertex, v2=vertex))
    return smd
