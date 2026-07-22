"""Skeleton surgery must never move a surviving bone."""

import math

import numpy as np
import pytest

from conftest import make_chain_smd, world_at

from goldsource.qc import QC, Attachment, HitBox
from goldsource.skeleton import (
    compute_keep_set,
    graft_ancestors,
    remove_bones,
    renumber,
    topo_order,
    qc_referenced_bones,
)
from goldsource.smd import SMD, Node, BoneTransform, SkeletonFrame, Triangle, Vertex


def test_remove_bone_preserves_world_pose_in_every_frame():
    smd = make_chain_smd(frames=4)
    before = [world_at(smd, i) for i in range(len(smd.skeleton))]

    removed = remove_bones(smd, {"B"})
    assert removed == ["B"]

    after = [world_at(smd, i) for i in range(len(smd.skeleton))]
    for frame_before, frame_after in zip(before, after):
        for name in ("A", "C", "D"):
            assert np.allclose(frame_before[name], frame_after[name], atol=1e-9), name


def test_removing_a_root_promotes_children_and_keeps_pose():
    """Stripping a redundant top-level bone is what lets models share a skeleton."""
    smd = make_chain_smd()
    before = world_at(smd)

    remove_bones(smd, {"A"})

    assert {n.name for n in smd.nodes} == {"B", "C", "D"}
    assert next(n for n in smd.nodes if n.name == "B").parent_id == -1
    after = world_at(smd)
    for name in ("B", "C", "D"):
        assert np.allclose(before[name], after[name], atol=1e-9), name


def test_remove_bones_refuses_bones_carrying_geometry():
    smd = make_chain_smd()
    assert remove_bones(smd, {"C"}) == []
    assert any(n.name == "C" for n in smd.nodes)


def test_renumber_is_contiguous_and_parents_precede_children():
    smd = make_chain_smd()
    remove_bones(smd, {"B"})
    renumber(smd)

    ids = [n.id for n in smd.nodes]
    assert ids == list(range(len(ids)))
    for node in smd.nodes:
        if node.parent_id != -1:
            assert node.parent_id < node.id

    frame_ids = {b.bone_id for b in smd.skeleton[0].bones}
    assert frame_ids == set(ids)
    for tri in smd.triangles:
        for vertex in tri.vertices:
            assert vertex.bone_id in set(ids)


def test_renumber_keeps_vertex_bindings_pointing_at_the_same_bone():
    smd = make_chain_smd()
    bound_to = {n.id: n.name for n in smd.nodes}[smd.triangles[0].v0.bone_id]
    remove_bones(smd, {"A", "B"})
    renumber(smd)
    still_bound_to = {n.id: n.name for n in smd.nodes}[smd.triangles[0].v0.bone_id]
    assert still_bound_to == bound_to


def test_keep_set_excludes_ancestors_but_keeps_attachments():
    smd = make_chain_smd()
    qc = QC(
        attachments=[Attachment(id=0, bone="D", x=0, y=0, z=0)],
        hboxes=[HitBox(group=0, bone="A", x1=0, y1=0, z1=0, x2=1, y2=1, z2=1)],
    )

    keep = compute_keep_set([smd], qc)
    assert "C" in keep      # carries geometry
    assert "D" in keep      # named by an attachment
    assert "A" not in keep  # hitbox-only: prunable by default
    assert "B" not in keep  # pass-through ancestor

    keep_all = compute_keep_set([smd], qc, keep_ancestors=True, keep_hitbox_bones=True)
    assert {"A", "B", "C", "D"} <= keep_all


def test_hitboxes_do_not_pin_bones_by_default():
    qc = QC(hboxes=[HitBox(group=0, bone="root", x1=0, y1=0, z1=0, x2=1, y2=1, z2=1)])
    assert "root" not in qc_referenced_bones(qc, include_hitboxes=False)
    assert "root" in qc_referenced_bones(qc, include_hitboxes=True)


def test_graft_ancestors_restores_parentage_without_moving_the_mesh():
    """A substituted mesh must adopt the weapon mesh's hierarchy in place."""
    authority = make_chain_smd()

    # A partial mesh that only knows about B and C, with B as its root.
    partial = SMD(
        version=1,
        nodes=[Node(id=0, name="B", parent_id=-1), Node(id=1, name="C", parent_id=0)],
        skeleton=[SkeletonFrame(time=0, bones=[
            BoneTransform(0, 4.0, 5.0, 6.0, 0.0, 0.5, 0.6),
            BoneTransform(1, 7.0, 8.0, 9.0, 0.7, 0.8, 0.0),
        ])],
    )
    before = world_at(partial)

    grafted = graft_ancestors(partial, authority)

    assert grafted == ["A"]
    parent_of_b = next(n.parent_id for n in partial.nodes if n.name == "B")
    assert next(n.name for n in partial.nodes if n.id == parent_of_b) == "A"

    after = world_at(partial)
    for name in ("B", "C"):
        assert np.allclose(before[name], after[name], atol=1e-9), name


def test_folding_a_long_chain_through_gimbal_lock_stays_accurate():
    """
    Composing bones can land on gimbal lock even when no single bone is near it.
    The Euler extraction must not discard the Z rotation there: the error is of
    order ``cy`` and every joint further down the chain multiplies it by its
    lever arm, so a shoulder-sized slip shows up as visible drift at the tip.
    """
    # A chain whose composed Y rotation passes through 90 degrees.
    smd = SMD(nodes=[Node(id=0, name="b0", parent_id=-1)])
    frame = SkeletonFrame(time=0, bones=[BoneTransform(0, 0.0, 0.0, 0.0, 0.0, math.pi / 2, 0.0)])
    for i in range(1, 8):
        smd.nodes.append(Node(id=i, name=f"b{i}", parent_id=i - 1))
        frame.bones.append(BoneTransform(i, 4.0, 0.5, -0.25, 0.3, 0.0, 0.2))
    smd.skeleton.append(frame)

    # Bind geometry to the tip so only the intermediates are prunable.
    tip = Vertex(bone_id=7, x=1.0, y=2.0, z=3.0, nx=0.0, ny=0.0, nz=1.0, u=0.0, v=0.0)
    smd.triangles.append(Triangle(material="t.bmp", v0=tip, v1=tip, v2=tip))

    before = world_at(smd)
    remove_bones(smd, {f"b{i}" for i in range(1, 7)})
    after = world_at(smd)

    drift = float(np.abs(before["b7"] - after["b7"]).max())
    assert drift < 1e-9, f"tip drifted by {drift}"


def test_graft_ancestors_fills_every_animation_frame():
    """
    A grafted bone must exist in all frames — writing only the first would
    leave later frames missing a bone the hierarchy hangs off.
    """
    authority = make_chain_smd()

    partial = SMD(
        version=1,
        nodes=[Node(id=0, name="B", parent_id=-1), Node(id=1, name="C", parent_id=0)],
    )
    for t in range(4):
        partial.skeleton.append(SkeletonFrame(time=t, bones=[
            BoneTransform(0, 4.0 + t, 5.0, 6.0, 0.0, 0.5, 0.6),
            BoneTransform(1, 7.0, 8.0 - t, 9.0, 0.7, 0.8, 0.0),
        ]))
    before = [world_at(partial, i) for i in range(len(partial.skeleton))]

    grafted = graft_ancestors(partial, authority)
    assert grafted == ["A"]

    bone_count = len(partial.nodes)
    for frame in partial.skeleton:
        assert len(frame.bones) == bone_count
        assert {b.bone_id for b in frame.bones} == {n.id for n in partial.nodes}

    after = [world_at(partial, i) for i in range(len(partial.skeleton))]
    for index, (was, now) in enumerate(zip(before, after)):
        for name in ("B", "C"):
            assert np.allclose(was[name], now[name], atol=1e-9), f"frame {index} {name}"


def test_graft_ancestors_is_a_noop_when_hierarchies_agree():
    authority = make_chain_smd()
    same = make_chain_smd()
    assert graft_ancestors(same, authority) == []


def test_topo_order_handles_out_of_order_nodes():
    smd = SMD(nodes=[
        Node(id=0, name="child", parent_id=1),
        Node(id=1, name="root", parent_id=-1),
    ])
    assert topo_order(smd) == [1, 0]
