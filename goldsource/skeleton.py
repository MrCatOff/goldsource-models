"""
Skeleton surgery primitives shared by the hand-normalisation and optimisation
passes.

Everything here operates on a single :class:`~goldsource.smd.SMD` and preserves
the *world-space* pose of every surviving bone in every frame.  The two core
operations are:

``remove_bones``
    Delete bones and fold their local transform into their children, so a
    child's world transform is unchanged.  Works for bones with any number of
    children (including roots), which is what makes it usable for collapsing
    redundant top-level bones such as ``root`` / ``Bone_Root``.

``renumber``
    Rewrite bone ids so they are contiguous and every parent has a lower id
    than its children — a hard requirement of studiomdl.

A bone is never removed while it still carries mesh vertices; callers compute a
*keep set* first (see :func:`compute_keep_set`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from goldsource.qc import QC
from goldsource.smd import SMD, Node, BoneTransform
from goldsource.optimise import _mat4_from_bt, _bt_from_mat4


# ---------------------------------------------------------------------------
# Hierarchy helpers
# ---------------------------------------------------------------------------

def child_ids(smd: SMD) -> dict[int, list[int]]:
    """Return ``{bone_id: [child_id, ...]}`` in node order."""
    result: dict[int, list[int]] = {n.id: [] for n in smd.nodes}
    for n in smd.nodes:
        if n.parent_id != -1 and n.parent_id in result:
            result[n.parent_id].append(n.id)
    return result


def topo_order(smd: SMD) -> list[int]:
    """
    Bone ids ordered parents-before-children, preserving the original relative
    order of siblings.  Orphans (parent id not present) are treated as roots.
    """
    kids = child_ids(smd)
    known = {n.id for n in smd.nodes}
    order: list[int] = []
    visited: set[int] = set()

    def visit(bid: int) -> None:
        if bid in visited:
            return
        visited.add(bid)
        order.append(bid)
        for cid in kids.get(bid, []):
            visit(cid)

    for n in smd.nodes:
        if n.parent_id == -1 or n.parent_id not in known:
            visit(n.id)
    # Anything left is part of a cycle — append defensively so bones aren't lost.
    for n in smd.nodes:
        visit(n.id)
    return order


def vertex_bone_names(smd: SMD) -> set[str]:
    """Names of bones that at least one mesh vertex is bound to."""
    id_to_name = {n.id: n.name for n in smd.nodes}
    names: set[str] = set()
    for tri in smd.triangles:
        for v in tri.vertices:
            name = id_to_name.get(v.bone_id)
            if name is not None:
                names.add(name)
    return names


def ancestors_of(smd: SMD, names: set[str]) -> set[str]:
    """All ancestors (transitively) of *names* within *smd*, excluding *names*."""
    by_name = {n.name: n for n in smd.nodes}
    by_id = {n.id: n for n in smd.nodes}
    result: set[str] = set()
    for name in names:
        node = by_name.get(name)
        if node is None:
            continue
        parent = by_id.get(node.parent_id)
        while parent is not None and parent.name not in result:
            result.add(parent.name)
            parent = by_id.get(parent.parent_id)
    return result - names


def qc_referenced_bones(qc: QC, include_hitboxes: bool = True) -> set[str]:
    """
    Bones named by ``$attachment`` / ``$controller`` / ``$keepbone``, and by
    ``$hbox`` when *include_hitboxes* is set.

    Attachments and controllers are driven by game code and must survive.
    Hitboxes are inert on view models, and pinning a bone for one costs far
    more than it is worth: a single ``$hbox "root"`` keeps a redundant root
    alive, which conflicts with the other models' hierarchy and forces the
    whole shared hand skeleton to be duplicated.  Hitboxes left pointing at a
    removed bone are dropped from the merged QC.
    """
    names: set[str] = set()
    for att in qc.attachments:
        names.add(att.bone)
    for ctrl in qc.controllers:
        names.add(ctrl.bone)
    if include_hitboxes:
        for hbox in qc.hboxes:
            names.add(hbox.bone)
    names.update(qc.keepbones)
    return names


# ---------------------------------------------------------------------------
# Keep-set computation
# ---------------------------------------------------------------------------

def compute_keep_set(
    reference_smds: list[SMD],
    qc: QC | None = None,
    extra_keep: set[str] | None = None,
    keep_ancestors: bool = False,
    keep_hitbox_bones: bool = False,
) -> set[str]:
    """
    Bones that must survive pruning: those carrying mesh vertices in any
    reference SMD, plus anything named by the QC (attachments, hitboxes,
    controllers, ``$keepbone``).

    Ancestors are deliberately *not* kept by default.  studiomdl preserves them
    because it cannot rebuild the hierarchy, but :func:`remove_bones` folds a
    removed bone's transform into its children, so a pass-through ancestor can
    be dropped with no change to any pose.  That matters for merging: models
    from different authors wrap the same hand rig in different top-level bones
    (``root``, ``Bone_Root``, or none at all).  Keeping them would give the same
    bone name two different parents across models, which studiomdl rejects as
    "illegal parent bone replacement" — forcing the merger to rename the bones
    apart and duplicate the entire shared skeleton.

    Pass ``keep_ancestors=True`` to instead mirror studiomdl's own behaviour,
    which is what predicts the compiled bone count of an *unpruned* model.
    """
    keep: set[str] = set(extra_keep or set())

    for smd in reference_smds:
        keep |= vertex_bone_names(smd)

    if qc is not None:
        keep |= qc_referenced_bones(qc, include_hitboxes=keep_hitbox_bones)

    if keep_ancestors:
        for smd in reference_smds:
            keep |= ancestors_of(smd, keep)

    return keep


# ---------------------------------------------------------------------------
# Bone removal (transform-preserving)
# ---------------------------------------------------------------------------

def remove_bones(smd: SMD, remove: set[str]) -> list[str]:
    """
    Remove every bone in *remove* from *smd*, folding its local transform into
    each of its children so their world transforms are unchanged::

        child_local_new = removed_local @ child_local

    Children are re-parented to the removed bone's parent.  Removing a root
    therefore promotes its children to roots, with the root's transform baked
    in — exactly what is needed to strip redundant ``root`` / ``Bone_Root``
    bones so that models from different authors share one hierarchy.

    Bones that still carry mesh vertices are skipped.  Returns the names of the
    bones actually removed.
    """
    if not remove:
        return []

    carries_vertices = vertex_bone_names(smd)
    removed: list[str] = []

    # Parents before children, so a removed chain folds correctly step by step.
    for bid in topo_order(smd):
        node = next((n for n in smd.nodes if n.id == bid), None)
        if node is None or node.name not in remove:
            continue
        if node.name in carries_vertices:
            continue  # geometry is bound to it — must stay

        parent_id = node.parent_id
        children = [n for n in smd.nodes if n.parent_id == bid]

        for frame in smd.skeleton:
            by_id = {b.bone_id: b for b in frame.bones}
            bt_removed = by_id.get(bid)
            if bt_removed is None:
                continue
            m_removed = _mat4_from_bt(bt_removed)
            for child in children:
                bt_child = by_id.get(child.id)
                if bt_child is None:
                    continue
                folded = _bt_from_mat4(child.id, m_removed @ _mat4_from_bt(bt_child))
                idx = frame.bones.index(bt_child)
                frame.bones[idx] = folded

        for child in children:
            child.parent_id = parent_id

        smd.nodes = [n for n in smd.nodes if n.id != bid]
        for frame in smd.skeleton:
            frame.bones = [b for b in frame.bones if b.bone_id != bid]

        removed.append(node.name)

    return removed


def renumber(smd: SMD) -> None:
    """
    Rewrite bone ids in-place so they are contiguous, start at 0, and every
    parent precedes its children.  Node order, skeleton frames and triangle
    vertex bindings are all updated to match.
    """
    order = topo_order(smd)
    remap = {old_id: new_id for new_id, old_id in enumerate(order)}

    by_id = {n.id: n for n in smd.nodes}
    new_nodes: list[Node] = []
    for old_id in order:
        node = by_id[old_id]
        new_nodes.append(Node(
            id=remap[old_id],
            name=node.name,
            parent_id=remap[node.parent_id] if node.parent_id in remap else -1,
        ))
    smd.nodes = new_nodes

    for frame in smd.skeleton:
        rebound: list[BoneTransform] = []
        for bt in frame.bones:
            if bt.bone_id not in remap:
                continue
            bt.bone_id = remap[bt.bone_id]
            rebound.append(bt)
        rebound.sort(key=lambda b: b.bone_id)
        frame.bones = rebound

    for tri in smd.triangles:
        for v in tri.vertices:
            v.bone_id = remap.get(v.bone_id, v.bone_id)


def rename_bones(smd: SMD, mapping: dict[str, str]) -> None:
    """Apply ``{old_name: new_name}`` to every node in *smd*."""
    if not mapping:
        return
    for node in smd.nodes:
        node.name = mapping.get(node.name, node.name)


# ---------------------------------------------------------------------------
# World-space pose
# ---------------------------------------------------------------------------

def world_transforms(smd: SMD, frame_index: int = 0) -> dict[str, np.ndarray]:
    """
    World-space 4×4 matrix per bone name at *frame_index* (the bind pose for a
    reference SMD).  Bones missing a transform in that frame are skipped.
    """
    if not smd.skeleton:
        return {}
    frame = smd.skeleton[frame_index]
    local = {b.bone_id: _mat4_from_bt(b) for b in frame.bones}
    by_id = {n.id: n for n in smd.nodes}

    cache: dict[int, np.ndarray] = {}

    def resolve(bid: int) -> np.ndarray | None:
        if bid in cache:
            return cache[bid]
        node = by_id.get(bid)
        mat = local.get(bid)
        if node is None or mat is None:
            return None
        if node.parent_id != -1:
            parent = resolve(node.parent_id)
            if parent is not None:
                mat = parent @ mat
        cache[bid] = mat
        return mat

    result: dict[str, np.ndarray] = {}
    for node in smd.nodes:
        mat = resolve(node.id)
        if mat is not None:
            result[node.name] = mat
    return result


# ---------------------------------------------------------------------------
# Hierarchy reconciliation
# ---------------------------------------------------------------------------

def graft_ancestors(target: SMD, authority: SMD) -> list[str]:
    """
    Give every root of *target* the same ancestry it has in *authority*.

    studiomdl builds one bone table from all of a model's SMDs and rejects the
    same bone appearing under two different parents ("illegal parent bone
    replacement").  A substituted mesh — a replacement hand, say — only knows
    about its own bones, so if the weapon mesh still parents them under some
    surviving bone, the missing ancestors are copied in from *authority*.

    The grafted bones take their local transforms from *authority*'s bind pose,
    and the re-parented root's own local transform is recomputed so its
    **world-space bind pose is unchanged** — otherwise the mesh would shift,
    because reference SMDs store vertices in world space.

    Returns the names of the bones grafted in.
    """
    if not target.skeleton or not authority.skeleton:
        return []

    authority_parent = {
        node.name: (
            next((n.name for n in authority.nodes if n.id == node.parent_id), None)
            if node.parent_id != -1 else None
        )
        for node in authority.nodes
    }
    authority_local = {
        node.name: bt
        for node in authority.nodes
        for bt in authority.skeleton[0].bones
        if bt.bone_id == node.id
    }

    target_world = world_transforms(target)
    grafted: list[str] = []
    frame = target.skeleton[0]
    next_id = max((n.id for n in target.nodes), default=-1) + 1

    for root in [n for n in target.nodes if n.parent_id == -1]:
        chain: list[str] = []
        parent = authority_parent.get(root.name)
        while parent is not None and not any(n.name == parent for n in target.nodes):
            chain.append(parent)
            parent = authority_parent.get(parent)
        if not chain:
            continue

        # chain is nearest-first; create from the far end so parents come first.
        previous_id = (
            next((n.id for n in target.nodes if n.name == parent), -1)
            if parent is not None else -1
        )
        for name in reversed(chain):
            source = authority_local.get(name)
            if source is None:
                continue
            target.nodes.append(Node(id=next_id, name=name, parent_id=previous_id))
            frame.bones.append(BoneTransform(
                bone_id=next_id,
                tx=source.tx, ty=source.ty, tz=source.tz,
                rx=source.rx, ry=source.ry, rz=source.rz,
            ))
            grafted.append(name)
            previous_id = next_id
            next_id += 1

        if previous_id == -1:
            continue

        root.parent_id = previous_id

        # Preserve the root's world pose under its new parent.
        new_parent_world = world_transforms(target).get(
            next(n.name for n in target.nodes if n.id == previous_id)
        )
        original_world = target_world.get(root.name)
        if new_parent_world is not None and original_world is not None:
            local = np.linalg.inv(new_parent_world) @ original_world
            for index, bt in enumerate(frame.bones):
                if bt.bone_id == root.id:
                    frame.bones[index] = _bt_from_mat4(root.id, local)
                    break

    if grafted:
        renumber(target)
    return grafted


@dataclass
class PruneStats:
    """What a prune pass did to one model."""
    removed: list[str] = field(default_factory=list)
    files_changed: int = 0
