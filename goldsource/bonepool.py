"""
Bone pooling — let unrelated weapons share the same bone slots.

Merging N models normally costs the *sum* of their weapon bones, which is what
puts the 128-bone limit only a handful of models away.  But only one weapon is
ever drawn at a time, and every sequence belongs to exactly one weapon, so a
bone slot can serve a different weapon in every sequence.  ``v_model4_30`` — a
hand-built 23-weapon model — does exactly this: its weapon bones are a numbered
pool (``Bone_WPNJ1_TYPE1`` …) that 23 weapons draw from, 70 of the 104 slots
shared by more than one of them.

Two things make it safe:

*   **Bind poses are per-mesh.**  studiomdl converts each reference SMD's
    vertices to bone-local space using *that SMD's own* skeleton block, so two
    weapons may sit on one slot with completely different rest poses.
*   **Re-parenting is exact.**  A bone's animation is stored parent-relative,
    so moving it under a different parent only means re-solving
    ``local = parent_world⁻¹ @ world`` for every frame.  World transforms — and
    therefore the geometry — are untouched.  :func:`reparent` does that.

The cost is animation *size*, not accuracy.  A bone re-parented away from the
joint it used to follow no longer inherits that joint's motion, so channels
that were constant become time-varying and studiomdl's run-length encoding
gets less out of them (the same trade-off ``--keep-animated-bones`` exists
for).  :func:`plan_pool` therefore reuses a structurally matching slot when it
can and only reshapes when the alternative is allocating another bone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from goldsource.optimise import _bt_from_mat4, _mat4_from_bt
from goldsource.smd import SMD
from goldsource.skeleton import renumber, world_transforms


#: Prefix for generated pool bone names, matching ``v_model4_30``'s convention.
POOL_PREFIX = "Bone_WPNJ"
POOL_SUFFIX = "_TYPE1"


# ---------------------------------------------------------------------------
# Exact re-parenting
# ---------------------------------------------------------------------------

def reparent(smd: SMD, new_parents: dict[str, str | None]) -> list[str]:
    """
    Move bones onto new parents in *smd*, keeping every world transform.

    *new_parents* maps a bone name to its new parent's name, or ``None`` to make
    it a root.  Bones it does not mention keep the parent they have.  Each
    frame's local transforms are re-solved against the new hierarchy, so all
    animations come out unchanged.

    Returns the names actually moved.
    """
    if not new_parents or not smd.nodes:
        return []

    by_name = {node.name: node for node in smd.nodes}
    moved: list[str] = []
    targets: dict[int, int] = {}
    for name, parent_name in new_parents.items():
        node = by_name.get(name)
        if node is None:
            continue
        if parent_name is None:
            parent_id = -1
        elif parent_name in by_name:
            parent_id = by_name[parent_name].id
        else:
            continue  # the new parent is not in this SMD; leave the bone alone
        if parent_id == node.parent_id:
            continue
        targets[node.id] = parent_id
        moved.append(name)

    if not targets:
        return []

    # World poses under the *old* hierarchy, captured before anything moves.
    world = [world_transforms(smd, index) for index in range(len(smd.skeleton))]

    by_id = {node.id: node for node in smd.nodes}
    for bone_id, parent_id in targets.items():
        by_id[bone_id].parent_id = parent_id

    # A bone the pool did not move may still sit *under* one that did — a hand
    # bone left parented to a weapon joint, say — and re-parenting that joint
    # onto the hand then closes a loop.  Rooting the moved bone breaks it; the
    # frames below are solved from the world poses either way, so nothing shifts.
    for bone_id in targets:
        cursor = by_id[bone_id].parent_id
        seen = {bone_id}
        while cursor != -1 and cursor not in seen:
            seen.add(cursor)
            cursor = by_id[cursor].parent_id
        if cursor != -1:
            by_id[bone_id].parent_id = -1

    identity = np.eye(4)
    id_to_name = {node.id: node.name for node in smd.nodes}

    for index, frame in enumerate(smd.skeleton):
        poses = world[index]
        for bone in frame.bones:
            if bone.bone_id not in targets:
                continue
            name = id_to_name.get(bone.bone_id)
            if name is None or name not in poses:
                continue
            parent_id = by_id[bone.bone_id].parent_id
            parent_name = id_to_name.get(parent_id) if parent_id != -1 else None
            parent_world = identity if parent_name is None else poses.get(parent_name)
            if parent_world is None:
                # The new parent has no transform in this frame; the bone would
                # be resolved against an unknown pose, so leave it as it is.
                continue
            local = np.linalg.inv(parent_world) @ poses[name]
            solved = _bt_from_mat4(bone.bone_id, local)
            bone.tx, bone.ty, bone.tz = solved.tx, solved.ty, solved.tz
            bone.rx, bone.ry, bone.rz = solved.rx, solved.ry, solved.rz

    renumber(smd)
    return moved


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class PoolPlan:
    """Where every model's weapon bones land in the shared pool."""

    #: pool bone name -> parent name (a shared bone, or another pool bone)
    parents: dict[str, str | None] = field(default_factory=dict)
    #: model name -> {model bone name: pool bone name}
    assignments: dict[str, dict[str, str]] = field(default_factory=dict)
    #: model name -> how many of its bones landed somewhere structurally new
    reshaped: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.parents)


def _subtree_sizes(parents: dict[str, str | None], roots) -> dict[str, int]:
    children: dict[str | None, list[str]] = {}
    for name, parent in parents.items():
        children.setdefault(parent, []).append(name)

    sizes: dict[str, int] = {}

    def walk(name: str) -> int:
        if name in sizes:
            return sizes[name]
        sizes[name] = 1 + sum(walk(child) for child in children.get(name, ()))
        return sizes[name]

    for root in roots:
        walk(root)
    return sizes


def plan_pool(
    forests: dict[str, dict[str, str | None]],
    shared: set[str],
    anchor: str,
    max_slots: int | None = None,
    prefix: str = POOL_PREFIX,
    suffix: str = POOL_SUFFIX,
) -> PoolPlan:
    """
    Work out a single bone pool that every model's weapon skeleton fits into.

    *forests* maps a model name to its full ``{bone: parent}`` hierarchy;
    *shared* names the bones every model already has in common (the normalised
    hand), which stay exactly where they are.  Everything else is a weapon bone
    and gets a pool slot, re-anchored under *anchor*.

    Slots are handed out largest-model-first so the biggest weapon defines the
    pool's shape and later models reuse it.  For each bone the search prefers,
    in order: a free slot in the ideal position (under the slot its own parent
    got), a free slot further up that chain, and finally any free slot at all —
    each step trading a little animation compressibility for a bone.  Only when
    the pool is fully claimed does it grow.

    The result therefore never exceeds the *largest single model's* weapon bone
    count, however many models are merged.
    """
    plan = PoolPlan()
    pool_children: dict[str | None, list[str]] = {}
    counter = 0

    def weapon_bones(hierarchy: dict[str, str | None]) -> set[str]:
        return {bone for bone in hierarchy if bone not in shared}

    ordered = sorted(forests.items(), key=lambda item: -len(weapon_bones(item[1])))

    for model_name, hierarchy in ordered:
        bones = weapon_bones(hierarchy)
        if not bones:
            plan.assignments[model_name] = {}
            plan.reshaped[model_name] = 0
            continue

        children: dict[str | None, list[str]] = {}
        for bone in bones:
            children.setdefault(hierarchy[bone], []).append(bone)

        local_parents = {bone: (hierarchy[bone] if hierarchy[bone] in bones else None)
                         for bone in bones}
        roots = [bone for bone in bones if local_parents[bone] is None]
        sizes = _subtree_sizes(local_parents, roots)

        # Widest subtree first, so the bones with the most structure below them
        # claim the slots that have the most structure below *them*.
        order: list[str] = []
        queue = sorted(roots, key=lambda b: -sizes[b])
        while queue:
            bone = queue.pop(0)
            order.append(bone)
            queue = sorted(children.get(bone, []), key=lambda b: -sizes[b]) + queue

        pool_sizes = _subtree_sizes(plan.parents, list(plan.parents))
        assigned: dict[str, str] = {}
        claimed: set[str] = set()
        reshaped = 0

        for bone in order:
            parent = hierarchy[bone]
            ideal = anchor if (parent is None or parent in shared) else assigned[parent]

            slot = None
            free = [c for c in pool_children.get(ideal, []) if c not in claimed]
            if free:
                slot = max(free, key=lambda c: pool_sizes.get(c, 1))

            # A slot in the wrong place still works, but the bone stops
            # inheriting the motion of the joint it used to follow, so its
            # animation channels stop being constant and studiomdl's run-length
            # encoding gets less out of them — which is what pushes a sequence
            # past the 64 KB cap.  While the bone budget allows it, spending a
            # slot is the cheaper trade; only once it is gone does the search
            # widen to slots that do not fit the bone's own shape.
            out_of_budget = max_slots is not None and len(plan.parents) >= max_slots
            if slot is None and out_of_budget:
                node: str | None = plan.parents.get(ideal)
                while node is not None:
                    candidates = [c for c in pool_children.get(node, []) if c not in claimed]
                    if candidates:
                        slot = max(candidates, key=lambda c: pool_sizes.get(c, 1))
                        reshaped += 1
                        break
                    node = plan.parents.get(node)

                if slot is None:
                    # Any free slot will do, but only if this model already
                    # holds every pool bone above it.  Take one it does not and
                    # the slot has to fall back to some *other* ancestor for
                    # this model — so the merged model sees one bone name with
                    # two different parents and renames them apart, undoing the
                    # sharing for every model involved.
                    spare = [
                        name for name in plan.parents
                        if name not in claimed and _ancestors_claimed(plan.parents, name, claimed)
                    ]
                    if spare:
                        slot = min(spare, key=lambda c: _depth(plan.parents, c))
                        reshaped += 1

            if slot is None:
                counter += 1
                slot = f"{prefix}{counter}{suffix}"
                plan.parents[slot] = ideal
                pool_children.setdefault(ideal, []).append(slot)
                pool_sizes[slot] = 1

            assigned[bone] = slot
            claimed.add(slot)

        plan.assignments[model_name] = assigned
        plan.reshaped[model_name] = reshaped

    return plan


def _ancestors_claimed(
    parents: dict[str, str | None], name: str, claimed: set[str],
) -> bool:
    """True when every pool bone above *name* is already claimed."""
    cursor = parents.get(name)
    while cursor is not None and cursor in parents:
        if cursor not in claimed:
            return False
        cursor = parents.get(cursor)
    return True


def _depth(parents: dict[str, str | None], name: str) -> int:
    depth = 0
    seen = set()
    while name in parents and parents[name] is not None and name not in seen:
        seen.add(name)
        name = parents[name]  # type: ignore[assignment]
        depth += 1
    return depth


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_pool(
    smds: dict[str, SMD],
    assignment: dict[str, str],
    pool_parents: dict[str, str | None],
) -> int:
    """
    Rename one model's weapon bones onto their pool slots and re-parent them to
    match the pool's hierarchy, across every SMD (reference meshes *and*
    animations).  Returns the number of bones moved.

    Renaming happens first and in one pass, so a model bone whose name happens
    to equal some other model's pool slot cannot collide with it.
    """
    if not assignment:
        return 0

    moved = 0
    for smd in smds.values():
        present = {node.name for node in smd.nodes}
        mapping = {bone: slot for bone, slot in assignment.items() if bone in present}
        if not mapping:
            continue

        _rename_atomically(smd, mapping)

        renamed = {node.name for node in smd.nodes}
        new_parents: dict[str, str | None] = {}
        for slot in mapping.values():
            # Walk up the pool until a bone this mesh actually has.  Leaving the
            # bone on its old parent instead would be worse than wrong: that
            # parent may itself have been renamed to one of this slot's own pool
            # descendants, closing a cycle in the hierarchy.
            parent = pool_parents.get(slot)
            while parent is not None and parent not in renamed:
                parent = pool_parents.get(parent)
            new_parents[slot] = parent
        moved += len(reparent(smd, new_parents))

    return moved


def _rename_atomically(smd: SMD, mapping: dict[str, str]) -> None:
    """
    Apply *mapping* to bone names without letting a target name collide with a
    source name that has not been renamed yet.
    """
    by_name = {node.name: node for node in smd.nodes}
    staged: dict[int, str] = {}
    for old, new in mapping.items():
        node = by_name.get(old)
        if node is not None:
            staged[node.id] = new
    for node in smd.nodes:
        if node.id in staged:
            node.name = staged[node.id]
