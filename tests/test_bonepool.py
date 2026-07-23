"""Tests for weapon-bone pooling — the pass that lets models share bone slots."""

import numpy as np
import pytest

from conftest import make_chain_smd, world_at

from goldsource.bonepool import _ancestors_claimed, apply_pool, plan_pool, reparent
from goldsource.hands import load_reference_hand
from goldsource.merger import ModelInput, ModelMerger
from goldsource.pipeline import (
    dedupe_bodygroup_names,
    discover_models,
    normalise_hands,
    pack_always_on_parts,
    pool_bones,
    prune_model,
    select_weapon_groups,
)
from goldsource.smd import SMD


# ---------------------------------------------------------------------------
# reparent
# ---------------------------------------------------------------------------

def test_reparent_preserves_every_world_transform():
    """Moving a bone up the chain must not move it in the world."""
    smd = make_chain_smd(frames=4)
    before = [world_at(smd, i) for i in range(len(smd.skeleton))]

    moved = reparent(smd, {"C": "A", "D": None})

    assert set(moved) == {"C", "D"}
    for index, old in enumerate(before):
        new = world_at(smd, index)
        for bone, matrix in old.items():
            assert np.allclose(matrix, new[bone], atol=1e-12), f"{bone} moved at frame {index}"


def test_reparent_ignores_bones_the_mesh_does_not_have():
    smd = make_chain_smd()
    assert reparent(smd, {"Z": "A"}) == []
    # A parent that is absent leaves the bone where it was, rather than orphaning it.
    assert reparent(smd, {"C": "Z"}) == []
    assert {n.name for n in smd.nodes if n.parent_id != -1} == {"B", "C", "D"}


def test_reparent_breaks_a_cycle_rather_than_hanging():
    """
    A bone that is not being moved may sit under one that is; re-parenting the
    moved bone onto it would close a loop and make world resolution recurse
    forever.  The moved bone is rooted instead.
    """
    smd = make_chain_smd()
    before = [world_at(smd, i) for i in range(len(smd.skeleton))]

    reparent(smd, {"B": "C"})  # C is B's own child

    by_name = {node.name: node for node in smd.nodes}
    assert by_name["B"].parent_id == -1
    for index, old in enumerate(before):
        new = world_at(smd, index)
        for bone, matrix in old.items():
            assert np.allclose(matrix, new[bone], atol=1e-12)


# ---------------------------------------------------------------------------
# plan_pool
# ---------------------------------------------------------------------------

SHARED = {"hand"}


def _forest(*chains: tuple[str, ...]) -> dict[str, str | None]:
    """Build a {bone: parent} forest from chains rooted at the shared bone."""
    forest: dict[str, str | None] = {"hand": None}
    for chain in chains:
        parent = "hand"
        for bone in chain:
            forest[bone] = parent
            parent = bone
    return forest


def test_pool_never_exceeds_the_largest_model():
    """
    However many models are merged, the pool costs what the biggest one needs —
    that is the whole point of sharing slots.
    """
    forests = {
        f"m{i}": _forest(tuple(f"m{i}_b{j}" for j in range(size)))
        for i, size in enumerate([5, 3, 4, 2, 5, 1])
    }
    plan = plan_pool(forests, SHARED, anchor="hand")

    assert plan.size == 5
    for name, forest in forests.items():
        weapon = {b for b in forest if b not in SHARED}
        assert set(plan.assignments[name]) == weapon
        # No two bones of one model may land on the same slot.
        assert len(set(plan.assignments[name].values())) == len(weapon)


def test_every_claimed_slot_has_its_ancestors_claimed_too():
    """
    A model that takes a slot must hold every pool bone above it.  Otherwise
    that slot has to fall back to a different parent for this model, the merged
    model sees one bone name with two parents, and the merger renames them
    apart — undoing the sharing.
    """
    forests = {
        "deep": _forest(("d1", "d2", "d3", "d4")),
        "wide": _forest(("w1",), ("w2",), ("w3",), ("w4",), ("w5",)),
        "mixed": _forest(("x1", "x2"), ("y1",)),
    }
    plan = plan_pool(forests, SHARED, anchor="hand")

    for name, assignment in plan.assignments.items():
        claimed = set(assignment.values())
        for slot in claimed:
            assert _ancestors_claimed(plan.parents, slot, claimed), \
                f"{name} took {slot} without its pool ancestors"


def test_max_slots_caps_growth_and_forces_reuse():
    forests = {
        "a": _forest(tuple(f"a{i}" for i in range(6))),
        "b": _forest(("b1",), ("b2",), ("b3",), ("b4",), ("b5",), ("b6",)),
    }
    generous = plan_pool(forests, SHARED, anchor="hand", max_slots=None)
    tight = plan_pool(forests, SHARED, anchor="hand", max_slots=6)

    assert tight.size == 6
    assert tight.size <= generous.size
    assert sum(tight.reshaped.values()) >= sum(generous.reshaped.values())


def test_pool_is_a_forest_rooted_at_the_anchor():
    forests = {"a": _forest(("a1", "a2")), "b": _forest(("b1",), ("b2",))}
    plan = plan_pool(forests, SHARED, anchor="hand")

    for slot, parent in plan.parents.items():
        assert parent == "hand" or parent in plan.parents
        depth, cursor = 0, parent
        while cursor in plan.parents:
            cursor = plan.parents[cursor]
            depth += 1
            assert depth <= plan.size, f"{slot} sits in a cycle"


# ---------------------------------------------------------------------------
# apply_pool
# ---------------------------------------------------------------------------

def test_apply_pool_renames_and_reparents_without_moving_geometry():
    smd = make_chain_smd(frames=3)
    before = [world_at(smd, i) for i in range(len(smd.skeleton))]

    assignment = {"B": "slot1", "C": "slot2", "D": "slot3"}
    apply_pool({"ref": smd}, assignment, {"slot1": "A", "slot2": "slot1", "slot3": "A"})

    names = {node.name for node in smd.nodes}
    assert names == {"A", "slot1", "slot2", "slot3"}
    for index, old in enumerate(before):
        new = world_at(smd, index)
        for bone, matrix in old.items():
            assert np.allclose(matrix, new[assignment.get(bone, bone)], atol=1e-12)


def test_apply_pool_survives_a_slot_name_colliding_with_a_source_name():
    """
    Renaming one bone at a time would let a target name land on a bone that has
    not been renamed yet, silently merging the two.
    """
    smd = make_chain_smd()
    apply_pool({"ref": smd}, {"B": "C", "C": "B"}, {"B": "A", "C": "A"})
    assert sorted(node.name for node in smd.nodes) == ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# End to end, on the sample models
# ---------------------------------------------------------------------------

def _prepared(directory, reference, rigs):
    model = ModelInput.from_directory(directory.name, directory)
    dedupe_bodygroup_names(model.qc)
    normalisation = normalise_hands(model, reference, rigs, texture="default_hand.bmp")
    select_weapon_groups(model, set())
    pack_always_on_parts(
        model, skip_keys=set(normalisation.replaced_keys) if normalisation.ok else set(),
    )
    prune_model(model)
    return model


@pytest.fixture(scope="module")
def pooled_pistols(pistols_dir, default_hand_path):
    reference, rigs = load_reference_hand(default_hand_path)
    models = [_prepared(d, reference, rigs) for d in discover_models(pistols_dir)]
    before = {
        model.name: {
            key: [world_at(smd, i) for i in range(len(smd.skeleton))]
            for key, smd in model.smds.items()
        }
        for model in models
    }
    shared = {node.name for node in reference.nodes}
    plan, moved = pool_bones(models, shared, reference_hand=reference)
    return models, plan, moved, before, shared


def test_pooling_preserves_every_animation(pooled_pistols):
    """
    The pass renames, re-parents and re-solves thousands of frames; every bone
    must still be exactly where it was in the world, in every frame.
    """
    models, plan, _moved, before, _shared = pooled_pistols

    compared = 0
    worst = 0.0
    for model in models:
        assignment = plan.assignments.get(model.name, {})
        for key, smd in model.smds.items():
            for index, old in enumerate(before[model.name][key]):
                new = world_at(smd, index)
                for bone, matrix in old.items():
                    target = assignment.get(bone, bone)
                    assert target in new, f"{model.name}/{key}: {bone} vanished"
                    worst = max(worst, float(np.abs(matrix - new[target]).max()))
                    compared += 1

    assert compared > 10_000
    assert worst < 1e-6, f"animation drifted by {worst}"


def test_pooling_shrinks_the_merged_skeleton(pooled_pistols, pistols_dir, default_hand_path):
    """The merged bone count should fall well short of the un-pooled sum."""
    models, plan, _moved, _before, shared = pooled_pistols

    merger = ModelMerger()
    for model in models:
        merger.add_model(model)
    pooled = merger.merge("pooled.mdl")

    reference, rigs = load_reference_hand(default_hand_path)
    plain = ModelMerger()
    for directory in discover_models(pistols_dir):
        plain.add_model(_prepared(directory, reference, rigs))
    unpooled = plain.merge("plain.mdl")

    assert pooled.report.total_unique_bones < unpooled.report.total_unique_bones
    # Nothing beyond the shared bones plus one pool.
    assert pooled.report.total_unique_bones <= plan.size + len(shared)


def test_pooling_leaves_no_bone_name_with_two_parents(pooled_pistols):
    """
    A slot that ends up under different parents in different models is renamed
    apart by the merger, which is exactly the cost pooling exists to avoid.
    """
    models, _plan, _moved, _before, _shared = pooled_pistols

    parents: dict[str, tuple[str, str | None]] = {}
    for model in models:
        for smd in model.smds.values():
            by_id = {node.id: node.name for node in smd.nodes}
            for node in smd.nodes:
                parent = by_id.get(node.parent_id) if node.parent_id >= 0 else None
                if parent is None:
                    continue  # roots are reconciled by grafting, not by name
                owner, known = parents.setdefault(node.name, (model.name, parent))
                assert known == parent, (
                    f"{node.name} is under {known} in {owner} but {parent} in {model.name}"
                )


def test_merged_model_keeps_one_weapon_bodygroup_per_model(pooled_pistols):
    models, _plan, _moved, _before, _shared = pooled_pistols
    merger = ModelMerger()
    for model in models:
        merger.add_model(model)
    merged = merger.merge("pooled.mdl")

    weapon_groups = [g for g in merged.qc.bodygroups if g.name.startswith("weapon")]
    assert len(weapon_groups) == 1, [g.name for g in merged.qc.bodygroups]
    # One entry per model, plus the blank shared by models with nothing there.
    assert len(weapon_groups[0].entries) <= len(models) + 1


# ---------------------------------------------------------------------------
# select_weapon_groups
# ---------------------------------------------------------------------------

def test_switchable_groups_collapse_to_their_first_entry():
    from goldsource.qc import QC, BodyGroup, BodyGroupEntry

    model = ModelInput(name="m", qc=QC(modelname="m.mdl"), smds={}, textures={})
    model.qc.bodygroups = [
        BodyGroup(name="scope", entries=[BodyGroupEntry(smd="a"), BodyGroupEntry(smd="b")]),
        BodyGroup(name="led", entries=[BodyGroupEntry(smd="c"), BodyGroupEntry(smd="d")]),
        BodyGroup(name="body", entries=[BodyGroupEntry(smd="e")]),
    ]

    collapsed = select_weapon_groups(model, keep={"led"})

    assert collapsed == ["scope"]
    assert [len(g.entries) for g in model.qc.bodygroups] == [1, 2, 1]
    assert model.qc.bodygroups[0].entries[0].smd == "a"


def test_group_names_are_canonicalised_so_models_share_bodyparts(
    pistols_dir, default_hand_path,
):
    """
    Source models call the same slot ``bodypart1``, ``body``, ``studio``, even
    ``waepon``.  Each spelling that survives becomes its own bodypart, and
    ``pev_body`` is a product over every bodypart — so the names have to agree.
    """
    reference, rigs = load_reference_hand(default_hand_path)
    for directory in discover_models(pistols_dir):
        model = _prepared(directory, reference, rigs)
        for group in model.qc.bodygroups:
            assert group.name == "hands" or group.name.startswith("weapon"), \
                f"{directory.name} kept a group named {group.name!r}"


def test_a_kept_switchable_group_gets_its_own_bodypart():
    """
    Its variants must not land in the same bodypart as other models' weapon
    pieces, so it is named for the model rather than folded into the sequence.
    """
    from goldsource.qc import QC, BodyGroup, BodyGroupEntry
    from goldsource.pipeline import _canonicalise_group_names

    model = ModelInput(name="v_x", qc=QC(modelname="m.mdl"), smds={}, textures={})
    model.qc.bodygroups = [
        BodyGroup(name="body", entries=[BodyGroupEntry(smd="a")]),
        BodyGroup(name="led", entries=[BodyGroupEntry(smd="b"), BodyGroupEntry(smd="c")]),
    ]

    _canonicalise_group_names(model, "weapon", skip=set(), keep_switchable={"led"})

    assert [g.name for g in model.qc.bodygroups] == ["weapon", "v_x_led"]


def test_a_partial_rig_is_completed_to_the_one_shared_hand(
    more_weapons_dir, default_hand_path,
):
    """
    A model whose rig maps four fingers instead of five would otherwise carry a
    trimmed hand mesh no other model shares.  The missing finger is added back
    frozen, so the mesh becomes byte-identical to the full hand and one shared
    hand serves every model whose hands are complete.
    """
    reference, rigs = load_reference_hand(default_hand_path)
    full_hand = {node.name for node in reference.nodes}

    dragon_bones = None
    for directory in discover_models(more_weapons_dir):
        if directory.name != "v_ak47_dragon":
            continue
        model = ModelInput.from_directory(directory.name, directory)
        dedupe_bodygroup_names(model.qc)
        norm = normalise_hands(model, reference, rigs, texture="default_hand.bmp")
        assert norm.ok
        dragon_bones = {node.name for node in model.smds[norm.replaced_keys[0]].nodes}

    # v_ak47_dragon is one of the four-finger rigs; the missing finger is added
    # back so it holds the full hand skeleton (minus the folded root).
    assert dragon_bones is not None
    assert dragon_bones == full_hand - {"Universal_Root"}


def test_hand_is_reposed_onto_the_models_own_bind(more_weapons_dir, default_hand_path):
    """
    The optimised hand must land exactly where the model's own hand bones are,
    or it stretches away from the weapon during animation.  v_ak47chimera's hand
    sits ~90 units from the reference pose, so this is where the bug bit.
    """
    from goldsource.pipeline import _reference_keys
    from goldsource.skeleton import world_transforms

    reference, rigs = load_reference_hand(default_hand_path)
    model = ModelInput.from_directory("v_ak47chimera", more_weapons_dir / "v_ak47chimera")
    dedupe_bodygroup_names(model.qc)
    norm = normalise_hands(model, reference, rigs, texture="default_hand.bmp")
    assert norm.ok

    hand = world_transforms(model.smds[norm.replaced_keys[0]], 0)
    weapon_key = next(k for k in _reference_keys(model) if k not in norm.replaced_keys)
    weapon = world_transforms(model.smds[weapon_key], 0)

    for bone in ("Bip01_L_Forearm", "Bip01_L_Hand", "Bip01_R_Hand"):
        assert bone in hand and bone in weapon
        delta = float(np.linalg.norm(hand[bone][:3, 3] - weapon[bone][:3, 3]))
        assert delta < 1e-3, f"{bone} hand/weapon bind disagree by {delta}"


def test_one_handed_weapons_are_not_given_a_frozen_second_hand(
    more_weapons_dir, default_hand_path,
):
    """A hand the model entirely lacks stays absent — a whole frozen hand would
    float beside the weapon."""
    reference, rigs = load_reference_hand(default_hand_path)
    full = len({node.name for node in reference.nodes})

    model = ModelInput.from_directory("v_portal", more_weapons_dir / "v_portal")
    dedupe_bodygroup_names(model.qc)
    norm = normalise_hands(model, reference, rigs, texture="default_hand.bmp")
    assert norm.ok
    bones = {node.name for node in model.smds[norm.replaced_keys[0]].nodes}
    assert len(bones) < full - 1  # one hand's worth, not both


def test_completion_leaves_the_real_animation_untouched(
    more_weapons_dir, default_hand_path,
):
    """
    Completion only *adds* a static bone; every bone the model already animated
    must keep its exact world transform in every frame.
    """
    reference, rigs = load_reference_hand(default_hand_path)

    raw = ModelInput.from_directory("v_ak47_dragon", more_weapons_dir / "v_ak47_dragon")
    dedupe_bodygroup_names(raw.qc)
    anim_key = next(k for k, s in raw.smds.items() if s.is_animation)
    before = [world_at(raw.smds[anim_key], i) for i in range(len(raw.smds[anim_key].skeleton))]
    original_bones = {node.name for node in raw.smds[anim_key].nodes}

    norm = normalise_hands(raw, reference, rigs, texture="default_hand.bmp")
    assert norm.ok
    renames = norm.bone_renames

    after = [world_at(raw.smds[anim_key], i) for i in range(len(raw.smds[anim_key].skeleton))]
    for index, old in enumerate(before):
        new = after[index]
        for bone, matrix in old.items():
            target = renames.get(bone, bone)
            assert target in new, f"{bone} vanished at frame {index}"
            assert np.allclose(matrix, new[target], atol=1e-9), f"{bone} moved at frame {index}"


def test_hand_groups_are_never_collapsed():
    from goldsource.qc import QC, BodyGroup, BodyGroupEntry

    model = ModelInput(name="m", qc=QC(modelname="m.mdl"), smds={}, textures={})
    model.qc.bodygroups = [
        BodyGroup(name="hands", entries=[BodyGroupEntry(smd="l"), BodyGroupEntry(smd="r")]),
    ]

    assert select_weapon_groups(model, keep=set()) == []
    assert len(model.qc.bodygroups[0].entries) == 2
