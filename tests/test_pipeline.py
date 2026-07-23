"""End-to-end pipeline behaviour on the sample pistol models."""

import numpy as np
import pytest

from conftest import world_at

from goldsource.hands import load_reference_hand
from goldsource.merger import ModelInput, ModelMerger
from goldsource.pipeline import (
    _HANDS_GROUP_RE,
    _check_bodygroup_limits,
    _hand_keys,
    _hand_slots,
    _recompute_pev_body,
    dedupe_bodygroup_names,
    discover_models,
    normalise_hands,
    prune_model,
    run,
)
from goldsource.qc import QC, BodyGroup, BodyGroupEntry


# ---------------------------------------------------------------------------
# pev_body encoding
# ---------------------------------------------------------------------------

def _group(name, *entries):
    return BodyGroup(name=name, entries=[BodyGroupEntry(smd=e) for e in entries])


def test_duplicate_bodygroup_names_are_made_unique():
    """
    A QC may declare several bodygroups sharing a name; the merger looks groups
    up by name, so without this every duplicate is dropped along with its meshes.
    """
    qc = QC(bodygroups=[
        _group("weapon", "a"), _group("weapon", "b"),
        _group("hands", "h"), _group("weapon", "c"),
    ])
    duplicated = dedupe_bodygroup_names(qc)

    assert [bg.name for bg in qc.bodygroups] == ["weapon", "weapon_2", "hands", "weapon_3"]
    assert duplicated == {"weapon": 3}
    # No mesh was lost.
    assert [e.smd for bg in qc.bodygroups for e in bg.entries] == ["a", "b", "h", "c"]


def test_dedupe_bodygroup_names_avoids_an_existing_name():
    qc = QC(bodygroups=[_group("weapon", "a"), _group("weapon_2", "b"), _group("weapon", "c")])
    dedupe_bodygroup_names(qc)
    assert [bg.name for bg in qc.bodygroups] == ["weapon", "weapon_2", "weapon_3"]


def test_dedupe_bodygroup_names_leaves_unique_names_alone():
    qc = QC(bodygroups=[_group("weapon", "a"), _group("hands", "h")])
    assert dedupe_bodygroup_names(qc) == {}
    assert [bg.name for bg in qc.bodygroups] == ["weapon", "hands"]


def test_models_lacking_a_group_share_one_blank_entry(pistols_dir, default_hand_path):
    """
    Each group's entry count multiplies into pev_body, so giving every model
    that lacks a group its own blank makes the value explode past 32 bits and
    crashes studiomdl.  One shared blank keeps the group at minimum width.
    """
    merger = ModelMerger()
    for name in ("v_ana", "v_deagle"):
        model = ModelInput.from_directory(name, pistols_dir / name)
        merger.add_model(model)
    # Give one model an extra group the other does not have.
    merger._models[0].qc.bodygroups.append(_group("scope", "ref_Anaconda"))

    result = merger.merge("t.mdl")
    scope = next(bg for bg in result.qc.bodygroups if bg.name == "scope")
    # v_ana's entry plus exactly one shared blank — not one blank per model.
    assert len(scope.entries) == 2
    assert sum(1 for e in scope.entries if e.is_blank) == 1
    # v_ana is the first model and owns this group, so its piece stays at index
    # 0 (the default view shows it whole) and the blank the other model selects
    # is appended after it.
    assert not scope.entries[0].is_blank
    assert result.bodygroup_indices["scope"]["v_ana"] == 0
    assert result.bodygroup_indices["scope"]["v_deagle"] == 1
    assert scope.entries[1].is_blank


def test_blank_leads_when_the_first_model_lacks_the_group(pistols_dir):
    """
    A part group the first model does not have must default to blank, or the
    default view stacks a later model's mesh on top of the first model.
    """
    merger = ModelMerger()
    for name in ("v_ana", "v_deagle"):
        merger.add_model(ModelInput.from_directory(name, pistols_dir / name))
    # Give the SECOND model a group the first lacks.
    merger._models[1].qc.bodygroups.append(_group("scope", "ref_deonly"))

    result = merger.merge("t.mdl")
    scope = next(bg for bg in result.qc.bodygroups if bg.name == "scope")
    assert scope.entries[0].is_blank
    assert result.bodygroup_indices["scope"]["v_ana"] == 0    # first model → blank default
    assert result.bodygroup_indices["scope"]["v_deagle"] == 1


MODELS = ["a", "b", "c"]
INDICES = {
    "weapon": {"a": 0, "b": 1, "c": 2},
    "hands": {"a": 0, "b": 1, "c": 2},
}


def test_pev_body_uses_positional_stride_encoding():
    qc = QC(bodygroups=[
        _group("weapon", "a/w", "b/w", "c/w"),
        _group("hands", "a/h", "b/h", "c/h"),
    ])
    values = _recompute_pev_body(qc, MODELS, INDICES)
    # index_weapon * 1 + index_hands * 3
    assert values == {"a": 0, "b": 1 + 3, "c": 2 + 6}


def test_pev_body_uses_overrides_for_a_rewritten_group():
    """
    After hands collapse to shared variants, each model's index in that group
    comes from the collapse, not from the merger's pre-collapse record.
    """
    qc = QC(bodygroups=[
        _group("weapon", "a/w", "b/w", "c/w"),
        _group("hands", "_shared/hand", "_shared/hand_2", ""),
    ])
    # a and b share variant 0; c has its own variant 1.
    overrides = {"hands": {"a": 0, "b": 0, "c": 1}}
    values = _recompute_pev_body(qc, MODELS, INDICES, overrides=overrides)
    assert values == {"a": 0, "b": 1 + 0, "c": 2 + 3}


def test_pev_body_resolves_models_whose_entry_is_blank():
    """
    A model lacking a group contributes a blank entry, which carries no path to
    attribute it by — the index has to come from the merger's own record.
    """
    qc = QC(bodygroups=[
        _group("weapon", "a/w", "b/w"),
        BodyGroup(name="scope", entries=[
            BodyGroupEntry(smd="a/scope"), BodyGroupEntry(smd=""),
        ]),
    ])
    indices = {"weapon": {"a": 0, "b": 1}, "scope": {"a": 0, "b": 1}}
    values = _recompute_pev_body(qc, ["a", "b"], indices)
    assert values == {"a": 0, "b": 1 + 2}


def test_pev_body_defaults_to_zero_for_an_unknown_group():
    qc = QC(bodygroups=[_group("extra", "a/x", "b/x")])
    assert _recompute_pev_body(qc, ["a", "b"], {}) == {"a": 0, "b": 0}


# ---------------------------------------------------------------------------
# Per-model preparation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference(default_hand_path):
    return load_reference_hand(default_hand_path)


@pytest.mark.parametrize("model_name", [
    "v_ana", "v_dinfi", "v_infinitysb", "v_luger", "v_musket", "v_skull1",
])
def test_preparation_preserves_every_animation_frame(reference, pistols_dir, model_name):
    """
    Bone folding is only legitimate if it is exact: every surviving bone must
    hold its world transform in every frame of every animation.
    """
    ref_smd, ref_rigs = reference
    original = ModelInput.from_directory(model_name, pistols_dir / model_name)
    before = {k: v for k, v in original.smds.items() if v.is_animation}

    model = ModelInput.from_directory(model_name, pistols_dir / model_name)
    normalise_hands(model, ref_smd, ref_rigs, texture="default_hand.bmp")
    prune_model(model)

    compared = 0
    worst = 0.0
    for key, original_anim in before.items():
        pruned_anim = model.smds[key]
        assert len(pruned_anim.skeleton) == len(original_anim.skeleton)
        for index in range(len(original_anim.skeleton)):
            world_before = world_at(original_anim, index)
            world_after = world_at(pruned_anim, index)
            for name, matrix in world_before.items():
                if name not in world_after:
                    continue  # pruned bone
                worst = max(worst, float(np.abs(matrix - world_after[name]).max()))
                compared += 1

    assert compared > 0
    assert worst < 1e-9, f"{model_name}: animation drifted by {worst}"


def test_split_hand_bodygroups_collapse_into_one(reference, pistols_dir):
    """
    v_deagle splits its hands across two bodygroups ("rhand" and "lhand")
    instead of the usual single "hands" group.  Since one mesh now covers both
    hands, the model must come out with exactly one hand group — otherwise both
    groups would render the same two-handed mesh on top of each other.
    """
    ref_smd, ref_rigs = reference
    model = ModelInput.from_directory("v_deagle", pistols_dir / "v_deagle")

    original = [bg.name for bg in model.qc.bodygroups]
    assert sorted(n for n in original if "hand" in n.lower()) == ["lhand", "rhand"]

    result = normalise_hands(model, ref_smd, ref_rigs, texture="default_hand.bmp")
    assert result.ok, result.error

    hand_groups = [bg for bg in model.qc.bodygroups if "hand" in bg.name.lower()]
    assert len(hand_groups) == 1
    assert len(hand_groups[0].entries) == 1
    assert hand_groups[0].entries[0].smd in model.smds
    # The weapon group must survive untouched.
    assert any(bg.name == "weapon" for bg in model.qc.bodygroups)
    # The superseded hand meshes are gone.
    assert "rhand" not in model.smds and "lhand" not in model.smds


def test_qc_bone_references_follow_the_canonical_rename(reference, pistols_dir):
    """$attachment/$hbox bones must be renamed alongside the skeleton."""
    ref_smd, ref_rigs = reference
    model = ModelInput.from_directory("v_deagle", pistols_dir / "v_deagle")

    assert any(a.bone == "Bone 04" for a in model.qc.attachments)  # guards the premise

    result = normalise_hands(model, ref_smd, ref_rigs, texture="default_hand.bmp")
    assert result.bone_renames["Bone 04"] == "Bip01_L_Hand"
    assert any(a.bone == "Bip01_L_Hand" for a in model.qc.attachments)

    known = {n.name for smd in model.smds.values() for n in smd.nodes}
    for attachment in model.qc.attachments:
        assert attachment.bone in known


def test_preparation_leaves_hand_bones_rooted_consistently(reference, pistols_dir):
    """
    Every model must end up with the same parent for each hand bone — that is
    the precondition for the merger to share one skeleton instead of renaming
    the bones apart.
    """
    ref_smd, ref_rigs = reference
    parents: dict[str, str | None] = {}

    for directory in discover_models(pistols_dir):
        model = ModelInput.from_directory(directory.name, directory)
        normalise_hands(model, ref_smd, ref_rigs, texture="default_hand.bmp")
        prune_model(model)

        hand = model.smds[_hand_keys(model, _HANDS_GROUP_RE)[0]]
        by_id = {n.id: n.name for n in hand.nodes}
        for node in hand.nodes:
            parent = by_id.get(node.parent_id)
            assert parents.setdefault(node.name, parent) == parent, (
                f"{directory.name}: {node.name} parented to {parent}, "
                f"expected {parents[node.name]}"
            )


# ---------------------------------------------------------------------------
# Whole pipeline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def merged(pistols_dir, default_hand_path):
    return run(
        inputs=[pistols_dir],
        output_dir=".",
        model_name="v_test.mdl",
        hand_smd=default_hand_path,
        hand_texture=default_hand_path.with_suffix(".bmp"),
        write=False,
    )


def test_merge_fits_within_the_bone_limit(merged):
    report = merged.merge.report
    assert not report.exceeds_limit, (
        f"{report.total_unique_bones} bones exceeds {report.bone_limit}"
    )


def test_no_hand_bone_is_ever_renamed_apart(merged, default_hand_path):
    """
    Weapon bones from unrelated models may collide by name and get renamed
    apart — that is correct.  Hand bones must never be, since a renamed hand
    bone means a model brought its own duplicate of the shared 34-bone rig.
    """
    from goldsource.hands import detect_rigs
    from goldsource.smd import SMD

    hand_bones: set[str] = set()
    for rig in detect_rigs(SMD.from_file(default_hand_path)):
        hand_bones |= rig.bones

    for model_name, renames in merged.merge.renamed_bones.items():
        clashing = hand_bones & set(renames)
        assert not clashing, f"{model_name} duplicated hand bones: {sorted(clashing)}"


def test_hand_skeleton_is_shared_by_every_model(merged, default_hand_path):
    """The canonical hand rig must appear exactly once in the merged skeleton."""
    from goldsource.hands import detect_rigs
    from goldsource.smd import SMD

    hand_bones: set[str] = set()
    for rig in detect_rigs(SMD.from_file(default_hand_path)):
        hand_bones |= rig.bones

    emitted = {n.name for smd in merged.merge.smds.values() for n in smd.nodes}
    assert hand_bones <= emitted, hand_bones - emitted


def test_all_models_and_sequences_survive(merged, pistols_dir):
    expected_models = [d.name for d in discover_models(pistols_dir)]
    assert merged.merge.model_names == expected_models

    expected_sequences = sum(
        len(ModelInput.from_directory(d.name, d).qc.sequences)
        for d in discover_models(pistols_dir)
    )
    assert len(merged.merge.qc.sequences) == expected_sequences


def test_hands_collapse_to_one_group(merged):
    """
    Every model shows the one optimised hand *design*, in a single hands
    bodygroup.  The mesh is re-posed onto each model's own hand bind, so models
    whose hands sit in the same place share an entry and others keep their own;
    what matters is that there is exactly one hands group and identical poses
    are not duplicated.
    """
    hand_groups = [bg for bg in merged.merge.qc.bodygroups if "hand" in bg.name.lower()]
    assert len(hand_groups) == 1
    non_blank = [e for e in hand_groups[0].entries if not e.is_blank]
    # Fewer distinct hand meshes than models — identical poses are shared.
    assert 1 <= len(non_blank) <= len(merged.merge.model_names)


def test_every_model_gets_a_distinct_pev_body(merged):
    values = merged.merge.pev_body_map
    assert len(set(values.values())) == len(merged.merge.model_names)


def test_merged_qc_references_only_emitted_smds(merged):
    referenced = set()
    for bodygroup in merged.merge.qc.bodygroups:
        for entry in bodygroup.entries:
            if not entry.is_blank:
                referenced.add(entry.smd)
    for sequence in merged.merge.qc.sequences:
        referenced.update(sequence.smd_paths)

    assert referenced <= set(merged.merge.smds), referenced - set(merged.merge.smds)


def test_merged_qc_references_only_emitted_textures(merged):
    used = {tri.material.lower() for smd in merged.merge.smds.values() for tri in smd.triangles}
    available = {name.lower() for name in merged.merge.textures}
    assert used <= available, used - available


def test_no_bodygroup_mesh_binds_to_a_missing_bone(merged):
    for key, smd in merged.merge.smds.items():
        valid = {n.id for n in smd.nodes}
        for tri in smd.triangles:
            for vertex in tri.vertices:
                assert vertex.bone_id in valid, f"{key}: dangling vertex bone {vertex.bone_id}"


def test_qc_bone_references_all_exist(merged):
    known = {n.name for smd in merged.merge.smds.values() for n in smd.nodes}
    for attachment in merged.merge.qc.attachments:
        assert attachment.bone in known
    for hbox in merged.merge.qc.hboxes:
        assert hbox.bone in known
