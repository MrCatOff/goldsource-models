"""End-to-end pipeline behaviour on the sample pistol models."""

import numpy as np
import pytest

from conftest import world_at

from goldsource.hands import load_reference_hand
from goldsource.merger import ModelInput
from goldsource.pipeline import (
    _HANDS_GROUP_RE,
    _hand_keys,
    _recompute_pev_body,
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


def test_pev_body_uses_positional_stride_encoding():
    qc = QC(bodygroups=[
        _group("weapon", "a/w", "b/w", "c/w"),
        _group("hands", "a/h", "b/h", "c/h"),
    ])
    values = _recompute_pev_body(qc, ["a", "b", "c"])
    # index_weapon * 1 + index_hands * 3
    assert values == {"a": 0, "b": 1 + 3, "c": 2 + 6}


def test_pev_body_ignores_a_shared_single_entry_group():
    """Collapsing hands to one shared mesh must not shift any model's value."""
    qc = QC(bodygroups=[
        _group("weapon", "a/w", "b/w", "c/w"),
        _group("hands", "_shared/hand"),
    ])
    assert _recompute_pev_body(qc, ["a", "b", "c"]) == {"a": 0, "b": 1, "c": 2}


def test_pev_body_refuses_to_guess_when_a_model_owns_no_entry():
    qc = QC(bodygroups=[_group("weapon", "a/w", "b/w")])
    assert _recompute_pev_body(qc, ["a", "b", "c"]) is None


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


def test_hand_mesh_is_shared_once_across_all_models(merged):
    assert merged.shared_hand
    hand_groups = [bg for bg in merged.merge.qc.bodygroups if "hand" in bg.name.lower()]
    assert len(hand_groups) == 1
    assert len(hand_groups[0].entries) == 1


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
