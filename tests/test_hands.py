"""Hand rig detection and geometric bone matching."""

import numpy as np
import pytest

from conftest import world_at

from goldsource.hands import (
    build_normalised_hand,
    canonical_rename_map,
    detect_rigs,
    load_reference_hand,
    match_hands,
    safe_rename_map,
)
from goldsource.merger import ModelInput
from goldsource.pipeline import _HANDS_GROUP_RE, _hand_keys


@pytest.fixture(scope="module")
def reference(default_hand_path):
    return load_reference_hand(default_hand_path)


def test_reference_hand_has_two_five_fingered_rigs(reference):
    _, rigs = reference
    assert len(rigs) == 2
    assert {rig.hand for rig in rigs} == {"Bip01_L_Hand", "Bip01_R_Hand"}
    for rig in rigs:
        assert len(rig.chains) == 5
        assert all(len(chain) == 3 for chain in rig.chains)


def _donor(pistols_dir, name):
    model = ModelInput.from_directory(name, pistols_dir / name)
    return model.smds[_hand_keys(model, _HANDS_GROUP_RE)[0]]


def test_finger_mapping_follows_geometry_not_bone_numbering(reference, pistols_dir):
    """
    The CSO rig numbers left-hand fingers in reverse and right-hand fingers
    forward, so a name-based mapping would swap index and pinky on one side.
    Matching by bind-pose position gets both sides right.
    """
    ref_smd, ref_rigs = reference
    donor = _donor(pistols_dir, "v_luger")
    match = match_hands(ref_smd, ref_rigs, donor, detect_rigs(donor))

    assert match.mapping["Bip01_L_Forearm"] == "Bone01"
    assert match.mapping["Bip01_L_Hand"] == "Bone_Lefthand"
    # Left hand: descending — the counter-intuitive half.
    assert match.mapping["Bip01_L_Finger0"] == "Bone05"   # thumb
    assert match.mapping["Bip01_L_Finger1"] == "Bone21"   # index, NOT Bone09
    assert match.mapping["Bip01_L_Finger2"] == "Bone17"
    assert match.mapping["Bip01_L_Finger3"] == "Bone13"
    assert match.mapping["Bip01_L_Finger4"] == "Bone09"   # pinky
    # Right hand: ascending.
    assert match.mapping["Bip01_R_Forearm"] == "Bone04"
    assert match.mapping["Bip01_R_Hand"] == "Bone_Righthand"
    assert match.mapping["Bip01_R_Finger0"] == "Bone27"
    assert match.mapping["Bip01_R_Finger1"] == "Bone31"
    assert match.mapping["Bip01_R_Finger4"] == "Bone43"


def test_forearm_is_chosen_by_arm_length_not_by_direct_parent(reference, pistols_dir):
    """
    v_deagle inserts a near-coincident wrist bone directly above the palm
    (``Bone_Righthand``, 0.36 units away) with the real forearm one level higher
    (``Bone04``, 8.54 units — matching the reference's 8.45).  Taking the direct
    parent would bind the forearm mesh to the wrist and slide it down the arm.
    """
    ref_smd, ref_rigs = reference
    donor = _donor(pistols_dir, "v_deagle")
    rigs = detect_rigs(donor)

    right = next(rig for rig in rigs if rig.hand == "Bone26")
    assert right.ancestors[0] == "Bone_Righthand"   # guards the premise

    match = match_hands(ref_smd, ref_rigs, donor, rigs)
    assert match.mapping["Bip01_R_Hand"] == "Bone26"
    assert match.mapping["Bip01_R_Forearm"] == "Bone04"
    assert match.mapping["Bip01_L_Hand"] == "Bone 04"
    assert match.mapping["Bip01_L_Forearm"] == "Bone01"


def test_bone_names_containing_spaces_survive_a_round_trip(pistols_dir):
    """v_deagle has a bone literally named "Bone 04", distinct from "Bone04"."""
    from goldsource.smd import SMD

    donor = _donor(pistols_dir, "v_deagle")
    names = {node.name for node in donor.nodes}
    assert {"Bone 04", "Bone04"} <= names

    reparsed = SMD.from_string(donor.to_string())
    assert {node.name for node in reparsed.nodes} == names


def test_canonical_rename_map_inverts_the_match(reference, pistols_dir):
    ref_smd, ref_rigs = reference
    donor = _donor(pistols_dir, "v_deagle")
    match = match_hands(ref_smd, ref_rigs, donor, detect_rigs(donor))

    inverted = canonical_rename_map(match)
    assert inverted["Bone 04"] == "Bip01_L_Hand"
    assert inverted["Bone04"] == "Bip01_R_Forearm"
    assert len(inverted) == len(match.mapping)


def test_safe_rename_map_never_collapses_two_bones_into_one():
    # "Keep" already exists and is not itself renamed away, so renaming onto it
    # would merge two distinct bones — that entry must be dropped.
    mapping = {"A": "Keep", "B": "Fresh"}
    safe = safe_rename_map(mapping, existing={"A", "B", "Keep"})
    assert safe == {"B": "Fresh"}

    # When the occupant is itself being renamed away, the target is free.
    mapping = {"A": "B", "B": "C"}
    safe = safe_rename_map(mapping, existing={"A", "B"})
    assert safe == {"A": "B", "B": "C"}


def test_every_reference_bone_is_mapped_for_all_sample_models(reference, pistols_dir):
    ref_smd, ref_rigs = reference
    for directory in sorted(p for p in pistols_dir.iterdir() if p.is_dir()):
        donor = _donor(pistols_dir, directory.name)
        match = match_hands(ref_smd, ref_rigs, donor, detect_rigs(donor))
        assert not match.unmapped, f"{directory.name}: {match.unmapped}"
        assert len(match.mapping) == 34, directory.name
        # Distinct reference bones must never collapse onto one model bone.
        assert len(set(match.mapping.values())) == len(match.mapping), directory.name


def test_sides_are_resolved_even_when_rig_order_is_swapped(reference, pistols_dir):
    """v_dinfi lists the right hand first; the match must not mirror the hands."""
    ref_smd, ref_rigs = reference
    donor = _donor(pistols_dir, "v_dinfi")
    rigs = detect_rigs(donor)
    assert rigs[0].hand == "Bone_Righthand"  # guards the premise

    match = match_hands(ref_smd, ref_rigs, donor, rigs)
    assert match.mapping["Bip01_L_Hand"] == "Bone_Lefthand"
    assert match.mapping["Bip01_R_Hand"] == "Bone_Righthand"


def test_normalised_hand_keeps_reference_naming_and_retargets_texture(reference):
    ref_smd, ref_rigs = reference
    hand = build_normalised_hand(ref_smd, texture="default_hand.bmp")

    names = {node.name for node in hand.nodes}
    expected: set[str] = set()
    for rig in ref_rigs:
        expected |= rig.bones
    assert names == expected
    assert "Universal_Root" not in names          # folded away for the merger to re-add
    assert len(hand.triangles) == len(ref_smd.triangles)
    assert {tri.material for tri in hand.triangles} == {"default_hand.bmp"}
    for node in hand.nodes:
        if node.parent_id != -1:
            assert node.parent_id < node.id


def test_normalised_hand_is_identical_regardless_of_target_model(reference):
    """It depends only on the reference hand — that is what makes it shareable."""
    ref_smd, _ = reference
    first = build_normalised_hand(ref_smd, texture="default_hand.bmp")
    second = build_normalised_hand(ref_smd, texture="default_hand.bmp")
    assert first.to_string() == second.to_string()


def test_normalised_hand_preserves_reference_geometry(reference):
    """
    Re-rooting must not move a single vertex relative to its bone, otherwise
    the swapped hand would render displaced.
    """
    ref_smd, _ = reference
    hand = build_normalised_hand(ref_smd)

    ref_world, new_world = world_at(ref_smd), world_at(hand)
    ref_names = {n.id: n.name for n in ref_smd.nodes}
    new_names = {n.id: n.name for n in hand.nodes}

    worst = 0.0
    for ref_tri, new_tri in zip(ref_smd.triangles, hand.triangles):
        for ref_v, new_v in zip(ref_tri.vertices, new_tri.vertices):
            ref_bone = ref_names[ref_v.bone_id]
            assert new_names[new_v.bone_id] == ref_bone
            ref_local = np.linalg.inv(ref_world[ref_bone]) @ np.array(
                [ref_v.x, ref_v.y, ref_v.z, 1.0])
            new_local = np.linalg.inv(new_world[ref_bone]) @ np.array(
                [new_v.x, new_v.y, new_v.z, 1.0])
            worst = max(worst, float(np.abs(ref_local - new_local).max()))

    assert worst < 1e-6, f"vertices drifted by {worst}"
