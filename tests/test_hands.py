"""Hand rig detection and geometric bone matching."""

import numpy as np
import pytest

from conftest import world_at

from goldsource.hands import (
    build_normalised_hand,
    detect_rigs,
    load_reference_hand,
    match_hands,
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


def test_normalised_hand_keeps_only_mapped_bones_and_retargets_texture(reference, pistols_dir):
    ref_smd, ref_rigs = reference
    donor = _donor(pistols_dir, "v_ana")
    match = match_hands(ref_smd, ref_rigs, donor, detect_rigs(donor))

    hand = build_normalised_hand(ref_smd, match, texture="default_hand.bmp")

    names = {node.name for node in hand.nodes}
    assert names == set(match.mapping.values())
    assert "Universal_Root" not in names          # folded away for the merger to re-add
    assert len(hand.triangles) == len(ref_smd.triangles)
    assert {tri.material for tri in hand.triangles} == {"default_hand.bmp"}
    for node in hand.nodes:
        if node.parent_id != -1:
            assert node.parent_id < node.id


def test_normalised_hand_preserves_reference_geometry(reference, pistols_dir):
    """
    Renaming and re-rooting must not move a single vertex relative to its bone,
    otherwise the swapped hand would render displaced.
    """
    ref_smd, ref_rigs = reference
    donor = _donor(pistols_dir, "v_musket")
    match = match_hands(ref_smd, ref_rigs, donor, detect_rigs(donor))
    hand = build_normalised_hand(ref_smd, match)

    ref_world, new_world = world_at(ref_smd), world_at(hand)
    ref_names = {n.id: n.name for n in ref_smd.nodes}
    new_names = {n.id: n.name for n in hand.nodes}

    worst = 0.0
    for ref_tri, new_tri in zip(ref_smd.triangles, hand.triangles):
        for ref_v, new_v in zip(ref_tri.vertices, new_tri.vertices):
            ref_bone = ref_names[ref_v.bone_id]
            assert new_names[new_v.bone_id] == match.mapping[ref_bone]
            ref_local = np.linalg.inv(ref_world[ref_bone]) @ np.array(
                [ref_v.x, ref_v.y, ref_v.z, 1.0])
            new_local = np.linalg.inv(new_world[match.mapping[ref_bone]]) @ np.array(
                [new_v.x, new_v.y, new_v.z, 1.0])
            worst = max(worst, float(np.abs(ref_local - new_local).max()))

    assert worst < 1e-6, f"vertices drifted by {worst}"
