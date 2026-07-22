"""
Regressions found on the wider ``more_weapons`` set.

These models are far less uniform than the pistols: bodygroups sharing a name,
hands parked in a group called "body", rigs with four fingers or a single hand,
and hand meshes with no finger bones at all.
"""

import pytest

from goldsource.hands import build_normalised_hand, detect_rigs, load_reference_hand, match_hands
from goldsource.merger import ModelInput
from goldsource.pipeline import (
    _HANDS_GROUP_RE,
    _check_bodygroup_limits,
    _hand_slots,
    dedupe_bodygroup_names,
    normalise_hands,
    prune_model,
    _resolve_smd,
)
from goldsource.qc import BodyGroup, BodyGroupEntry


@pytest.fixture(scope="module")
def reference(default_hand_path):
    return load_reference_hand(default_hand_path)


def _load(more_weapons_dir, name):
    model = ModelInput.from_directory(name, more_weapons_dir / name)
    dedupe_bodygroup_names(model.qc)
    return model


# ---------------------------------------------------------------------------
# Hand discovery
# ---------------------------------------------------------------------------

def test_hand_found_in_a_group_not_named_for_it(more_weapons_dir, reference):
    """
    v_rpg_remapped keeps its hand mesh in a bodygroup called "body".  The slot
    has to be found by geometry, and the *entry* recorded — rediscovering it by
    name later would leave it pointing at a mesh that no longer exists.
    """
    model = _load(more_weapons_dir, "v_rpg_remapped")
    assert not any("hand" in bg.name.lower() for bg in model.qc.bodygroups)

    slots = _hand_slots(model, _HANDS_GROUP_RE)
    assert len(slots) == 1
    assert slots[0].key == "rpg_hand"
    assert slots[0].group in model.qc.bodygroups


def test_no_dangling_mesh_reference_after_normalising(more_weapons_dir, reference):
    """Every bodygroup entry must still resolve to an SMD the model has."""
    ref_smd, ref_rigs = reference
    for name in ("v_rpg_remapped", "v_portal", "v_ak47chimera", "v_skull5"):
        model = _load(more_weapons_dir, name)
        normalise_hands(model, ref_smd, ref_rigs, texture="default_hand.bmp")
        prune_model(model)
        for bodygroup in model.qc.bodygroups:
            for entry in bodygroup.entries:
                if entry.is_blank:
                    continue
                assert _resolve_smd(model, entry.smd) is not None, (
                    f"{name}: '{bodygroup.name}' -> {entry.smd} is missing"
                )


# ---------------------------------------------------------------------------
# Partial rigs
# ---------------------------------------------------------------------------

def test_single_handed_rig_gets_a_single_handed_mesh(more_weapons_dir, reference):
    """
    v_portal has only a right hand.  Emitting the reference's left hand anyway
    would leave bones no animation drives — a detached hand frozen mid-air.
    """
    ref_smd, ref_rigs = reference
    model = _load(more_weapons_dir, "v_portal")
    donor = model.smds[_hand_slots(model, _HANDS_GROUP_RE)[0].key]

    rigs = detect_rigs(donor)
    assert len(rigs) == 1
    assert rigs[0].hand == "ValveBiped.Bip01_R_Hand"

    match = match_hands(ref_smd, ref_rigs, donor, rigs)
    # The surviving hand is the RIGHT one, so the reference's right hand must be
    # the one bound to it — the two sides score within 0.2% on position alone,
    # so this is decided by handedness.
    assert match.pairs == [("Bip01_R_Hand", "ValveBiped.Bip01_R_Hand")]

    hand = build_normalised_hand(ref_smd, mapped=set(match.mapping))
    names = {node.name for node in hand.nodes}
    assert names == set(match.mapping)
    assert not any(n.startswith("Bip01_L_") for n in names)
    assert hand.triangles, "the matched hand's geometry must survive"
    assert len(hand.triangles) < len(ref_smd.triangles)


def test_missing_finger_keeps_its_geometry_on_the_palm(more_weapons_dir, reference):
    """
    v_skull5's left hand has four fingers.  The unmatched finger's bone goes,
    but its geometry is rebound to the palm rather than deleted, so the hand
    keeps its silhouette instead of gaining a hole.
    """
    ref_smd, ref_rigs = reference
    model = _load(more_weapons_dir, "v_skull5")
    donor = model.smds[_hand_slots(model, _HANDS_GROUP_RE)[0].key]
    match = match_hands(ref_smd, ref_rigs, donor, detect_rigs(donor))

    assert match.pairs and len(match.pairs) == 2
    assert "Bip01_L_Finger2" in match.unmapped

    hand = build_normalised_hand(ref_smd, mapped=set(match.mapping))
    names = {node.name for node in hand.nodes}
    assert "Bip01_L_Finger2" not in names
    # Both hands are present, so no triangles should have been dropped.
    assert len(hand.triangles) == len(ref_smd.triangles)
    assert "Bip01_L_Hand" in names


def test_hand_mesh_without_finger_bones_is_left_alone(more_weapons_dir, reference):
    """
    v_ak47_beast's hand has three bones per arm and no fingers, so the
    reference hand cannot be bound to it.  It must keep its own mesh untouched
    rather than be silently mangled.
    """
    ref_smd, ref_rigs = reference
    model = _load(more_weapons_dir, "v_ak47_beast")
    before = model.smds["handBL"].to_string()

    result = normalise_hands(model, ref_smd, ref_rigs, texture="default_hand.bmp")

    assert not result.ok
    assert "no hand rig" in result.error
    assert model.smds["handBL"].to_string() == before
    assert any(
        not e.is_blank and _resolve_smd(model, e.smd) == "handBL"
        for bg in model.qc.bodygroups for e in bg.entries
    )


# ---------------------------------------------------------------------------
# Engine limits
# ---------------------------------------------------------------------------

def _fake_result(entry_counts):
    class _Fake:
        pass

    result = _Fake()
    result.qc = type("QC", (), {"bodygroups": [
        BodyGroup(name=f"g{i}", entries=[BodyGroupEntry(smd=f"m{j}") for j in range(n)])
        for i, n in enumerate(entry_counts)
    ]})()
    result.pev_body_map = {}
    return result


def test_bodygroup_limits_flag_too_many_bodyparts():
    problems = _check_bodygroup_limits(_fake_result([2] * 40))
    assert any("exceeds studiomdl's limit" in p for p in problems)


def test_bodygroup_limits_flag_pev_body_overflow():
    # 23 groups of 8 entries is 8**23 combinations — far past a 32-bit int.
    problems = _check_bodygroup_limits(_fake_result([8] * 23))
    assert any("overflow" in p for p in problems)


def test_bodygroup_limits_pass_a_reasonable_layout():
    assert _check_bodygroup_limits(_fake_result([9, 2])) == []
