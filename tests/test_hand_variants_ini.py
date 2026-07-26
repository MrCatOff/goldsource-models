"""
With ``--default-hands`` the hands become a fixed set of selectable meshes
(male, female) chosen independently of the weapon.  models.ini must spell out
the pev_body for each variant: ``pev_body`` is the first (male), and every
further variant gets a ``pev_body_<name>`` line offset by the hands stride.
"""

from goldsource.merger import MergeResult, MergeReport
from goldsource.qc import QC


def _result(**kw) -> MergeResult:
    report = MergeReport(
        bone_stats=[], total_unique_bones=0, bone_limit=127,
        conflicts=[], exceeds_limit=False, removal_suggestions=[], warnings=[],
    )
    return MergeResult(
        qc=QC(modelname="m.mdl"), smds={}, textures={},
        renamed_bones={}, renamed_textures={}, report=report,
        model_names=["v_deagle", "v_glock18"],
        pev_body_map={"v_deagle": 4, "v_glock18": 8},
        **kw,
    )


def test_female_pev_body_line_is_male_plus_stride():
    ini = _result(
        hand_variant_stride=34,
        hand_variant_names=["male", "female"],
    )._build_models_ini()

    assert "[v_deagle]\npev_body = 4\npev_body_female = 38" in ini
    assert "[v_glock18]\npev_body = 8\npev_body_female = 42" in ini


def test_three_variants_each_get_a_line():
    ini = _result(
        hand_variant_stride=10,
        hand_variant_names=["male", "female", "robot"],
    )._build_models_ini()

    # v_deagle base 4 -> female 14, robot 24
    assert "pev_body = 4\npev_body_female = 14\npev_body_robot = 24" in ini


def test_no_variant_lines_without_default_hands():
    ini = _result()._build_models_ini()   # stride 0, no names
    assert "pev_body = 4" in ini
    assert "pev_body_" not in ini
