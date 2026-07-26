"""
``--index-sequences`` renames every sequence to ``{model}_seq_{i}`` so each is
self-identifying in Model Viewer, while the original names are preserved as the
models.ini keys so the AMXX index map (``anim_draw = N``) is not lost.
"""

import re

import pytest

from goldsource.merger import MergeConfig, ModelInput, ModelMerger


@pytest.fixture(scope="module")
def two_models(pistols_dir):
    names = ["v_deagle", "v_glock18"]
    models = []
    for n in names:
        d = pistols_dir / n
        if not d.is_dir():
            pytest.skip(f"sample model {n} not present")
        models.append(ModelInput.from_directory(n, d))
    return models


def _merge(models, **cfg):
    merger = ModelMerger()
    for m in models:
        merger.add_model(m)
    return merger.merge("merged.mdl", config=MergeConfig(**cfg))


def test_indexed_names_are_model_prefixed_and_zero_based(two_models):
    result = _merge(two_models, index_sequence_names=True)
    names = [s.name for s in result.qc.sequences]

    # Every name matches {model}_seq_{i}; no originals leak through.
    assert all(re.fullmatch(r"v_(deagle|glock18)_seq_\d+", n) for n in names), names
    # All unique.
    assert len(names) == len(set(names))
    # Per-model indices restart at 0 and are contiguous.
    for model in ("v_deagle", "v_glock18"):
        idx = sorted(int(n.rsplit("_", 1)[1]) for n in names if n.startswith(model + "_seq_"))
        assert idx == list(range(len(idx))), (model, idx)


def test_models_ini_uses_indexed_keys_with_source_name_comment(two_models):
    result = _merge(two_models, index_sequence_names=True)
    ini = result._build_models_ini()

    # The ini is keyed by the Model Viewer name...
    assert "anim_v_deagle_seq_0 = " in ini
    # ...with the original source name preserved as a trailing comment.
    assert re.search(r"^anim_v_deagle_seq_0 = \d+  ; \w+", ini, re.MULTILINE)
    # Every ini anim value is a valid merged-sequence index.
    n_seq = len(result.qc.sequences)
    for val in map(int, re.findall(r"^anim_\S+ = (\d+)", ini, re.MULTILINE)):
        assert 0 <= val < n_seq


def test_models_ini_keeps_source_names_when_not_indexed(two_models):
    """Default (no indexing): ini stays keyed by the source name, no comment."""
    ini = _merge(two_models)._build_models_ini()
    assert re.search(r"^anim_\w+ = \d+$", ini, re.MULTILINE)
    assert "_seq_" not in ini
    assert ";" not in ini


def test_default_off_preserves_source_names(two_models):
    result = _merge(two_models)  # index_sequence_names defaults to False
    names = [s.name for s in result.qc.sequences]
    assert not any(re.fullmatch(r"v_\w+_seq_\d+", n) for n in names), names
