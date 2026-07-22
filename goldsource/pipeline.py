"""
Autonomous merge pipeline.

Turns a directory of decompiled models into one compiled GoldSource model with
per-weapon submodels, running the whole sequence unattended:

1.  **Discover**   — every subdirectory holding exactly one ``.qc``.
2.  **Sanitise**   — rename non-ASCII files studiomdl cannot open.
3.  **Normalise hands** — rebind one optimised hand mesh onto each model's own
    hand bones (see :mod:`goldsource.hands`), so all models end up sharing an
    identical hand skeleton *and* an identical hand mesh.
4.  **Prune**      — drop bones that carry no geometry and are not referenced by
    the QC, folding their transforms into their children so every animation is
    preserved exactly.  This also strips redundant top-level bones
    (``root`` / ``Bone_Root``), which is what lets bones with the same name
    across models collapse into one shared bone instead of being renamed apart.
5.  **Merge**      — combine into one QC with aligned bodygroups
    (:mod:`goldsource.merger`).
6.  **Share hands** — when every model's normalised hand mesh is identical, the
    hands bodygroup collapses to a single entry instead of one copy per model.
7.  **Compile**    — run studiomdl.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from goldsource.compiler import CompileResult, compile_qc
from goldsource.hands import (
    HandNormalisation,
    build_normalised_hand,
    detect_rigs,
    load_reference_hand,
    match_hands,
)
from goldsource.merger import (
    MergeConfig,
    MergeResult,
    ModelInput,
    ModelMerger,
    _norm_path,
    _ref_smd_names,
)
from goldsource.qc import QC, BodyGroupEntry
from goldsource.sanitize import sanitize_directory
from goldsource.skeleton import (
    compute_keep_set,
    graft_ancestors,
    remove_bones,
    renumber,
)
from goldsource.smd import SMD


SHARED_HAND_KEY = "_shared/hand"
_HANDS_GROUP_RE = re.compile(r"hand", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class ModelPrep:
    """Per-model record of what the preparation passes did."""
    name: str
    directory: Path
    renamed_files: dict[str, str] = field(default_factory=dict)
    hands: HandNormalisation | None = None
    pruned_bones: list[str] = field(default_factory=list)
    bones_before: int = 0
    bones_after: int = 0
    sequences: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Everything the pipeline produced."""
    preps: list[ModelPrep] = field(default_factory=list)
    merge: MergeResult | None = None
    output_dir: Path | None = None
    qc_path: Path | None = None
    shared_hand: bool = False
    compile: CompileResult | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines: list[str] = []
        lines.append("Models prepared:")
        for prep in self.preps:
            hand = "n/a"
            if prep.hands is not None:
                hand = "ok" if prep.hands.ok else f"FAILED ({prep.hands.error})"
            lines.append(
                f"  {prep.name:<16} bones {prep.bones_before:3d} -> {prep.bones_after:3d}"
                f"   pruned {len(prep.pruned_bones):3d}"
                f"   seqs {prep.sequences:3d}   hands {hand}"
            )

        if self.merge is not None:
            report = self.merge.report
            lines.append("")
            lines.append(
                f"Merged skeleton: {report.total_unique_bones} / {report.bone_limit} bones"
            )
            if report.conflicts:
                lines.append(f"Bone conflicts resolved by rename: {len(report.conflicts)}")
            lines.append(f"Sequences: {len(self.merge.qc.sequences)}")
            lines.append(f"Textures:  {len(self.merge.textures)}")
            lines.append(f"Hand mesh: {'shared (1 copy)' if self.shared_hand else 'per-model copies'}")
            lines.append("")
            lines.append("pev_body values:")
            for name in self.merge.model_names:
                lines.append(f"  {name:<16} {self.merge.pev_body_map.get(name, 0)}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        if self.compile is not None:
            lines.append("")
            status = "OK" if self.compile.ok else f"FAILED (exit {self.compile.returncode})"
            lines.append(f"Compile: {status}")
            if self.compile.output_mdl is not None and self.compile.ok:
                size = self.compile.output_mdl.stat().st_size
                lines.append(f"  {self.compile.output_mdl}  ({size / 1024:.0f} KB)")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(root: str | Path) -> list[Path]:
    """
    Return every directory under *root* that holds exactly one ``.qc`` file.
    *root* itself qualifies when it is a model directory.
    """
    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")

    if len(list(base.glob("*.qc"))) == 1:
        return [base]

    found = [
        child for child in sorted(base.iterdir())
        if child.is_dir() and len(list(child.glob("*.qc"))) == 1
    ]
    return found


# ---------------------------------------------------------------------------
# Per-model preparation
# ---------------------------------------------------------------------------

def _resolve_smd(model: ModelInput, raw_path: str) -> str | None:
    """Map a QC ``studio`` path onto a key in ``model.smds``."""
    norm = _norm_path(raw_path)
    if norm in model.smds:
        return norm
    base = norm.split("/")[-1].lower()
    for key in model.smds:
        if key.split("/")[-1].lower() == base:
            return key
    return None


def _reference_keys(model: ModelInput) -> list[str]:
    """Keys of every reference (mesh) SMD the QC points at."""
    keys: list[str] = []
    for raw in _ref_smd_names(model.qc):
        key = _resolve_smd(model, raw)
        if key is not None and key not in keys:
            keys.append(key)
    return keys


def _hand_keys(model: ModelInput, group_pattern: re.Pattern[str]) -> list[str]:
    """
    Keys of the SMDs that make up the model's hand bodygroup.

    Falls back to picking the reference mesh with the largest share of vertices
    bound to a detected hand rig, for models whose bodygroup is named something
    other than "hands".
    """
    keys: list[str] = []
    for bodygroup in model.qc.bodygroups:
        if not group_pattern.search(bodygroup.name):
            continue
        for entry in bodygroup.entries:
            if entry.is_blank:
                continue
            key = _resolve_smd(model, entry.smd)
            if key is not None and key not in keys:
                keys.append(key)
    if keys:
        return keys

    best_key, best_share = None, 0.0
    for key in _reference_keys(model):
        smd = model.smds[key]
        rigs = detect_rigs(smd)
        if not rigs or not smd.triangles:
            continue
        hand_bones = set().union(*(rig.bones for rig in rigs))
        id_to_name = {n.id: n.name for n in smd.nodes}
        hits = sum(
            1
            for tri in smd.triangles
            for v in tri.vertices
            if id_to_name.get(v.bone_id) in hand_bones
        )
        share = hits / (len(smd.triangles) * 3)
        if share > best_share:
            best_key, best_share = key, share
    return [best_key] if best_key is not None and best_share > 0.5 else []


def normalise_hands(
    model: ModelInput,
    reference_hand: SMD,
    reference_rigs: list,
    group_pattern: re.Pattern[str] = _HANDS_GROUP_RE,
    texture: str | None = None,
) -> HandNormalisation:
    """
    Replace *model*'s hand mesh(es) with the optimised reference hand, rebound
    to the model's own hand bones so its animations still drive it.
    """
    result = HandNormalisation(model_name=model.name)

    keys = _hand_keys(model, group_pattern)
    if not keys:
        result.error = "no hand bodygroup found"
        return result

    # Detect the model's rig on the mesh that actually carries the hand geometry.
    donor = model.smds[keys[0]]
    model_rigs = detect_rigs(donor)
    if not model_rigs:
        result.error = f"no hand rig detected in {keys[0]}"
        return result

    match = match_hands(reference_hand, reference_rigs, donor, model_rigs)
    if not match.mapping:
        result.error = "hand bones could not be matched"
        return result

    result.mapping = match.mapping
    result.pairs = match.pairs
    result.score = match.score
    result.unmapped = match.unmapped

    new_hand = build_normalised_hand(reference_hand, match, texture=texture)

    retired: set[str] = set()
    for key in keys:
        retired |= {tri.material for tri in model.smds[key].triangles}
        model.smds[key] = new_hand
        result.replaced_keys.append(key)

    # Extra hand meshes collapse onto the first key; the QC entries are rewritten
    # so every hand bodygroup entry points at the single normalised mesh.
    if len(keys) > 1:
        for bodygroup in model.qc.bodygroups:
            if not group_pattern.search(bodygroup.name):
                continue
            for entry in bodygroup.entries:
                if not entry.is_blank:
                    entry.smd = keys[0]

    still_used = {
        tri.material
        for key, smd in model.smds.items()
        if key not in result.replaced_keys
        for tri in smd.triangles
    }
    result.retired_textures = sorted(
        name for name in retired
        if name.lower() not in {u.lower() for u in still_used}
    )
    result.smd = new_hand
    return result


def prune_model(model: ModelInput, keep_hitbox_bones: bool = False) -> tuple[list[str], int, int]:
    """
    Drop every bone that carries no geometry and is not named by the QC, from
    the reference mesh *and* every animation, folding transforms into children
    so all animations are preserved.

    Returns ``(removed_bone_names, bones_before, bones_after)``.
    """
    reference_keys = _reference_keys(model)
    references = [model.smds[key] for key in reference_keys]
    if not references:
        references = [smd for smd in model.smds.values() if not smd.is_animation]

    keep = compute_keep_set(references, model.qc, keep_hitbox_bones=keep_hitbox_bones)

    all_bones: set[str] = set()
    for smd in model.smds.values():
        all_bones |= {n.name for n in smd.nodes}

    bones_before = len(all_bones)
    doomed = all_bones - keep

    removed: set[str] = set()
    for smd in model.smds.values():
        removed.update(remove_bones(smd, doomed))
        renumber(smd)

    # Reference skeletons are authoritative; graft any ancestor a mesh still
    # needs (e.g. when a shared root survived pruning) so every SMD of this
    # model agrees on parentage — studiomdl rejects mismatches outright.
    if references:
        authority = references[0]
        for smd in model.smds.values():
            if smd is not authority:
                graft_ancestors(smd, authority)

    remaining: set[str] = set()
    for smd in model.smds.values():
        remaining |= {n.name for n in smd.nodes}

    return sorted(removed), bones_before, len(remaining)


# ---------------------------------------------------------------------------
# Post-merge clean-up
# ---------------------------------------------------------------------------

def _collapse_shared_hands(
    result: MergeResult,
    hand_keys_by_model: dict[str, list[str]],
    group_pattern: re.Pattern[str] = _HANDS_GROUP_RE,
) -> bool:
    """
    When every model's normalised hand mesh is byte-identical, replace the
    per-model hand bodygroup entries with a single shared entry and keep one
    copy of the mesh.  Returns True when the collapse happened.
    """
    if len(hand_keys_by_model) < 2:
        return False

    output_keys: list[str] = []
    for model_name, keys in hand_keys_by_model.items():
        for key in keys:
            full = f"{model_name}/{key}"
            if full in result.smds:
                output_keys.append(full)

    if len(output_keys) < 2:
        return False

    rendered = {result.smds[key].to_string() for key in output_keys}
    if len(rendered) != 1:
        return False  # meshes differ — keep per-model copies

    groups = [bg for bg in result.qc.bodygroups if group_pattern.search(bg.name)]
    if not groups:
        return False

    shared = result.smds[output_keys[0]]
    for key in output_keys:
        del result.smds[key]
    result.smds[SHARED_HAND_KEY] = shared

    for group in groups:
        group.entries = [BodyGroupEntry(smd=SHARED_HAND_KEY)]

    return True


def _recompute_pev_body(qc: QC, model_names: list[str]) -> dict[str, int] | None:
    """
    Recompute each model's ``pev_body`` after the bodygroup layout changed.

    Bodygroup selections are encoded positionally: value = Σ index_g × stride_g
    where stride_g is the product of the entry counts of all preceding groups.
    Returns ``None`` when a model owns no entry in a multi-entry group, since
    the mapping would then be ambiguous.
    """
    values = {name: 0 for name in model_names}
    stride = 1

    for group in qc.bodygroups:
        count = len(group.entries)
        owners: dict[str, int] = {}
        for index, entry in enumerate(group.entries):
            if entry.is_blank:
                continue
            owner = entry.smd.split("/")[0]
            if owner in values and owner not in owners:
                owners[owner] = index

        if count > 1 and len(owners) < len(model_names):
            return None

        for name in model_names:
            values[name] += owners.get(name, 0) * stride
        stride *= count

    return values


def _strip_unused_textures(result: MergeResult) -> list[str]:
    """Drop texture files and ``$texrendermode`` rows no surviving mesh uses."""
    used = {
        tri.material.lower()
        for smd in result.smds.values()
        for tri in smd.triangles
    }
    # Textures named by a $texturegroup row must stay even if no mesh names them
    # directly — they are runtime skin replacements.
    for group in result.qc.texturegroups:
        for skin in group.skins:
            used.update(name.lower() for name in skin)

    dropped = [name for name in result.textures if name.lower() not in used]
    for name in dropped:
        del result.textures[name]

    result.qc.texturemodes = [
        mode for mode in result.qc.texturemodes if mode.texture.lower() in used
    ]
    return sorted(dropped)


def _dedupe_shared_hand_warnings(
    warnings: list[str],
    hand_keys_by_model: dict[str, list[str]],
) -> list[str]:
    """
    Collapse the per-model mesh warnings that all describe the one shared hand.

    The merger reports mesh-size warnings per source model, so once every model
    points at the same hand mesh the same warning appears N times under N
    different names.  Keep one, renamed to the path actually emitted.
    """
    prefixes = {
        f"{model_name}/{key}:"
        for model_name, keys in hand_keys_by_model.items()
        for key in keys
    }

    kept: list[str] = []
    hand_warning: str | None = None
    for warning in warnings:
        matched = next((p for p in prefixes if warning.startswith(p)), None)
        if matched is None:
            kept.append(warning)
        elif hand_warning is None:
            hand_warning = f"{SHARED_HAND_KEY}:{warning[len(matched):]}"

    if hand_warning is not None:
        kept.append(hand_warning)
    return kept


def _strip_dangling_bone_refs(result: MergeResult) -> list[str]:
    """Remove hitboxes/attachments/controllers pointing at pruned bones."""
    known: set[str] = set()
    for smd in result.smds.values():
        known |= {n.name for n in smd.nodes}

    messages: list[str] = []

    kept_hboxes = [h for h in result.qc.hboxes if h.bone in known]
    if len(kept_hboxes) != len(result.qc.hboxes):
        dropped = {h.bone for h in result.qc.hboxes} - known
        messages.append(f"dropped {len(result.qc.hboxes) - len(kept_hboxes)} "
                        f"$hbox entries on removed bones: {', '.join(sorted(dropped))}")
        result.qc.hboxes = kept_hboxes

    kept_attachments = [a for a in result.qc.attachments if a.bone in known]
    if len(kept_attachments) != len(result.qc.attachments):
        dropped = {a.bone for a in result.qc.attachments} - known
        messages.append(f"dropped {len(result.qc.attachments) - len(kept_attachments)} "
                        f"$attachment entries on removed bones: {', '.join(sorted(dropped))}")
        result.qc.attachments = kept_attachments

    kept_controllers = [c for c in result.qc.controllers if c.bone in known]
    if len(kept_controllers) != len(result.qc.controllers):
        result.qc.controllers = kept_controllers

    return messages


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    inputs: list[str | Path],
    output_dir: str | Path,
    model_name: str = "merged.mdl",
    hand_smd: str | Path | None = None,
    hand_texture: str | Path | None = None,
    normalise: bool = True,
    prune: bool = True,
    keep_hitbox_bones: bool = False,
    share_hands: bool = True,
    sanitise: bool = True,
    exclude: list[str] | None = None,
    merge_config: MergeConfig | None = None,
    compile_model: bool = False,
    studiomdl: str | Path | None = None,
    ignore_warnings: bool = False,
    write: bool = True,
    log=lambda message: None,
) -> PipelineResult:
    """
    Run the full pipeline.  *inputs* may be model directories or parent
    directories containing them.
    """
    result = PipelineResult()
    excluded = {name.lower() for name in (exclude or [])}

    directories: list[Path] = []
    for item in inputs:
        for directory in discover_models(item):
            if directory.name.lower() in excluded:
                log(f"skip {directory.name} (excluded)")
                continue
            if directory not in directories:
                directories.append(directory)

    if not directories:
        raise ValueError(f"No model directories found in: {', '.join(str(i) for i in inputs)}")

    reference_hand: SMD | None = None
    reference_rigs: list = []
    hand_texture_name: str | None = None
    if normalise:
        if hand_smd is None:
            raise ValueError("Hand normalisation requested but no reference hand SMD given.")
        reference_hand, reference_rigs = load_reference_hand(hand_smd)
        log(f"reference hand: {Path(hand_smd).name} "
            f"({len(reference_hand.nodes)} bones, {len(reference_hand.triangles)} triangles, "
            f"{len(reference_rigs)} rigs)")
        if hand_texture is not None:
            hand_texture_name = Path(hand_texture).name

    merger = ModelMerger()
    hand_keys_by_model: dict[str, list[str]] = {}

    for directory in directories:
        prep = ModelPrep(name=directory.name, directory=directory)
        log(f"--- {directory.name}")

        if sanitise:
            prep.renamed_files = sanitize_directory(directory)
            if prep.renamed_files:
                log(f"    sanitised {len(prep.renamed_files)} filename(s)")

        model = ModelInput.from_directory(directory.name, directory)
        prep.sequences = len(model.qc.sequences)

        if normalise and reference_hand is not None:
            normalisation = normalise_hands(
                model, reference_hand, reference_rigs, texture=hand_texture_name,
            )
            prep.hands = normalisation
            if normalisation.ok:
                pairs = ", ".join(f"{a}->{b}" for a, b in normalisation.pairs)
                log(f"    hands rebound ({pairs}), match cost {normalisation.score:.2f}")
                hand_keys_by_model[model.name] = list(normalisation.replaced_keys)
                if normalisation.unmapped:
                    prep.warnings.append(
                        f"reference hand bones left unmapped: {', '.join(normalisation.unmapped)}"
                    )
            else:
                prep.warnings.append(f"hand normalisation skipped: {normalisation.error}")
                log(f"    hand normalisation skipped: {normalisation.error}")

        if hand_texture is not None and normalise:
            texture_path = Path(hand_texture)
            if texture_path.exists():
                model.textures[texture_path.name] = texture_path.read_bytes()

        if prune:
            removed, before, after = prune_model(model, keep_hitbox_bones=keep_hitbox_bones)
            prep.pruned_bones = removed
            prep.bones_before, prep.bones_after = before, after
            log(f"    bones {before} -> {after} ({len(removed)} pruned)")
        else:
            names: set[str] = set()
            for smd in model.smds.values():
                names |= {n.name for n in smd.nodes}
            prep.bones_before = prep.bones_after = len(names)

        merger.add_model(model)
        result.preps.append(prep)
        result.warnings.extend(f"{prep.name}: {w}" for w in prep.warnings)

    log("--- merging")
    merged = merger.merge(model_name, config=merge_config)
    result.merge = merged

    if share_hands and normalise:
        if _collapse_shared_hands(merged, hand_keys_by_model):
            result.shared_hand = True
            recomputed = _recompute_pev_body(merged.qc, merged.model_names)
            if recomputed is not None:
                merged.pev_body_map = recomputed
            else:
                result.warnings.append(
                    "hand meshes were shared but pev_body could not be recomputed; "
                    "verify bodygroup indices manually"
                )
            log("    hand mesh shared across all models (1 copy)")
        else:
            log("    hand meshes differ per model, keeping separate copies")

    dropped = _strip_unused_textures(merged)
    if dropped:
        log(f"    dropped {len(dropped)} unused texture(s)")
    result.warnings.extend(_strip_dangling_bone_refs(merged))
    result.warnings.extend(
        _dedupe_shared_hand_warnings(merged.report.warnings, hand_keys_by_model)
        if result.shared_hand else merged.report.warnings
    )

    if merged.report.exceeds_limit:
        result.warnings.append(
            f"merged skeleton has {merged.report.total_unique_bones} bones, "
            f"over the {merged.report.bone_limit} limit; "
            f"consider excluding: {', '.join(merged.report.removal_suggestions)}"
        )

    if write:
        destination = Path(output_dir)
        merged.save(destination)
        result.output_dir = destination
        result.qc_path = destination / (Path(merged.qc.modelname).stem + ".qc")
        log(f"--- wrote {result.qc_path}")

        if compile_model:
            log("--- compiling")
            result.compile = compile_qc(
                result.qc_path,
                studiomdl=studiomdl,
                ignore_warnings=ignore_warnings,
            )

    return result
