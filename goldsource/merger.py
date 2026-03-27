"""
GoldSource model merger.

Combines multiple decompiled weapon models into one QC file with bodygroups,
allowing weapon switching via the submodel (bodygroup index) system.

Key constraints handled
-----------------------
- Max 128 bones in a compiled GoldSource model.
- Illegal parent bone replacement: same bone name must have the same parent
  across all reference SMDs. Conflicts are resolved by renaming the minority
  variant, prefixed with the model identifier.
- Too many normals: studiomdl warns at >2048 normals per reference mesh.
  The merger reports per-model triangle counts so you can anticipate this.

Typical workflow
----------------
    from goldsource import QC, SMD
    from goldsource.merger import ModelMerger, ModelInput

    merger = ModelMerger()
    merger.add_model(ModelInput.from_directory("v_ana",   "storage/decompiled/v_ana"))
    merger.add_model(ModelInput.from_directory("v_dinfi", "storage/decompiled/v_dinfi"))

    report = merger.analyze()
    print(report.summary())

    if not report.exceeds_limit:
        result = merger.merge("v_merged.mdl")
        result.qc.save("output/v_merged.qc")
        for rel_path, smd in result.smds.items():
            smd.save(f"output/{rel_path}.smd")

Merged QC bodygroup layout
---------------------------
Same-named bodygroups across models are combined into a single group where
each model contributes one entry (in insertion order). Models that lack a
particular group get a "blank" entry so the indices stay aligned:

    $bodygroup "weapon" {
        studio "v_ana/ref_Anaconda"        // index 0 → v_ana visible
        studio "v_dinfi/NEXON_CSO_..."     // index 1 → v_dinfi visible
    }
    $bodygroup "hands" {
        studio "v_ana/Hand"                // index 0
        studio "v_dinfi/NEXON_CSO_hand..." // index 1
    }

To show model A: set weapon=0, hands=0.
To show model B: set weapon=1, hands=1.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from goldsource.qc import (
    QC,
    Sequence,
    BodyGroup,
    BodyGroupEntry,
    Attachment,
    HitBox,
    BoneController,
    TextureGroup,
)
from goldsource.smd import SMD, Node, BoneTransform, SkeletonFrame


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class ModelInput:
    """
    One model contributing to the merge.

    *name* — a short unique identifier used as a prefix in output paths and for
              renamed bones.  Keep it filesystem-safe (no spaces, slashes, etc.).
    *qc* — parsed QC file.
    *smds* — all SMDs belonging to this model, keyed by their path as referenced
              in the QC (normalized to forward slashes, no ``.smd`` extension).
    """

    name: str
    qc: QC
    smds: dict[str, SMD]
    # Raw texture bytes keyed by filename as it appears in SMD material fields
    # (e.g. "Anaconda_512.BMP").  Populated automatically by from_directory().
    textures: dict[str, bytes] = field(default_factory=dict)

    @classmethod
    def from_directory(cls, name: str, directory: str | Path) -> ModelInput:
        """
        Load a model from a decompiled directory.

        Expects exactly one ``.qc`` file in *directory* and all ``.smd`` files
        referenced by it to be present (recursively).  Any ``.bmp`` / ``.BMP``
        files found are loaded as textures.
        """
        base = Path(directory)
        qc_files = list(base.glob("*.qc"))
        if not qc_files:
            raise FileNotFoundError(f"No .qc file found in {base}")
        if len(qc_files) > 1:
            raise ValueError(f"Multiple .qc files in {base}: {[f.name for f in qc_files]}")

        qc = QC.from_file(qc_files[0])

        smds: dict[str, SMD] = {}
        for smd_file in base.rglob("*.smd"):
            rel = smd_file.relative_to(base).with_suffix("")
            smds[rel.as_posix()] = SMD.from_file(smd_file)

        textures: dict[str, bytes] = {}
        for tex_file in base.rglob("*"):
            if tex_file.suffix.lower() == ".bmp":
                # Key is exactly the filename as referenced in SMD material fields.
                textures[tex_file.name] = tex_file.read_bytes()

        return cls(name=name, qc=qc, smds=smds, textures=textures)


@dataclass
class BoneStats:
    """Per-model bone analysis."""

    model_name: str
    total_bones: int                # all bones in this model's skeleton
    unique_bones: frozenset[str]    # bones present ONLY in this model
    shared_bones: frozenset[str]    # bones present in at least one other model

    @property
    def unique_count(self) -> int:
        return len(self.unique_bones)

    @property
    def shared_count(self) -> int:
        return len(self.shared_bones)


@dataclass
class BoneConflict:
    """
    Two or more models define the same bone name with different parents.
    Studiomdl would reject this as "illegal parent bone replacement".
    The merger resolves it by renaming the minority variant.
    """

    bone_name: str
    # Each entry: (model_name, parent_bone_name_or_None)
    usages: list[tuple[str, str | None]] = field(default_factory=list)


@dataclass
class MergeReport:
    """
    Analysis of what a merge would produce, without actually performing it.
    Obtain via :meth:`ModelMerger.analyze`.
    """

    bone_stats: list[BoneStats]
    total_unique_bones: int     # bones in the merged skeleton (after conflict resolution)
    bone_limit: int             # 128 for standard GoldSource studiomdl
    conflicts: list[BoneConflict]
    exceeds_limit: bool
    removal_suggestions: list[str]  # model names to drop (greedy) to stay under limit
    warnings: list[str]

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines: list[str] = []
        lines.append(
            f"Models analysed: {len(self.bone_stats)}"
        )
        lines.append(
            f"Merged skeleton: {self.total_unique_bones} / {self.bone_limit} bones"
            f"  (includes Universal_Root)"
        )
        lines.append("")
        lines.append("Per-model breakdown:")
        for s in self.bone_stats:
            lines.append(
                f"  {s.model_name:<20} total={s.total_bones:3d}  "
                f"unique={s.unique_count:3d}  shared={s.shared_count:3d}"
            )

        if self.conflicts:
            lines.append(f"\nBone conflicts requiring rename ({len(self.conflicts)}):")
            for c in self.conflicts:
                usages_str = "  |  ".join(
                    f"{m} parent={p!r}" for m, p in c.usages
                )
                lines.append(f"  {c.bone_name!r}:  {usages_str}")

        if self.exceeds_limit:
            lines.append(
                f"\nWARNING: merged skeleton exceeds the {self.bone_limit}-bone limit."
            )
            lines.append(
                "  Suggested models to remove (greedy, most unique bones first):"
            )
            for name in self.removal_suggestions:
                stat = next((s for s in self.bone_stats if s.model_name == name), None)
                freed = stat.unique_count if stat else "?"
                lines.append(f"    - {name}  ({freed} unique bones freed)")

        for w in self.warnings:
            lines.append(f"WARNING: {w}")

        return "\n".join(lines)


@dataclass
class MergeResult:
    """Output of :meth:`ModelMerger.merge`."""

    qc: QC
    # Output SMDs keyed by relative path (forward slashes, no .smd extension).
    # Paths are already prefixed with the model name, e.g. "v_ana/ref_Anaconda".
    smds: dict[str, SMD]
    # Output textures: final filename → raw bytes.
    # Identical textures from multiple models are deduplicated to a single entry.
    # Conflicting textures (same name, different content) are renamed to
    # "{model_name}__{original_stem}.ext".
    textures: dict[str, bytes]
    # Bone renames applied per model: model_name -> {original_name: new_name}
    renamed_bones: dict[str, dict[str, str]]
    # Texture renames applied per model: model_name -> {original_name: new_name}
    renamed_textures: dict[str, dict[str, str]]
    report: MergeReport

    def save(self, output_dir: str | Path) -> None:
        """
        Write the entire merged model to *output_dir*:

        - ``<output_dir>/<modelname>.qc``
        - ``<output_dir>/<model_name>/<smd_name>.smd``  (one subdirectory per source model)
        - ``<output_dir>/<texture>.BMP``                (flat texture directory)
        """
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)

        # QC
        qc_name = Path(self.qc.modelname).stem + ".qc"
        self.qc.save(base / qc_name)

        # SMDs
        for rel_path, smd in self.smds.items():
            smd.save(base / f"{rel_path}.smd")

        # Textures
        for filename, data in self.textures.items():
            (base / filename).write_bytes(data)


# ---------------------------------------------------------------------------
# Merge configuration (optional)
# ---------------------------------------------------------------------------

@dataclass
class TextureReplacement:
    """Single texture swap within a skin variant."""
    original: str      # texture filename as in SMD material field (e.g. "hand.bmp")
    replacement: str   # filename of the replacement texture (e.g. "hand_blood.bmp")


@dataclass
class SkinVariant:
    """A named set of texture replacements for one model."""
    name: str
    model_name: str
    replacements: list[TextureReplacement] = field(default_factory=list)
    # Raw bytes for replacement textures, keyed by filename
    replacement_data: dict[str, bytes] = field(default_factory=dict)


@dataclass
class SkinSlot:
    """One global skin slot in the merged model (slot 0 = all defaults is implicit).
    Maps model_name → the variant name to activate for that model."""
    name: str
    assignments: dict[str, str] = field(default_factory=dict)


@dataclass
class MergeConfig:
    """Optional configuration that customises the merge behaviour."""
    # String replacement rules applied in order to every sequence name.
    # E.g. [("SP_", ""), ("DEPLOY", "draw")] renames "SP_DEPLOY_idle" → "draw_idle".
    sequence_renames: list[tuple[str, str]] = field(default_factory=list)
    # Per-model skin variants: model_name → list of variants
    skin_variants: dict[str, list[SkinVariant]] = field(default_factory=dict)
    # Global skin slots (skin 0 = defaults is always prepended automatically)
    skin_slots: list[SkinSlot] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Merger
# ---------------------------------------------------------------------------

class ModelMerger:
    """Accumulates models and produces a merged result."""

    BONE_LIMIT: int = 127
    # studiomdl warns / errors when a single reference mesh exceeds this many
    # raw triangles (each contributes 3 normals → ~682 triangles ≈ 2046 normals).
    NORMALS_PER_MESH_SOFT_LIMIT: int = 682

    def __init__(self) -> None:
        self._models: list[ModelInput] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_model(self, model: ModelInput) -> None:
        """Add a model to the pool. Names must be unique."""
        if any(m.name == model.name for m in self._models):
            raise ValueError(f"A model named {model.name!r} is already present.")
        self._models.append(model)

    def analyze(self) -> MergeReport:
        """
        Analyse bone statistics and conflicts without performing the merge.

        This is fast and side-effect free — call it before :meth:`merge` to
        decide whether the merge is feasible, and to obtain removal suggestions
        when over the bone limit.
        """
        if not self._models:
            return MergeReport(
                bone_stats=[], total_unique_bones=0, bone_limit=self.BONE_LIMIT,
                conflicts=[], exceeds_limit=False, removal_suggestions=[],
                warnings=[],
            )

        model_bones = self._collect_model_bones()
        bone_stats = _compute_bone_stats(self._models, model_bones)
        canonical, conflicts = _find_canonical_and_conflicts(self._models, model_bones)
        rename_maps = _build_rename_maps(self._models, model_bones, canonical)
        effective = _effective_bone_sets(self._models, model_bones, rename_maps)

        # Build the output reference SMDs (rename + inject Universal_Root) using
        # the same bodygroup-deduplication logic as _build_merged_qc, then run
        # the studiomdl pruning estimate to get an accurate bone count.
        tex_plan = _build_texture_plan(self._models)
        preview_smds: dict[str, SMD] = {}
        preview_ref_names: list[str] = []
        all_group_names = _ordered_unique(
            bg.name for m in self._models for bg in m.qc.bodygroups
        )
        for model in self._models:
            b_rmap = rename_maps[model.name]
            t_rmap = tex_plan.model_renames[model.name]
            # $body entry
            if model.qc.body and not model.qc.body.is_blank:
                raw = model.qc.body.smd
                norm = _norm_path(raw)
                smd = model.smds.get(norm) or next(
                    (s for k, s in model.smds.items() if k.split("/")[-1] == norm.split("/")[-1]),
                    None,
                )
                if smd:
                    out_key = f"{model.name}/{norm}"
                    preview_smds[out_key] = _inject_universal_root(_rewrite_smd(smd, b_rmap, t_rmap))
                    preview_ref_names.append(out_key)
            # $bodygroup entries — one group per unique name (same as _build_merged_qc)
            for group_name in all_group_names:
                bg = model.qc.bodygroup_by_name(group_name)
                if bg is None:
                    continue
                for entry in bg.entries:
                    if entry.is_blank:
                        continue
                    norm = _norm_path(entry.smd)
                    smd = model.smds.get(norm) or next(
                        (s for k, s in model.smds.items() if k.split("/")[-1] == norm.split("/")[-1]),
                        None,
                    )
                    if smd:
                        out_key = f"{model.name}/{norm}"
                        if out_key not in preview_smds:
                            preview_smds[out_key] = _inject_universal_root(
                                _rewrite_smd(smd, b_rmap, t_rmap)
                            )
                        if out_key not in preview_ref_names:
                            preview_ref_names.append(out_key)

        total = _studiomdl_bone_count(preview_smds, preview_ref_names)
        if total == 0:
            # Fallback to simple union count if no reference SMDs resolved
            total = (len(set().union(*effective.values())) if effective else 0) + 1
        exceeds = total > self.BONE_LIMIT

        suggestions = _suggest_removals(effective, self.BONE_LIMIT) if exceeds else []
        warnings = _build_warnings(self._models)

        return MergeReport(
            bone_stats=bone_stats,
            total_unique_bones=total,
            bone_limit=self.BONE_LIMIT,
            conflicts=conflicts,
            exceeds_limit=exceeds,
            removal_suggestions=suggestions,
            warnings=warnings,
        )

    def merge(
        self,
        output_modelname: str = "merged.mdl",
        config: MergeConfig | None = None,
    ) -> MergeResult:
        """
        Perform the merge.

        Returns a :class:`MergeResult` containing:
        - The merged :class:`QC` (call ``.save()`` to write it).
        - A dict of rewritten SMDs keyed by their output-relative path.
        - A dict of bone renames applied per model (for auditing).
        - The :class:`MergeReport` (same as calling :meth:`analyze`).
        """
        if not self._models:
            return MergeResult(
                qc=QC(modelname=output_modelname),
                smds={},
                textures={},
                renamed_bones={},
                renamed_textures={},
                report=self.analyze(),
            )

        model_bones = self._collect_model_bones()
        canonical, _ = _find_canonical_and_conflicts(self._models, model_bones)
        bone_maps = _build_rename_maps(self._models, model_bones, canonical)
        tex_plan = _build_texture_plan(self._models)

        # Mutable copy so skin replacement textures can be appended.
        output_textures: dict[str, bytes] = dict(tex_plan.output_textures)

        # Rewrite every SMD: apply bone renames + texture renames.
        output_smds: dict[str, SMD] = {}
        for model in self._models:
            b_rmap = bone_maps[model.name]
            t_rmap = tex_plan.model_renames[model.name]
            for smd_key, smd in model.smds.items():
                out_key = f"{model.name}/{_norm_path(smd_key)}"
                output_smds[out_key] = _rewrite_smd(smd, b_rmap, t_rmap)

        # Inject Universal_Root into every output SMD so all bodygroup meshes
        # share a common top-level bone.
        output_smds = {k: _inject_universal_root(v) for k, v in output_smds.items()}

        merged_qc = _build_merged_qc(self._models, bone_maps, output_modelname, config)

        # Build $texturegroup if skin slots are configured.
        if config and config.skin_slots:
            tg = _build_texturegroup(self._models, config, tex_plan, output_textures)
            if tg:
                merged_qc.texturegroups.append(tg)

        report = self.analyze()

        # Replace the predicted bone count with the studiomdl-equivalent count
        # from the actual merged output SMDs.
        #
        # studiomdl only keeps bones that are:
        #   a) directly vertex-referenced in at least one reference mesh, OR
        #   b) ancestors of such bones (needed to form the hierarchy).
        # Bones that appear only in the node section with no geometry (effect
        # helpers, IK targets, etc.) are pruned from the compiled .mdl.
        ref_names = _ref_smd_names(merged_qc)
        actual_count = _studiomdl_bone_count(output_smds, ref_names)
        if actual_count > 0:
            report.total_unique_bones = actual_count
            report.exceeds_limit = actual_count > self.BONE_LIMIT

        return MergeResult(
            qc=merged_qc,
            smds=output_smds,
            textures=output_textures,
            renamed_bones={k: v for k, v in bone_maps.items() if v},
            renamed_textures={k: v for k, v in tex_plan.model_renames.items() if v},
            report=report,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_model_bones(self) -> dict[str, dict[str, str | None]]:
        """
        Returns {model_name: {bone_name: parent_name_or_None}}.

        Only reference SMDs (the mesh SMDs listed in ``$bodygroup`` /
        ``$body`` entries) are used.  Animation SMDs are excluded because
        they can carry different parent assignments for the same bone (e.g.
        motion-extraction rigs), which would produce false conflicts and
        inflate the predicted bone count.

        All reference SMDs for a model are unioned so that models with
        multiple bodygroup meshes (each potentially covering a slightly
        different subset of the skeleton) are handled correctly.
        """
        result: dict[str, dict[str, str | None]] = {}
        for model in self._models:
            bones: dict[str, str | None] = {}

            for raw_name in _ref_smd_names(model.qc):
                norm = _norm_path(raw_name)
                smd = model.smds.get(norm)
                if smd is None:
                    # Try basename match (QC may use relative paths)
                    base = norm.split("/")[-1]
                    smd = next(
                        (s for k, s in model.smds.items() if k.split("/")[-1] == base),
                        None,
                    )
                if smd:
                    bones.update(_bone_map(smd))

            # Fallback: no QC bodygroup entries resolved → use any non-animation SMD
            if not bones:
                for smd in model.smds.values():
                    if not smd.is_animation:
                        bones.update(_bone_map(smd))
                        break

            result[model.name] = bones
        return result


# ---------------------------------------------------------------------------
# Internal: bone analysis
# ---------------------------------------------------------------------------

def _bone_map(smd: SMD) -> dict[str, str | None]:
    """Return {bone_name: parent_name} (None for root bones)."""
    id_to_name = {n.id: n.name for n in smd.nodes}
    return {
        n.name: (id_to_name[n.parent_id] if n.parent_id != -1 else None)
        for n in smd.nodes
    }


def _compute_bone_stats(
    models: list[ModelInput],
    model_bones: dict[str, dict[str, str | None]],
) -> list[BoneStats]:
    bone_sets = {m.name: frozenset(model_bones[m.name]) for m in models}
    stats: list[BoneStats] = []
    for model in models:
        own = bone_sets[model.name]
        others = frozenset().union(*(
            s for name, s in bone_sets.items() if name != model.name
        ))
        stats.append(BoneStats(
            model_name=model.name,
            total_bones=len(own),
            unique_bones=own - others,
            shared_bones=own & others,
        ))
    return stats


def _find_canonical_and_conflicts(
    models: list[ModelInput],
    model_bones: dict[str, dict[str, str | None]],
) -> tuple[dict[str, str | None], list[BoneConflict]]:
    """
    For each bone name, determine the canonical parent (majority vote).
    Returns (canonical_map, conflict_list).
    """
    # bone_name -> list of (parent, model_name)
    bone_usages: dict[str, list[tuple[str | None, str]]] = defaultdict(list)
    for model in models:
        for bone_name, parent in model_bones[model.name].items():
            bone_usages[bone_name].append((parent, model.name))

    canonical: dict[str, str | None] = {}
    conflicts: list[BoneConflict] = []

    for bone_name, usages in bone_usages.items():
        # Count how many models use each parent variant.
        from collections import Counter
        counts: Counter[str | None] = Counter(p for p, _ in usages)
        most_common_parent = counts.most_common(1)[0][0]
        canonical[bone_name] = most_common_parent

        if len(counts) > 1:
            conflicts.append(BoneConflict(
                bone_name=bone_name,
                usages=[(model_name, parent) for parent, model_name in usages],
            ))

    return canonical, conflicts


def _build_rename_maps(
    models: list[ModelInput],
    model_bones: dict[str, dict[str, str | None]],
    canonical: dict[str, str | None],
) -> dict[str, dict[str, str]]:
    """
    For each model, build {old_bone_name: new_bone_name} for every bone that
    cannot share its name safely with the canonical skeleton.

    Two situations require renaming:
    1. **Direct conflict** — the bone exists in multiple models but its parent
       differs from the canonical parent (studiomdl "illegal parent bone
       replacement").
    2. **Cascade conflict** — a bone's parent was renamed (situation 1 or 2).
       Because the parent node has a new name in this model while keeping the
       original name in others, the child bone would present two different
       parent names for the same bone name → same error.

    The fix: whenever a bone is renamed, *all of its descendants in the same
    model* are also renamed with the same ``{model_name}__`` prefix so that
    they become fully model-specific and cannot conflict with bones of the
    same name in other models.
    """
    # Bones that appear in more than one model (by original name).
    # Only these can create a parent-mismatch after a cascade rename.
    from collections import Counter as _Counter
    bone_model_count: _Counter[str] = _Counter(
        b for m in models for b in model_bones[m.name]
    )
    shared_bones: frozenset[str] = frozenset(
        b for b, cnt in bone_model_count.items() if cnt > 1
    )

    rename_maps: dict[str, dict[str, str]] = {m.name: {} for m in models}

    for model in models:
        bones = model_bones[model.name]

        # Build a children index for this model's bone tree.
        children: dict[str, list[str]] = defaultdict(list)
        for bone_name, parent in bones.items():
            if parent is not None:
                children[parent].append(bone_name)

        # Step 1 — find directly conflicting bones.
        to_rename: set[str] = set()
        for bone_name, parent in bones.items():
            if canonical.get(bone_name) != parent:
                to_rename.add(bone_name)

        # Step 2 — cascade only into descendants that are SHARED across models.
        # Bones that exist exclusively in this model can never cause a
        # parent-name mismatch in another model, so they don't need renaming.
        queue = list(to_rename)
        while queue:
            bone = queue.pop()
            for child in children.get(bone, []):
                if child not in to_rename and child in shared_bones:
                    to_rename.add(child)
                    queue.append(child)

        # Build the final map.
        for bone_name in to_rename:
            rename_maps[model.name][bone_name] = f"{model.name}__{bone_name}"

    return rename_maps


def _effective_bone_sets(
    models: list[ModelInput],
    model_bones: dict[str, dict[str, str | None]],
    rename_maps: dict[str, dict[str, str]],
) -> dict[str, frozenset[str]]:
    """Post-rename effective bone names per model (what will actually go into studiomdl)."""
    result: dict[str, frozenset[str]] = {}
    for model in models:
        rmap = rename_maps[model.name]
        result[model.name] = frozenset(
            rmap.get(b, b) for b in model_bones[model.name]
        )
    return result


def _suggest_removals(
    effective_bones: dict[str, frozenset[str]],
    limit: int,
) -> list[str]:
    """
    Greedy algorithm: repeatedly remove the model that frees the most
    unique bones until the total fits within *limit*.
    Returns the list of model names to remove (in removal order).
    """
    remaining: dict[str, frozenset[str]] = dict(effective_bones)
    removed: list[str] = []

    def _total(pool: dict[str, frozenset[str]]) -> int:
        return len(frozenset().union(*pool.values())) if pool else 0

    while _total(remaining) > limit and remaining:
        # For each candidate, count bones that are ONLY in that model.
        def _freed(name: str) -> int:
            others = frozenset().union(*(
                b for n, b in remaining.items() if n != name
            ))
            return len(remaining[name] - others)

        best = max(remaining, key=_freed)
        removed.append(best)
        del remaining[best]

    return removed


# ---------------------------------------------------------------------------
# Internal: texture plan
# ---------------------------------------------------------------------------

@dataclass
class _TexturePlan:
    # per-model rename map: original_filename -> output_filename
    model_renames: dict[str, dict[str, str]]
    # deduplicated output textures: output_filename -> raw bytes
    output_textures: dict[str, bytes]


def _build_texture_plan(models: list[ModelInput]) -> _TexturePlan:
    """
    Resolve output filenames for all textures across all models.

    Rules
    -----
    - Texture only in one model → keep original filename unchanged.
    - Same filename, identical bytes across models → one shared copy, no rename.
    - Same filename, different bytes → rename EVERY copy to
      ``{model_name}__{stem}{ext}`` so that no information is silently lost.
    """
    # Group by filename: tex_name -> [(model_name, bytes)]
    by_name: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
    for model in models:
        for tex_name, data in model.textures.items():
            by_name[tex_name].append((model.name, data))

    model_renames: dict[str, dict[str, str]] = {m.name: {} for m in models}
    output_textures: dict[str, bytes] = {}

    for tex_name, entries in by_name.items():
        if len(entries) == 1:
            # Unique filename across all models — keep as-is.
            model_name, data = entries[0]
            output_textures[tex_name] = data

        else:
            hashes = [hashlib.md5(data).hexdigest() for _, data in entries]
            if len(set(hashes)) == 1:
                # Identical content — one shared copy, no rename for any model.
                output_textures[tex_name] = entries[0][1]

            else:
                # Genuinely different content — rename every copy.
                dot = tex_name.rfind(".")
                stem = tex_name[:dot] if dot != -1 else tex_name
                ext = tex_name[dot:] if dot != -1 else ""
                for model_name, data in entries:
                    new_name = f"{model_name}__{stem}{ext}"
                    model_renames[model_name][tex_name] = new_name
                    output_textures[new_name] = data

    return _TexturePlan(model_renames=model_renames, output_textures=output_textures)


# ---------------------------------------------------------------------------
# Internal: SMD rewriting
# ---------------------------------------------------------------------------

def _rewrite_smd(
    smd: SMD,
    bone_rename: dict[str, str],
    tex_rename: dict[str, str],
) -> SMD:
    """Return a copy of *smd* with bone names and triangle materials updated."""
    if not bone_rename and not tex_rename:
        return smd
    new_smd = deepcopy(smd)
    if bone_rename:
        for node in new_smd.nodes:
            if node.name in bone_rename:
                node.name = bone_rename[node.name]
    if tex_rename:
        for tri in new_smd.triangles:
            if tri.material in tex_rename:
                tri.material = tex_rename[tri.material]
    return new_smd


# ---------------------------------------------------------------------------
# Internal: Universal_Root injection
# ---------------------------------------------------------------------------

UNIVERSAL_ROOT_NAME = "Universal_Root"


def _inject_universal_root(smd: SMD) -> SMD:
    """Prepend a ``Universal_Root`` bone as ID 0, shifting all existing bone
    IDs up by one, so that the root bone always has the lowest ID.

    studiomdl expects every parent bone to have a lower ID than its children.
    Appending the root at the *end* (highest ID) breaks that contract.  We
    therefore insert it at ID 0 and increment every existing bone_id reference
    throughout the nodes, skeleton *and* triangle-vertex sections.

    Idempotent: if the SMD already contains a bone named ``Universal_Root``,
    the original is returned unchanged.
    """
    if any(n.name == UNIVERSAL_ROOT_NAME for n in smd.nodes):
        return smd

    new_smd = deepcopy(smd)

    # ── Step 1: shift every existing bone ID up by 1 ───────────────────
    for node in new_smd.nodes:
        node.id += 1
        if node.parent_id != -1:
            node.parent_id += 1
        # Old root bones (parent_id was -1) keep parent_id=-1 for now;
        # we reassign them to Universal_Root (id=0) in step 3.

    for frame in new_smd.skeleton:
        for bt in frame.bones:
            bt.bone_id += 1

    for tri in new_smd.triangles:
        for v in (tri.v0, tri.v1, tri.v2):
            v.bone_id += 1

    # ── Step 2: insert Universal_Root as bone 0 ─────────────────────────
    new_smd.nodes.insert(0, Node(id=0, name=UNIVERSAL_ROOT_NAME, parent_id=-1))

    # ── Step 3: re-parent old root bones to Universal_Root ──────────────
    for node in new_smd.nodes:
        if node.id != 0 and node.parent_id == -1:
            node.parent_id = 0

    # ── Step 4: add identity transform for Universal_Root in every frame ─
    zero = BoneTransform(bone_id=0, tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0)
    for frame in new_smd.skeleton:
        frame.bones.insert(0, zero)

    return new_smd


# ---------------------------------------------------------------------------
# Internal: skin / sequence helpers
# ---------------------------------------------------------------------------

def _apply_seq_renames(name: str, renames: list[tuple[str, str]]) -> str:
    """Apply ordered substring-replacement rules to a sequence name."""
    for old, new in renames:
        name = name.replace(old, new)
    return name


def _build_texturegroup(
    models: list[ModelInput],
    config: MergeConfig,
    tex_plan: _TexturePlan,
    output_textures: dict[str, bytes],
) -> TextureGroup | None:
    """
    Build a ``$texturegroup "skins"`` block from *config.skin_slots*.

    Row 0 = default (original) textures.
    Row N = all originals with substitutions from slot N applied.
    Only textures that appear in at least one slot's replacements are listed.
    Replacement texture bytes are added to *output_textures* in-place.
    """
    if not config.skin_slots:
        return None

    # Collect, in encounter order, all original texture names that are
    # referenced by any variant anywhere.  Translate through the texture
    # plan rename map so we reference the final merged filenames.
    ordered: list[str] = []
    seen: set[str] = set()

    for slot in config.skin_slots:
        for model_name, variant_name in slot.assignments.items():
            for variant in config.skin_variants.get(model_name, []):
                if variant.name != variant_name:
                    continue
                for rep in variant.replacements:
                    final = tex_plan.model_renames.get(model_name, {}).get(
                        rep.original, rep.original
                    )
                    if final not in seen:
                        seen.add(final)
                        ordered.append(final)

    if not ordered:
        return None

    # Row 0: defaults
    rows: list[list[str]] = [list(ordered)]

    # Rows 1..N: per slot
    for slot in config.skin_slots:
        row = list(ordered)
        for model_name, variant_name in slot.assignments.items():
            for variant in config.skin_variants.get(model_name, []):
                if variant.name != variant_name:
                    continue
                for rep in variant.replacements:
                    final_orig = tex_plan.model_renames.get(model_name, {}).get(
                        rep.original, rep.original
                    )
                    if final_orig in seen:
                        idx = ordered.index(final_orig)
                        row[idx] = rep.replacement
                        # Ensure replacement bytes are in the output
                        if rep.replacement not in output_textures:
                            data = variant.replacement_data.get(rep.replacement)
                            if data is not None:
                                output_textures[rep.replacement] = data
        rows.append(row)

    return TextureGroup(name="skins", skins=rows)


# ---------------------------------------------------------------------------
# Internal: merged QC construction
# ---------------------------------------------------------------------------

def _build_merged_qc(
    models: list[ModelInput],
    rename_maps: dict[str, dict[str, str]],
    output_modelname: str,
    config: MergeConfig | None = None,
) -> QC:
    base = models[0].qc
    merged = QC(
        modelname=output_modelname,
        cd=base.cd,
        cdtexture=base.cdtexture,
        cliptotextures=base.cliptotextures,
        scale=base.scale,
        flags=base.flags,
        bbox=base.bbox,
        cbox=base.cbox,
    )

    # ---- bodygroups -------------------------------------------------------
    # Collect all group names in order of first appearance.
    all_group_names: list[str] = _ordered_unique(
        bg.name for m in models for bg in m.qc.bodygroups
    )

    for group_name in all_group_names:
        combined = BodyGroup(name=group_name)
        for model in models:
            bg = model.qc.bodygroup_by_name(group_name)
            if bg is None:
                combined.entries.append(BodyGroupEntry(smd=""))  # blank slot
            else:
                for entry in bg.entries:
                    if entry.is_blank:
                        combined.entries.append(BodyGroupEntry(smd=""))
                    else:
                        # Prefix path with model name
                        new_smd = f"{model.name}/{_norm_path(entry.smd)}"
                        combined.entries.append(BodyGroupEntry(
                            smd=new_smd,
                            reverse=entry.reverse,
                            scale=entry.scale,
                        ))
        merged.bodygroups.append(combined)

    # ---- attachments -------------------------------------------------------
    # GoldSource supports attachment IDs 0–3.  First model per ID wins; warn
    # about collisions in the report (handled by _build_warnings).
    seen_ids: set[int] = set()
    for model in models:
        for att in model.qc.attachments:
            if att.id not in seen_ids:
                merged.attachments.append(att)
                seen_ids.add(att.id)

    # ---- hitboxes ----------------------------------------------------------
    for model in models:
        merged.hboxes.extend(model.qc.hboxes)

    # ---- bone controllers --------------------------------------------------
    seen_ctrl_ids: set[int] = set()
    for model in models:
        for ctrl in model.qc.controllers:
            if ctrl.id not in seen_ctrl_ids:
                merged.controllers.append(ctrl)
                seen_ctrl_ids.add(ctrl.id)

    # ---- texture render modes ----------------------------------------------
    for model in models:
        merged.texturemodes.extend(model.qc.texturemodes)

    # ---- sequences ---------------------------------------------------------
    # Apply rename rules, then prefix with model name on collision.
    seq_renames = config.sequence_renames if config else []
    seen_seq_names: set[str] = set()
    for model in models:
        for seq in model.qc.sequences:
            out_seq = deepcopy(seq)
            # Rewrite SMD paths to include the model prefix.
            out_seq.smd_paths = [
                f"{model.name}/{_norm_path(p)}" for p in seq.smd_paths
            ]
            out_seq.name = _apply_seq_renames(out_seq.name, seq_renames)
            if out_seq.name in seen_seq_names:
                out_seq.name = f"{model.name}__{out_seq.name}"
            seen_seq_names.add(out_seq.name)
            merged.sequences.append(out_seq)

    return merged


# ---------------------------------------------------------------------------
# Internal: misc helpers
# ---------------------------------------------------------------------------

def _studiomdl_bone_count(output_smds: dict[str, "SMD"], ref_names: list[str]) -> int:
    """
    Estimate the number of bones studiomdl will include in the compiled model.

    studiomdl keeps a bone if and only if it (a) is directly referenced by at
    least one vertex in any reference mesh, OR (b) is an ancestor of such a
    bone in the skeleton hierarchy.  Bones that only appear in the nodes
    section with no geometry (effect helpers, IK targets, …) are pruned.

    Returns 0 if no reference SMDs could be resolved.
    """
    needed: set[str] = set()

    for raw in ref_names:
        norm = _norm_path(raw)
        smd = output_smds.get(norm)
        if smd is None:
            base = norm.split("/")[-1]
            smd = next(
                (s for k, s in output_smds.items() if k.split("/")[-1] == base),
                None,
            )
        if smd is None:
            continue

        id_to_node = {n.id: n for n in smd.nodes}
        name_to_node = {n.name: n for n in smd.nodes}

        # Collect vertex-referenced bone ids
        vertex_ids: set[int] = set()
        for tri in smd.triangles:
            for v in (tri.v0, tri.v1, tri.v2):
                vertex_ids.add(v.bone_id)

        # Walk up the parent chain for each vertex-referenced bone
        for bid in vertex_ids:
            node = id_to_node.get(bid)
            while node is not None:
                if node.name in needed:
                    break  # already processed this ancestor chain
                needed.add(node.name)
                parent = id_to_node.get(node.parent_id)
                node = parent

    return len(needed)


def _ref_smd_names(qc: QC) -> list[str]:
    """SMD names referenced in $body and all $bodygroup studio entries."""
    names: list[str] = []
    if qc.body and not qc.body.is_blank:
        names.append(qc.body.smd)
    for bg in qc.bodygroups:
        for entry in bg.entries:
            if not entry.is_blank:
                names.append(entry.smd)
    return names


def _pick_ref_smd(model: ModelInput) -> SMD | None:
    """Return the first reference SMD found in *model*, or None."""
    for raw_name in _ref_smd_names(model.qc):
        norm = _norm_path(raw_name)
        # Exact key match
        if norm in model.smds:
            return model.smds[norm]
        # Basename match (handles cases where $cd shifts the base)
        base = norm.split("/")[-1]
        for key, smd in model.smds.items():
            if key == base or key.split("/")[-1] == base:
                return smd
    # Fallback: any SMD that has triangles (= reference mesh)
    for smd in model.smds.values():
        if not smd.is_animation:
            return smd
    return next(iter(model.smds.values()), None)


def _norm_path(path: str) -> str:
    """Normalise a QC path to forward slashes and strip leading ./"""
    return path.replace("\\", "/").lstrip("./")


def _ordered_unique(iterable) -> list:
    seen: set = set()
    result: list = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _build_warnings(models: list[ModelInput]) -> list[str]:
    warnings: list[str] = []

    # Attachment ID collisions.
    attach_id_models: dict[int, list[str]] = defaultdict(list)
    for model in models:
        for att in model.qc.attachments:
            attach_id_models[att.id].append(model.name)
    for aid, names in attach_id_models.items():
        if len(names) > 1:
            warnings.append(
                f"Attachment ID {aid} is defined by multiple models "
                f"({', '.join(names)}); only the first definition is kept."
            )

    # Large reference meshes (normals risk).
    for model in models:
        for raw_name in _ref_smd_names(model.qc):
            norm = _norm_path(raw_name)
            smd = model.smds.get(norm)
            if smd is None:
                base = norm.split("/")[-1]
                smd = next(
                    (s for k, s in model.smds.items() if k.split("/")[-1] == base),
                    None,
                )
            if smd and len(smd.triangles) > 682:
                warnings.append(
                    f"{model.name}/{norm}: {len(smd.triangles)} triangles "
                    f"(>{682} may exceed the 2048-normals-per-mesh limit in "
                    "standard studiomdl; use an extended compiler if needed)."
                )

    return warnings
