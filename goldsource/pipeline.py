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

import numpy as np

from goldsource.bonepool import PoolPlan, apply_pool, plan_pool
from goldsource.decimate import decimate_mesh
from goldsource.optimise import _bt_from_mat4
from goldsource.compiler import CompileResult, compile_qc
from goldsource.hands import (
    HandNormalisation,
    build_normalised_hand,
    canonical_rename_map,
    detect_rigs,
    load_reference_hand,
    match_hands,
    safe_rename_map,
)
from goldsource.merger import (
    MergeConfig,
    MergeResult,
    ModelInput,
    ModelMerger,
    _norm_path,
    _ref_smd_names,
)
from goldsource.qc import QC, BodyGroup, BodyGroupEntry
from goldsource.sanitize import sanitize_directory
from goldsource.skeleton import (
    animated_bone_names,
    compute_keep_set,
    concat_meshes,
    graft_ancestors,
    remove_bones,
    rename_bones,
    renumber,
    topo_order,
    unique_vertex_count,
    world_transforms,
)
from goldsource.smd import SMD, BoneTransform, Node


SHARED_HAND_KEY = "_shared/hand"
HAND_SMD_KEY = "hands"
# Name for the packed always-on weapon parts.
PART_GROUP_PREFIX = "weapon"
# studiomdl's MAXSTUDIOVERTS per submodel.  The source models sit right at it
# (v_skull5 has a 2045-vertex part), so it is the budget their authors targeted.
VERTEX_BUDGET = 2048
# studiomdl's MAXSTUDIOBONES, minus the slot it reserves.
BONE_LIMIT = 127
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
    renamed_bodygroups: dict[str, int] = field(default_factory=dict)
    packed_groups: tuple[int, int] | None = None
    decimated: tuple[int, int] | None = None
    collapsed_groups: list[str] = field(default_factory=list)
    kept_groups: list[str] = field(default_factory=list)
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
    pool_slots: int = 0
    pool_reshaped: int = 0
    exceeds_bodygroup_limits: bool = False
    hand_variants: int = 0
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
            if self.pool_slots:
                lines.append(f"Pooled weapon bones: {self.pool_slots} slots shared by "
                             f"{len(self.preps)} models "
                             f"({self.pool_reshaped} re-anchored)")
            if report.conflicts:
                lines.append(f"Bone conflicts resolved by rename: {len(report.conflicts)}")
            lines.append(f"Sequences: {len(self.merge.qc.sequences)}")
            lines.append(f"Textures:  {len(self.merge.textures)}")
            if self.shared_hand:
                lines.append(f"Hand mesh: shared, {self.hand_variants} distinct "
                             f"{'copy' if self.hand_variants == 1 else 'copies'}")
            else:
                lines.append("Hand mesh: per-model copies")
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
            if self.compile.ok:
                lines.append("Compile: OK")
            else:
                lines.append(f"Compile: FAILED - {self.compile.failure_reason}")
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

def dedupe_bodygroup_names(qc: QC) -> dict[str, int]:
    """
    Make every ``$bodygroup`` name unique within *qc*, in place.

    A QC may legally declare several bodygroups sharing a name — they are
    independent submodel slots and only their order matters.  The merger,
    however, aligns groups across models *by name* and looks them up with
    ``bodygroup_by_name``, which returns only the first match.  Left alone,
    every duplicate after the first is silently dropped from the merged model,
    taking its meshes with it (``v_ak47chimera`` declares 19 groups, 17 of them
    sharing a name).

    Duplicates are renamed ``name_2``, ``name_3``, … so each keeps its own slot.
    Returns ``{original_name: occurrences}`` for names that needed it.
    """
    taken = set()
    duplicated: dict[str, int] = {}

    for bodygroup in qc.bodygroups:
        if bodygroup.name not in taken:
            taken.add(bodygroup.name)
            continue

        duplicated[bodygroup.name] = duplicated.get(bodygroup.name, 1) + 1
        counter = 2
        while f"{bodygroup.name}_{counter}" in taken:
            counter += 1
        renamed = f"{bodygroup.name}_{counter}"
        taken.add(renamed)
        bodygroup.name = renamed

    return duplicated


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


@dataclass
class _HandSlot:
    """A bodygroup entry that points at one of the model's hand meshes."""
    group: BodyGroup
    entry: BodyGroupEntry
    key: str


def _hand_slots(model: ModelInput, group_pattern: re.Pattern[str]) -> list[_HandSlot]:
    """
    Locate the bodygroup entries holding the model's hand meshes.

    The bodygroup is usually named for it ("hands", "rhand", …), but not
    always: ``v_rpg_remapped`` keeps its hand in a group called "body".  So
    when no name matches, fall back to the reference mesh whose vertices are
    mostly bound to a detected hand rig.

    The owning group and entry are returned alongside the SMD key, because the
    caller has to rewrite exactly those entries — rediscovering them by name
    later would miss the odd cases and leave an entry pointing at a mesh that
    no longer exists.
    """
    slots: list[_HandSlot] = []
    seen: set[str] = set()

    for bodygroup in model.qc.bodygroups:
        if not group_pattern.search(bodygroup.name):
            continue
        for entry in bodygroup.entries:
            if entry.is_blank:
                continue
            key = _resolve_smd(model, entry.smd)
            if key is not None and key not in seen:
                seen.add(key)
                slots.append(_HandSlot(group=bodygroup, entry=entry, key=key))
    if slots:
        return slots

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

    if best_key is None or best_share <= 0.5:
        return []

    for bodygroup in model.qc.bodygroups:
        for entry in bodygroup.entries:
            if not entry.is_blank and _resolve_smd(model, entry.smd) == best_key:
                return [_HandSlot(group=bodygroup, entry=entry, key=best_key)]
    return []


def _hand_keys(model: ModelInput, group_pattern: re.Pattern[str]) -> list[str]:
    """SMD keys of the model's hand meshes."""
    return [slot.key for slot in _hand_slots(model, group_pattern)]


def _complete_hand_bones(
    model: ModelInput,
    reference_hand: SMD,
    reference_rigs: list,
    mapped: set[str],
) -> set[str]:
    """
    For every reference hand the model *partially* has, add the finger bones it
    is missing, static at the reference bind pose, to all of its SMDs.

    A rig that maps four fingers instead of five yields a hand mesh trimmed of
    the fifth — which no longer matches the full hand, so the shared-hand pass
    cannot fold it in and the model carries its own near-duplicate copy.  Adding
    the missing bone back (frozen, since the model's animation never drives it)
    makes the mesh byte-identical to the full hand, so one shared hand serves
    every model whose hands are complete.

    A hand the model *entirely* lacks — a one-handed weapon — is left alone; a
    whole frozen hand would float beside the weapon in the bind pose.  Only
    hands with at least one mapped bone are completed.  Returns the reference
    bones now present (the ``mapped`` set grown by whatever was injected).
    """
    present_rigs = [rig for rig in reference_rigs if rig.bones & mapped]
    wanted: set[str] = set()
    for rig in present_rigs:
        wanted |= rig.bones
    to_add = wanted - mapped
    if not to_add:
        return mapped

    ref_by_id = {node.id: node for node in reference_hand.nodes}
    ref_by_name = {node.name: node for node in reference_hand.nodes}
    ref_local = {
        ref_by_id[bone.bone_id].name: bone
        for bone in reference_hand.skeleton[0].bones
        if bone.bone_id in ref_by_id
    } if reference_hand.skeleton else {}

    # Parents before children, so a finger's sub-bones attach to a bone that is
    # already in place.
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen or name not in ref_by_name:
            return
        seen.add(name)
        node = ref_by_name[name]
        if node.parent_id >= 0:
            visit(ref_by_id[node.parent_id].name)
        order.append(name)

    for name in to_add:
        visit(name)
    injectable = [name for name in order if name in to_add]

    for smd in model.smds.values():
        name_to_id = {node.name: node.id for node in smd.nodes}
        next_id = max((node.id for node in smd.nodes), default=-1) + 1
        added = False
        for name in injectable:
            parent_name = (ref_by_id[ref_by_name[name].parent_id].name
                           if ref_by_name[name].parent_id >= 0 else None)
            if parent_name is None or parent_name not in name_to_id:
                continue  # nothing to hang it off in this mesh
            source = ref_local.get(name)
            smd.nodes.append(Node(id=next_id, name=name, parent_id=name_to_id[parent_name]))
            name_to_id[name] = next_id
            for frame in smd.skeleton:
                frame.bones.append(BoneTransform(
                    bone_id=next_id,
                    tx=source.tx if source else 0.0,
                    ty=source.ty if source else 0.0,
                    tz=source.tz if source else 0.0,
                    rx=source.rx if source else 0.0,
                    ry=source.ry if source else 0.0,
                    rz=source.rz if source else 0.0,
                ))
            next_id += 1
            added = True
        if added:
            renumber(smd)

    return wanted | mapped


def _repose_hand_to_model(new_hand: SMD, donor: SMD) -> None:
    """
    Move the optimised hand mesh onto the *model's* hand bind pose, in place.

    By the time this runs *donor* — the model's own hand mesh — has had its hand
    bones renamed onto the reference naming, so a bone in *new_hand* and the same
    bone in *donor* share a name; *donor* just holds it at the model's position.
    Every such bone is placed exactly where the model has it and the vertices
    riding it are carried along, so the mesh keeps its optimised shape but now
    sits where the model's animations expect the hand — no stretch between bind
    and motion.

    Bones with no counterpart in *donor* (a finger frozen in by
    :func:`_complete_hand_bones`) keep their reference offset from their parent
    and ride it into the new pose.
    """
    if not new_hand.skeleton or not donor.skeleton:
        return

    ref_world = world_transforms(new_hand, 0)
    model_world = world_transforms(donor, 0)
    target: dict[str, np.ndarray | None] = {
        node.name: model_world.get(node.name) for node in new_hand.nodes
    }

    by_id = {node.id: node for node in new_hand.nodes}
    new_world: dict[int, np.ndarray] = {}
    vertex_xform: dict[int, np.ndarray] = {}

    for node_id in topo_order(new_hand):
        node = by_id[node_id]
        current = ref_world.get(node.name)
        if current is None:
            continue
        want = target.get(node.name)
        if want is None:
            # No model pose: ride the (already re-posed) parent with the same
            # offset this bone had in the reference hand.
            if node.parent_id != -1 and node.parent_id in new_world:
                ref_parent = ref_world[by_id[node.parent_id].name]
                want = new_world[node.parent_id] @ (np.linalg.inv(ref_parent) @ current)
            else:
                want = current
        new_world[node_id] = want
        vertex_xform[node_id] = want @ np.linalg.inv(current)

    # Carry the geometry across.
    for triangle in new_hand.triangles:
        for vertex in triangle.vertices:
            xform = vertex_xform.get(vertex.bone_id)
            if xform is None:
                continue
            p = xform @ np.array([vertex.x, vertex.y, vertex.z, 1.0])
            vertex.x, vertex.y, vertex.z = float(p[0]), float(p[1]), float(p[2])
            n = xform[:3, :3] @ np.array([vertex.nx, vertex.ny, vertex.nz])
            norm = float(np.linalg.norm(n))
            if norm > 1e-9:
                n = n / norm
            vertex.nx, vertex.ny, vertex.nz = float(n[0]), float(n[1]), float(n[2])

    # Rewrite each bone's local transform to realise the new world pose.
    for frame in new_hand.skeleton[:1]:
        for bone in frame.bones:
            world = new_world.get(bone.bone_id)
            if world is None:
                continue
            node = by_id[bone.bone_id]
            if node.parent_id != -1 and node.parent_id in new_world:
                local = np.linalg.inv(new_world[node.parent_id]) @ world
            else:
                local = world
            solved = _bt_from_mat4(bone.bone_id, local)
            bone.tx, bone.ty, bone.tz = solved.tx, solved.ty, solved.tz
            bone.rx, bone.ry, bone.rz = solved.rx, solved.ry, solved.rz


def normalise_hands(
    model: ModelInput,
    reference_hand: SMD,
    reference_rigs: list,
    group_pattern: re.Pattern[str] = _HANDS_GROUP_RE,
    texture: str | None = None,
    hands_group_name: str = "hands",
    complete_hands: bool = True,
    repose: bool = True,
) -> HandNormalisation:
    """
    Replace *model*'s hand mesh(es) with the optimised reference hand.

    The model's own hand bones are renamed onto the reference naming, so its
    animations keep driving the new mesh *and* every model ends up with an
    identically named hand skeleton.  However many hand bodygroups the model
    had, it comes out with exactly one holding the single shared mesh.
    """
    result = HandNormalisation(model_name=model.name)

    slots = _hand_slots(model, group_pattern)
    keys = [slot.key for slot in slots]
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

    # Rename this model's hand bones onto the reference naming, everywhere:
    # reference mesh, every animation, and the QC's bone references.  This is
    # what makes differently-named rigs converge on one shared hand skeleton.
    existing: set[str] = set()
    for smd in model.smds.values():
        existing |= {node.name for node in smd.nodes}
    renames = safe_rename_map(canonical_rename_map(match), existing)
    result.bone_renames = renames

    dropped = set(canonical_rename_map(match)) - set(renames)
    if dropped:
        result.error = (
            "hand bone renames would collide with existing bones: "
            + ", ".join(sorted(dropped))
        )
        return result

    for smd in model.smds.values():
        rename_bones(smd, renames)
    _rename_qc_bones(model.qc, renames)

    mapped = set(match.mapping)
    if complete_hands:
        # Grow a near-complete hand up to the full one so it shares the single
        # optimised mesh instead of carrying its own trimmed copy.
        mapped = _complete_hand_bones(model, reference_hand, reference_rigs, mapped)
        result.unmapped = sorted(set(result.unmapped) - mapped)

    new_hand = build_normalised_hand(
        reference_hand, texture=texture, mapped=mapped,
    )

    # The optimised hand is authored around the *reference* rig's pose, but the
    # model's animations drive its own hand bones, which may sit in a very
    # different place (v_ak47chimera's hands are ~90 units from where the
    # reference puts them).  Left as-is the mesh is bound at the reference pose
    # yet animated to the model's, so it stretches away from the weapon — the
    # forearm reads as an elongated bone and the hand detaches.  Re-posing the
    # mesh onto the model's own bind pose makes bind and animation agree again.
    # Skipping it lets every model keep the identical reference-posed mesh (one
    # shared hand, far less geometry) at the cost of that stretch — only sound
    # when the merged models all sit near the reference pose.
    if repose:
        _repose_hand_to_model(new_hand, donor)

    # Collapse however many hand bodygroups the model has (some split left and
    # right into separate groups) into a single group holding the one mesh.
    retired: set[str] = set()
    for key in keys:
        retired |= {tri.material for tri in model.smds[key].triangles}
        del model.smds[key]

    hand_key = _unique_key(model, HAND_SMD_KEY)
    model.smds[hand_key] = new_hand
    result.replaced_keys.append(hand_key)

    # Drop exactly the entries that pointed at the superseded meshes, then put
    # the unified group where the first of them lived.  Groups are matched by
    # identity, not by name, so a hand parked in a group called "body" is still
    # removed instead of being left pointing at a deleted mesh.
    unified = BodyGroup(name=hands_group_name, entries=[BodyGroupEntry(smd=hand_key)])
    doomed_entries = {id(slot.entry) for slot in slots}
    affected = {id(slot.group) for slot in slots}

    rebuilt: list[BodyGroup] = []
    inserted = False
    for bodygroup in model.qc.bodygroups:
        if id(bodygroup) in affected:
            bodygroup.entries = [
                entry for entry in bodygroup.entries if id(entry) not in doomed_entries
            ]
            if not inserted:
                rebuilt.append(unified)
                inserted = True
            # A group emptied by the removal disappears; one that still holds
            # other meshes stays.
            if bodygroup.entries:
                rebuilt.append(bodygroup)
            continue
        rebuilt.append(bodygroup)
    if not inserted:
        rebuilt.append(unified)
    model.qc.bodygroups = rebuilt

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


def _unique_key(model: ModelInput, preferred: str) -> str:
    """An SMD key that does not clash with one the model already uses."""
    if preferred not in model.smds:
        return preferred
    counter = 1
    while f"{preferred}_{counter}" in model.smds:
        counter += 1
    return f"{preferred}_{counter}"


def _rename_qc_bones(qc: QC, renames: dict[str, str]) -> None:
    """Apply a bone rename map to every bone reference in the QC."""
    if not renames:
        return
    for attachment in qc.attachments:
        attachment.bone = renames.get(attachment.bone, attachment.bone)
    for hbox in qc.hboxes:
        hbox.bone = renames.get(hbox.bone, hbox.bone)
    for controller in qc.controllers:
        controller.bone = renames.get(controller.bone, controller.bone)
    qc.keepbones = [renames.get(bone, bone) for bone in qc.keepbones]


def select_weapon_groups(
    model: ModelInput,
    keep: set[str] | None = None,
    hands_pattern: re.Pattern[str] = _HANDS_GROUP_RE,
) -> list[str]:
    """
    Reduce every switchable ``$bodygroup`` to its first entry, so the model
    contributes a single weapon submodel.  Returns the names collapsed.

    A source model's bodygroups mostly *split* one weapon into pieces that are
    all drawn together — those are single-entry and packing already folds them
    into one mesh.  A few carry a genuine choice (a scope on or off, a glowing
    LED strip cycling through frames), and every one of those multiplies the
    merged model's ``pev_body`` radix and costs a bodypart out of the 32
    available, for a variant nothing in the merged model ever selects.

    So by default only the first entry survives.  Naming a group in *keep*
    leaves it switchable, for the cases where the variants are the point.
    Hand groups are never touched — the shared-hand pass owns those.
    """
    kept = keep or set()
    collapsed: list[str] = []

    for bodygroup in model.qc.bodygroups:
        if len(bodygroup.entries) < 2:
            continue
        if bodygroup.name in kept or hands_pattern.search(bodygroup.name):
            continue
        first = next((e for e in bodygroup.entries if not e.is_blank), None)
        if first is None:
            continue
        bodygroup.entries = [first]
        collapsed.append(bodygroup.name)

    if collapsed:
        # The alternatives are gone; drop the meshes only they referenced.
        wanted = {
            _resolve_smd(model, entry.smd)
            for group in model.qc.bodygroups for entry in group.entries
            if not entry.is_blank
        }
        for key in [k for k in model.smds
                    if not model.smds[k].is_animation and k not in wanted]:
            model.smds.pop(key, None)

    return collapsed


def pack_always_on_parts(
    model: ModelInput,
    budget: int = VERTEX_BUDGET,
    skip_keys: set[str] | None = None,
    group_prefix: str = PART_GROUP_PREFIX,
    keep_switchable: set[str] | None = None,
) -> int:
    """
    Merge a model's always-on meshes into as few bodygroups as the vertex
    budget allows.  Returns the number of groups the model ends up with.

    Many models use bodygroups purely to *split* one weapon across several
    meshes — ``v_charger7`` ships ``v_charger7_01`` and ``_02``, ``v_ak47chimera``
    is cut into 19 pieces — and every piece is drawn at once.  They are not
    switchable variants: each such group holds exactly one entry.

    Left alone, the merged model needs one bodypart per piece per model, and
    since bodyparts are independent the viewer's default (every part at index 0)
    shows a mix of several weapons at once.  Packing the pieces back together
    collapses that to a handful of slots.

    Groups holding a real choice — two or more entries, or an explicit blank —
    are left untouched, since their whole purpose is to be switched.
    """
    skip = skip_keys or set()

    packable: list[tuple[BodyGroup, str]] = []
    for bodygroup in model.qc.bodygroups:
        if len(bodygroup.entries) != 1:
            continue  # a real choice, or already empty
        entry = bodygroup.entries[0]
        if entry.is_blank:
            continue
        key = _resolve_smd(model, entry.smd)
        if key is None or key in skip:
            continue
        if entry.reverse or entry.scale is not None:
            continue  # carries per-entry options that must not be merged away
        packable.append((bodygroup, key))

    if len(packable) < 2:
        _canonicalise_group_names(model, group_prefix, skip, keep_switchable=keep_switchable)
        return len(model.qc.bodygroups)

    # Largest first, so big meshes claim a slot and small ones fill the gaps.
    ordered = sorted(
        packable,
        key=lambda item: unique_vertex_count(model.smds[item[1]]),
        reverse=True,
    )

    packs: list[list[str]] = []
    sizes: list[int] = []
    for _bodygroup, key in ordered:
        size = unique_vertex_count(model.smds[key])
        for index, used in enumerate(sizes):
            if used + size <= budget:
                packs[index].append(key)
                sizes[index] += size
                break
        else:
            packs.append([key])
            sizes.append(size)

    if len(packs) >= len(packable):
        # Nothing would be gained by concatenating, but the groups still have to
        # agree on names with every other model's, or each spelling costs a
        # bodypart of its own.
        _canonicalise_group_names(model, group_prefix, skip, keep_switchable=keep_switchable)
        return len(model.qc.bodygroups)

    # Take the meshes out of the model before rebuilding, keeping a local
    # handle on them — they are the inputs to the concatenation.
    sources = {key: model.smds[key] for _bodygroup, key in packable}

    packed_groups = [bodygroup for bodygroup, _key in packable]
    position = model.qc.bodygroups.index(packed_groups[0])
    for bodygroup in packed_groups:
        model.qc.bodygroups.remove(bodygroup)
    for key in sources:
        model.smds.pop(key, None)

    replacements: list[BodyGroup] = []
    for index, keys in enumerate(packs):
        name = group_prefix if index == 0 else f"{group_prefix}_{index + 1}"
        smd_key = _unique_key(model, name)
        model.smds[smd_key] = concat_meshes([sources[key] for key in keys])
        replacements.append(BodyGroup(name=name, entries=[BodyGroupEntry(smd=smd_key)]))

    model.qc.bodygroups[position:position] = replacements
    dedupe_bodygroup_names(model.qc)
    _canonicalise_group_names(model, group_prefix, skip, keep_switchable=keep_switchable)
    return len(model.qc.bodygroups)


def _canonicalise_group_names(
    model: ModelInput,
    group_prefix: str,
    skip: set[str],
    hands_pattern: re.Pattern[str] = _HANDS_GROUP_RE,
    keep_switchable: set[str] | None = None,
) -> None:
    """
    Rename a model's weapon bodygroups to ``weapon``, ``weapon_2``, … in order.

    The merger aligns bodygroups across models *by name*, so two models that
    each contribute one weapon submodel share a bodypart only if they agree on
    what to call it.  Source models do not: the same slot is variously
    ``bodypart1``, ``body``, ``studio``, even ``waepon``.  Left alone each
    spelling becomes its own bodypart, and since ``pev_body`` is a mixed-radix
    product over *every* bodypart, each one multiplies the value every model
    needs — 22 bodyparts put it 3000x past the 32-bit ceiling.
    """
    index = 0
    kept = keep_switchable or set()
    for bodygroup in model.qc.bodygroups:
        if bodygroup.name in kept:
            # A group deliberately left switchable is this model's own choice,
            # not a slot to line up with other models' weapon pieces — sharing
            # the name would put its variants in the same bodypart as their
            # meshes.  Give it one of its own.
            bodygroup.name = f"{model.name}_{bodygroup.name}"
            continue
        holds_hand = any(_resolve_smd(model, entry.smd) in skip
                         for entry in bodygroup.entries if not entry.is_blank)
        if holds_hand or hands_pattern.search(bodygroup.name):
            # The hand is not always in a group named for it — ``v_rpg_remapped``
            # keeps it in ``body`` — and the shared-hand collapse finds its group
            # by name.  Naming it here is what lets every model's hand share one
            # bodypart instead of one apiece.
            bodygroup.name = HAND_SMD_KEY
            continue
        index += 1
        bodygroup.name = group_prefix if index == 1 else f"{group_prefix}_{index}"
    dedupe_bodygroup_names(model.qc)


def prune_model(
    model: ModelInput,
    keep_hitbox_bones: bool = False,
    keep_animated_bones: bool = False,
) -> tuple[list[str], int, int]:
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

    if keep_animated_bones:
        # Folding an animated bone spreads its motion into every child, which
        # can push a sequence past studiomdl's 64 KB cap even as the bone count
        # falls.  Keeping them costs bones but leaves the animation data as
        # compressible as it was.
        for smd in model.smds.values():
            keep |= animated_bone_names(smd)

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
        # The mesh with the fullest skeleton defines the hierarchy.  Taking the
        # first one would let a substituted hand — which only knows its own
        # bones — become the authority, leaving the weapon meshes' extra
        # ancestors ungrafted and their parentage in conflict.
        authority = max(references, key=lambda smd: len(smd.nodes))
        for smd in model.smds.values():
            if smd is not authority:
                graft_ancestors(smd, authority)

    remaining: set[str] = set()
    for smd in model.smds.values():
        remaining |= {n.name for n in smd.nodes}

    return sorted(removed), bones_before, len(remaining)


def _authority(model: ModelInput) -> SMD | None:
    """The reference mesh whose skeleton is the fullest — the model's hierarchy."""
    references = [model.smds[key] for key in _reference_keys(model) if key in model.smds]
    if not references:
        references = [smd for smd in model.smds.values() if not smd.is_animation]
    return max(references, key=lambda smd: len(smd.nodes)) if references else None


def _hierarchy(smd: SMD) -> dict[str, str | None]:
    by_id = {node.id: node.name for node in smd.nodes}
    return {
        node.name: (by_id.get(node.parent_id) if node.parent_id >= 0 else None)
        for node in smd.nodes
    }


def pick_pool_anchor(models: list[ModelInput], shared: set[str]) -> str | None:
    """
    The shared bone to hang pooled weapon bones off: whichever one the most
    models already attach a weapon root to, so the fewest bones have to move.
    """
    if not shared:
        return None
    votes: dict[str, int] = {}
    for model in models:
        authority = _authority(model)
        if authority is None:
            continue
        hierarchy = _hierarchy(authority)
        for bone, parent in hierarchy.items():
            if bone not in shared and parent in shared:
                votes[parent] = votes.get(parent, 0) + 1
    if votes:
        return max(votes, key=lambda name: (votes[name], name))
    return None


def _ensure_anchor(smd: SMD, anchor: str, donor: SMD) -> bool:
    """
    Give *smd* the *anchor* bone, placed where *donor* has it and static in
    every frame, then graft in whatever ancestors *donor* gives it.

    A model whose hand rig could not be normalised has none of the shared bones,
    so its pooled weapon roots would come out as roots while every other model's
    hang off the anchor.  The merger sees one name with two different parents
    and renames them apart — undoing the pooling for exactly the models that
    have the least to share.  An inert copy of the anchor chain costs nothing
    (those bones are shared with every other model anyway) and keeps the
    parentage agreeing.

    The pose is taken from a *prepared* model rather than the reference hand,
    so the grafted chain matches what pruning left the other models with.
    """
    present = {node.name for node in smd.nodes}
    if anchor in present:
        return False

    world = world_transforms(donor, 0).get(anchor)
    if world is None:
        return False

    next_id = max((node.id for node in smd.nodes), default=-1) + 1
    smd.nodes.append(Node(id=next_id, name=anchor, parent_id=-1))
    local = _bt_from_mat4(next_id, world)
    for frame in smd.skeleton:
        frame.bones.append(BoneTransform(
            bone_id=next_id, tx=local.tx, ty=local.ty, tz=local.tz,
            rx=local.rx, ry=local.ry, rz=local.rz,
        ))

    renumber(smd)
    graft_ancestors(smd, donor)
    return True


def pool_bones(
    models: list[ModelInput],
    shared: set[str],
    anchor: str | None = None,
    reference_hand: SMD | None = None,
    bone_limit: int = BONE_LIMIT,
) -> tuple[PoolPlan | None, dict[str, int]]:
    """
    Put every model's weapon bones onto a shared pool of bone slots, so merging
    costs the *largest* model's bone count rather than the sum of all of them.

    Slots are renamed and re-parented in place across each model's meshes and
    animations; :func:`goldsource.bonepool.reparent` re-solves every frame so
    the animations are unchanged.  QC bone references travel with the rename.

    Returns the plan and ``{model: bones moved}``.
    """
    anchor = anchor or pick_pool_anchor(models, shared)
    if anchor is None:
        return None, {}

    donor = next(
        (authority for authority in (_authority(model) for model in models)
         if authority is not None and any(n.name == anchor for n in authority.nodes)),
        None,
    )
    if donor is not None:
        for model in models:
            if any(n.name == anchor for n in (_authority(model) or SMD()).nodes):
                continue
            for smd in model.smds.values():
                _ensure_anchor(smd, anchor, donor)

    forests: dict[str, dict[str, str | None]] = {}
    for model in models:
        # Every bone the model has anywhere, not just in its fullest mesh: a
        # bone left out of the pool keeps its original name and collides with
        # the identically-named bone of some other model at merge time, which
        # is exactly the cost pooling exists to remove.
        authority = _authority(model)
        hierarchy: dict[str, str | None] = {}
        for smd in model.smds.values():
            for bone, parent in _hierarchy(smd).items():
                if bone not in hierarchy or hierarchy[bone] is None:
                    hierarchy[bone] = parent
        if authority is not None:
            hierarchy.update(_hierarchy(authority))
        if hierarchy:
            forests[model.name] = hierarchy

    # Slots the pool may grow to before it starts re-anchoring bones to reuse
    # one: everything the bone limit leaves over once the shared bones are in.
    max_slots = max(1, bone_limit - len(shared))
    plan = plan_pool(forests, shared, anchor, max_slots=max_slots)

    moved: dict[str, int] = {}
    for model in models:
        assignment = plan.assignments.get(model.name)
        if not assignment:
            continue
        moved[model.name] = apply_pool(model.smds, assignment, plan.parents)
        _rename_qc_bones(model.qc, assignment)

        # Re-parenting may have left a mesh without the anchor it now hangs off;
        # graft it back so every SMD of this model agrees on parentage.
        authority = _authority(model)
        if authority is not None:
            for smd in model.smds.values():
                if smd is not authority:
                    graft_ancestors(smd, authority)

    return plan, moved


# ---------------------------------------------------------------------------
# Post-merge clean-up
# ---------------------------------------------------------------------------

def _collapse_shared_hands(
    result: MergeResult,
    hand_keys_by_model: dict[str, list[str]],
    hands_group_name: str = HAND_SMD_KEY,
) -> tuple[dict[str, dict[str, int]], int]:
    """
    Replace the per-model hand bodygroup entries with one entry per *distinct*
    hand mesh, so identical hands are stored once.

    Most models end up with the exact same normalised hand, but not all: a rig
    with four fingers or a single hand yields a trimmed mesh (see
    :func:`~goldsource.hands.build_normalised_hand`).  Requiring every mesh to
    be identical before sharing would mean one odd model forces all 58 to carry
    their own copy, so meshes are grouped by content and shared within a group.

    A trailing blank entry is added for models whose hands were not normalised;
    they keep their own hand bodygroup and must not be shown a shared one.

    Only the group *normalisation itself created* is rewritten, matched by
    exact name.  Matching on a name pattern instead would also catch the
    original hand group of a model that could not be normalised — wiping the
    entry for the hand mesh it still needs.

    Returns ``({group_name: {model_name: entry_index}}, variant_count)``.
    """
    groups = [bg for bg in result.qc.bodygroups if bg.name == hands_group_name]
    if not groups or len(hand_keys_by_model) < 2:
        return {}, 0

    owned: dict[str, str] = {}
    for model_name, keys in hand_keys_by_model.items():
        for key in keys:
            full = f"{model_name}/{key}"
            if full in result.smds:
                owned[model_name] = full
                break

    if len(owned) < 2:
        return {}, 0

    # Group meshes by content, in model order so numbering is deterministic.
    variant_of: dict[str, int] = {}
    sources: list[str] = []
    assignment: dict[str, int] = {}
    for model_name in result.model_names:
        key = owned.get(model_name)
        if key is None:
            continue
        rendered = result.smds[key].to_string()
        if rendered not in variant_of:
            variant_of[rendered] = len(sources)
            sources.append(key)
        assignment[model_name] = variant_of[rendered]

    if len(sources) >= len(owned):
        return {}, 0  # every model's hand is unique — nothing to share

    shared_keys: list[str] = []
    for index, source in enumerate(sources):
        name = SHARED_HAND_KEY if index == 0 else f"{SHARED_HAND_KEY}_{index + 1}"
        shared_keys.append(name)
        result.smds[name] = result.smds[source]
    for key in set(owned.values()):
        del result.smds[key]

    # Models that kept their own hands select the trailing blank.
    blank_index = len(shared_keys)
    entries = [BodyGroupEntry(smd=key) for key in shared_keys]
    entries.append(BodyGroupEntry(smd=""))
    for model_name in result.model_names:
        assignment.setdefault(model_name, blank_index)

    for group in groups:
        group.entries = list(entries)

    return {group.name: dict(assignment) for group in groups}, len(shared_keys)


def _recompute_pev_body(
    qc: QC,
    model_names: list[str],
    group_indices: dict[str, dict[str, int]],
    overrides: dict[str, dict[str, int]] | None = None,
) -> dict[str, int]:
    """
    Recompute each model's ``pev_body`` after the bodygroup layout changed.

    Bodygroup selections are encoded positionally: value = Σ index_g × stride_g,
    where stride_g is the product of the entry counts of all preceding groups.

    The per-model entry indices come from *group_indices* (recorded by the
    merger) rather than being inferred from entry paths, because a model that
    lacks a group contributes a *blank* entry, and a blank carries no path to
    attribute it by.  *overrides* supplies replacement indices for groups whose
    entries were rewritten after the merge, such as the shared hands group.
    """
    values = {name: 0 for name in model_names}
    replaced = overrides or {}
    stride = 1

    for group in qc.bodygroups:
        indices = replaced.get(group.name) or group_indices.get(group.name, {})
        for name in model_names:
            values[name] += indices.get(name, 0) * stride
        stride *= len(group.entries)

    return values


# studiomdl's MAXSTUDIOBODYPARTS; it writes past the array without checking.
MAX_BODYPARTS = 32
# pev_body is a signed 32-bit int in the engine.
MAX_BODY_VALUE = 2 ** 31 - 1


def _check_bodygroup_limits(result: MergeResult) -> list[str]:
    """
    Flag bodygroup layouts that studiomdl or the engine cannot represent.

    Submodel selection is encoded as a single integer — the mixed-radix product
    of every group's entry count — so groups multiply rather than add.  Enough
    of them overflows the value (and crashes studiomdl outright, with no error
    message, when the bodypart count exceeds its fixed array).
    """
    messages: list[str] = []

    count = len(result.qc.bodygroups)
    if count > MAX_BODYPARTS:
        messages.append(
            f"{count} bodygroups exceeds studiomdl's limit of {MAX_BODYPARTS}; "
            f"the compiler will crash without reporting an error. Merge fewer "
            f"models per output, or ones with fewer bodygroups."
        )

    combinations = 1
    for group in result.qc.bodygroups:
        combinations *= max(1, len(group.entries))
    if combinations > MAX_BODY_VALUE:
        messages.append(
            f"bodygroup combinations ({combinations:.3g}) overflow the 32-bit "
            f"pev_body value; submodel selection would be undefined in game. "
            f"Merge fewer models per output."
        )

    largest = max(
        (value for value in result.pev_body_map.values()), default=0
    )
    if largest > MAX_BODY_VALUE:
        messages.append(
            f"largest pev_body value ({largest}) exceeds the 32-bit limit."
        )

    return messages


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


def _studiomdl_surviving_bones(result: MergeResult) -> set[str]:
    """
    The bones studiomdl will actually keep in the compiled model.

    It drops any bone that carries no vertices and is not an ancestor of one,
    regardless of whether the SMD still declares it — so a bone can be present
    in our output and still be absent from the .mdl.  Attachments and
    controllers pin their bones; hitboxes do not, and one left pointing at a
    dropped bone aborts the compile with "cannot find bone ... for bbox".
    """
    survivors: set[str] = set()

    for raw in _ref_smd_names(result.qc):
        key = _norm_path(raw)
        smd = result.smds.get(key)
        if smd is None:
            base = key.split("/")[-1]
            smd = next(
                (s for k, s in result.smds.items() if k.split("/")[-1] == base),
                None,
            )
        if smd is None:
            continue

        by_id = {n.id: n for n in smd.nodes}
        for bone_id in {v.bone_id for tri in smd.triangles for v in tri.vertices}:
            node = by_id.get(bone_id)
            while node is not None and node.name not in survivors:
                survivors.add(node.name)
                node = by_id.get(node.parent_id)

    survivors |= {a.bone for a in result.qc.attachments}
    survivors |= {c.bone for c in result.qc.controllers}
    survivors |= set(result.qc.keepbones)
    return survivors


def _strip_dangling_bone_refs(result: MergeResult) -> list[str]:
    """Remove hitboxes/attachments/controllers pointing at pruned bones."""
    known = _studiomdl_surviving_bones(result)

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
    decimate: float | None = None,
    decimate_overrides: dict[str, float] | None = None,
    pack_parts: bool = True,
    vertex_budget: int = VERTEX_BUDGET,
    keep_hitbox_bones: bool = False,
    keep_animated_bones: bool = False,
    share_hands: bool = True,
    repose_hands: bool = True,
    pool_bones_pass: bool = True,
    bone_target: int = BONE_LIMIT,
    keep_groups: dict[str, set[str] | str] | None = None,
    single_group: bool = True,
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
    prepared: list[ModelInput] = []

    for directory in directories:
        prep = ModelPrep(name=directory.name, directory=directory)
        log(f"--- {directory.name}")

        if sanitise:
            prep.renamed_files = sanitize_directory(directory)
            if prep.renamed_files:
                log(f"    sanitised {len(prep.renamed_files)} filename(s)")

        model = ModelInput.from_directory(directory.name, directory)
        prep.sequences = len(model.qc.sequences)

        prep.renamed_bodygroups = dedupe_bodygroup_names(model.qc)
        if prep.renamed_bodygroups:
            total = sum(count - 1 for count in prep.renamed_bodygroups.values())
            log(f"    renamed {total} duplicate bodygroup name(s): "
                f"{', '.join(sorted(prep.renamed_bodygroups))}")

        if normalise and reference_hand is not None:
            normalisation = normalise_hands(
                model, reference_hand, reference_rigs, texture=hand_texture_name,
                repose=repose_hands,
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
                # This model keeps its own hands, so it must not land in the
                # group the shared-hand collapse rewrites.
                for bodygroup in model.qc.bodygroups:
                    if bodygroup.name == HAND_SMD_KEY:
                        bodygroup.name = f"{HAND_SMD_KEY}_original"
                dedupe_bodygroup_names(model.qc)

        if hand_texture is not None and normalise:
            texture_path = Path(hand_texture)
            if texture_path.exists():
                model.textures[texture_path.name] = texture_path.read_bytes()

        if single_group:
            wanted = (keep_groups or {}).get(model.name, set())
            if wanted == "*":
                keep = {group.name for group in model.qc.bodygroups}
            else:
                keep = set(wanted)
            prep.kept_groups = sorted(keep)
            prep.collapsed_groups = select_weapon_groups(model, keep)
            if prep.collapsed_groups:
                log(f"    collapsed {len(prep.collapsed_groups)} switchable "
                    f"bodygroup(s) to one entry: {', '.join(prep.collapsed_groups)}")
            if keep:
                log(f"    kept switchable: {', '.join(sorted(keep))}")

        model_ratio = (decimate_overrides or {}).get(model.name, decimate)
        if model_ratio is not None and model_ratio < 1.0:
            # Weapon meshes only — the optimised hand is already lean and the
            # animation SMDs carry no geometry.  Do this before packing so the
            # packer sees the reduced counts and needs fewer submodels.
            hand_keys = set(prep.hands.replaced_keys) if (prep.hands and prep.hands.ok) else set()
            tb = ta = 0
            for key, smd in model.smds.items():
                if smd.is_animation or key in hand_keys:
                    continue
                before_t, after_t = decimate_mesh(smd, model_ratio)
                tb += before_t
                ta += after_t
            prep.decimated = (tb, ta)
            if tb:
                log(f"    decimated {tb} -> {ta} triangles ({100 * ta / tb:.0f}%"
                    f"{f', ratio {model_ratio}' if model_ratio != decimate else ''})")

        if pack_parts:
            hand_keys = set(prep.hands.replaced_keys) if (prep.hands and prep.hands.ok) else set()
            groups_before = len(model.qc.bodygroups)
            groups_after = pack_always_on_parts(
                model, budget=vertex_budget, skip_keys=hand_keys,
                keep_switchable=set(prep.kept_groups),
            )
            prep.packed_groups = (groups_before, groups_after)
            if groups_after < groups_before:
                log(f"    packed {groups_before} bodygroups -> {groups_after}")

        if prune:
            removed, before, after = prune_model(
                model, keep_hitbox_bones=keep_hitbox_bones,
                keep_animated_bones=keep_animated_bones,
            )
            prep.pruned_bones = removed
            prep.bones_before, prep.bones_after = before, after
            log(f"    bones {before} -> {after} ({len(removed)} pruned)")
        else:
            names: set[str] = set()
            for smd in model.smds.values():
                names |= {n.name for n in smd.nodes}
            prep.bones_before = prep.bones_after = len(names)

        prepared.append(model)
        result.preps.append(prep)
        result.warnings.extend(f"{prep.name}: {w}" for w in prep.warnings)

    if pool_bones_pass and reference_hand is not None and len(prepared) > 1:
        log("--- pooling weapon bones")
        shared = {node.name for node in reference_hand.nodes}
        plan, moved = pool_bones(prepared, shared, reference_hand=reference_hand,
                                 bone_limit=bone_target)
        if plan is not None:
            result.pool_slots = plan.size
            result.pool_reshaped = sum(plan.reshaped.values())
            log(f"    {plan.size} pooled slots serve {len(prepared)} models "
                f"({sum(len(a) for a in plan.assignments.values())} weapon bones), "
                f"{sum(moved.values())} bones re-parented")
            for prep in result.preps:
                prep.bones_after = len({
                    node.name
                    for model in prepared if model.name == prep.name
                    for smd in model.smds.values() for node in smd.nodes
                })
        else:
            log("    no shared anchor bone found, skipping")

    for model in prepared:
        merger.add_model(model)

    log("--- merging")
    merged = merger.merge(model_name, config=merge_config)
    result.merge = merged

    if share_hands and normalise:
        collapsed, variants = _collapse_shared_hands(merged, hand_keys_by_model)
        if collapsed:
            result.shared_hand = True
            result.hand_variants = variants
            merged.pev_body_map = _recompute_pev_body(
                merged.qc, merged.model_names, merged.bodygroup_indices,
                overrides=collapsed,
            )
            log(f"    hand mesh shared: {variants} distinct "
                f"{'copy' if variants == 1 else 'copies'} for "
                f"{len(hand_keys_by_model)} models")
        else:
            log("    hand meshes differ per model, keeping separate copies")

    dropped = _strip_unused_textures(merged)
    if dropped:
        log(f"    dropped {len(dropped)} unused texture(s)")
    result.warnings.extend(_strip_dangling_bone_refs(merged))
    limit_problems = _check_bodygroup_limits(merged)
    result.warnings.extend(limit_problems)
    result.exceeds_bodygroup_limits = bool(limit_problems)
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
