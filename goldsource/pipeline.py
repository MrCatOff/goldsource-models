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

from goldsource.bonepool import PoolPlan, apply_pool, plan_pool, reparent
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
    _mat4_from_bt,
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


def _rebind_offrig_hand_verts(mesh: SMD, reference_names: set[str]) -> dict[str, str]:
    """
    Move *mesh*'s vertices off any bone not in *reference_names* onto the nearest
    reference bone (by bind-pose position), in place.  Returns ``{off-rig bone:
    reference bone}`` for the bones actually moved.

    Used when a model keeps its **own** hand mesh because the reference matched it
    too poorly to repose (see :func:`normalise_hands`).  The own mesh is correct,
    but it stays rigged to a few bones the reference hand lacks — usually the
    intermediate wrist joint some rigs put between forearm and palm
    (``Bip01_R_Hand`` under ``Bone02`` rather than the forearm), or a stray dummy.
    Left in place those bones give the palm a parent no other model has, so the
    merger renames the whole hand — palm and every finger — apart, and 33 knives
    balloon from 127 to 215 bones.  Re-anchoring their vertices to the reference
    bone they sit on lets pruning fold the extras away, so the model shares the one
    reference hand skeleton while still showing its own correctly-shaped mesh.

    The move is rigid per vertex and only ever picks a *near-coincident* reference
    bone, so the geometry does not shift; only which bone drives it in animation
    changes, from an intermediate that already tracks its neighbour.
    """
    if not mesh.skeleton:
        return {}
    world = world_transforms(mesh, 0)
    reference_pos = {
        name: world[name][:3, 3] for name in reference_names if name in world
    }
    if not reference_pos:
        return {}
    reference_id = {node.name: node.id for node in mesh.nodes if node.name in reference_names}

    remap: dict[int, int] = {}
    moved: dict[str, str] = {}
    used = {vertex.bone_id for triangle in mesh.triangles for vertex in triangle.vertices}
    for node in mesh.nodes:
        if node.name in reference_names or node.id not in used or node.name not in world:
            continue
        position = world[node.name][:3, 3]
        nearest = min(
            reference_pos, key=lambda name: float(np.linalg.norm(reference_pos[name] - position))
        )
        remap[node.id] = reference_id[nearest]
        moved[node.name] = nearest

    for triangle in mesh.triangles:
        for vertex in triangle.vertices:
            if vertex.bone_id in remap:
                vertex.bone_id = remap[vertex.bone_id]
    return moved


def _retarget_fingers_to_reference(
    model: ModelInput, reference_hand: SMD, donor: SMD
) -> int:
    """
    Rewrite every animation's **finger** bones so they drive the shared reference
    hand without stretching, keeping the weapon's finger pose (the grip and its
    motion) but the reference hand's bone *lengths*.  Returns bone-frames rewritten.

    A single shared hand mesh cannot be re-posed per weapon (see
    :func:`_repose_hand_to_model`), so when every weapon shares one fixed hand
    (``--default-hands``) each weapon's animation still carries *its own* finger
    bone offsets frame by frame.  Those offsets are the source rig's bone lengths;
    driving the shared mesh — whose vertices expect the reference hand's lengths —
    with them stretches the fingers, and where a source bone is much longer or
    shorter a triangle explodes into a sliver (v_bhdagger's worst edge reached
    36x).  This is exactly the mismatch re-posing cancels for the mesh; here we
    cancel it on the animation instead.

    Each finger frame keeps the weapon's **absolute** local rotation and only
    swaps in the reference finger's translation ``[A_rot | S_trans]``.  The
    rotation is what curls the finger, so the weapon's grip and its per-frame
    motion carry over verbatim; forcing ``S``'s translation gives the bone the
    length the mesh was bound to, so it moves rigidly and cannot stretch.  A
    *delta* transfer (``S·W⁻¹·A``) was tried first and left the fingers frozen at
    the reference hand's **open** rest: these weapon hands are authored already
    gripping the weapon, so the motion relative to that grip-rest is almost
    nothing and the reference hand's own rest (open) showed through.  Palm and
    forearm are left on the weapon's own animation, so the hand still sits exactly
    where it grips the weapon.  *donor* is unused now but kept for signature
    stability.
    """
    is_finger = lambda name: "finger" in name.lower()
    reference_id = {node.id: node.name for node in reference_hand.nodes}
    if not reference_hand.skeleton:
        return 0
    shared = {
        reference_id[bone.bone_id]: _mat4_from_bt(bone)
        for bone in reference_hand.skeleton[0].bones
        if is_finger(reference_id.get(bone.bone_id, ""))
    }
    if not shared:
        return 0

    count = 0
    for smd in model.smds.values():
        id_to_name = {node.id: node.name for node in smd.nodes}
        for frame in smd.skeleton:
            for bone in frame.bones:
                name = id_to_name.get(bone.bone_id)
                if name not in shared:
                    continue
                if smd.is_animation:
                    local = _mat4_from_bt(bone).copy()   # weapon's grip + finger motion
                    local[:3, 3] = shared[name][:3, 3]   # reference hand's bone length
                else:
                    # A reference (non-animation) mesh: agree with the shared hand's
                    # finger bind so studiomdl's single per-bone bind is consistent.
                    # The weapon's own hand mesh is replaced, so its finger vertices
                    # never render at this bind.
                    local = shared[name]
                solved = _bt_from_mat4(bone.bone_id, local)
                bone.tx, bone.ty, bone.tz = solved.tx, solved.ty, solved.tz
                bone.rx, bone.ry, bone.rz = solved.rx, solved.ry, solved.rz
                count += 1
    return count


def normalise_hands(
    model: ModelInput,
    reference_hand: SMD,
    reference_rigs: list,
    group_pattern: re.Pattern[str] = _HANDS_GROUP_RE,
    texture: str | None = None,
    hands_group_name: str = "hands",
    complete_hands: bool = True,
    repose: bool = True,
    replace_mesh: bool = True,
    vertex_budget: int = VERTEX_BUDGET,
    max_match_cost: float | None = None,
    retarget_fingers: bool = False,
) -> HandNormalisation:
    """
    Replace *model*'s hand mesh(es) with the optimised reference hand.

    The model's own hand bones are renamed onto the reference naming, so its
    animations keep driving the new mesh *and* every model ends up with an
    identically named hand skeleton.  However many hand bodygroups the model
    had, it comes out with exactly one holding the single shared mesh.

    With *replace_mesh* ``False`` the model keeps its **own** hand mesh — only
    the *bones* are renamed onto the common naming, which is what lets the
    skeleton be shared and pooled so many weapons still fit one file.  Each
    weapon then shows its original hands (bone-sharing without hand-sharing).

    *max_match_cost* guards against forcing the shared hand onto a rig it fits
    badly.  The shared mesh is reposed onto the model's own bind, so when the
    reference fingers align poorly (``match.score`` high) the reposed hand comes
    out warped — ``v_bhdagger``/``v_hdagger`` score 3.92 and ``v_knifedragon``
    1005.90 against 0.00 for a matching rig, and their fingers distort.  Above
    the threshold the model keeps its own hand (``replace_mesh`` forced off);
    the bones are still renamed, so it shares the skeleton and merely lands in
    its own hands-bodygroup entry, which ``_collapse_shared_hands`` assigns via
    ``pev_body`` automatically.
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

    # Too poor a fit to repose the shared hand onto without warping the fingers:
    # keep this model's own hand mesh (bones are still renamed just below, so the
    # skeleton is shared regardless).
    if max_match_cost is not None and match.score > max_match_cost:
        replace_mesh = False
        result.kept_own_hand = True

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

    # With a single fixed hand shared by every weapon (no per-model re-pose), bake
    # each weapon's finger animation onto the reference hand's finger lengths so
    # the shared mesh curls without stretching.  Done before the mesh is swapped,
    # while the model's own hand bind is still available as the source pose.
    if retarget_fingers and replace_mesh:
        result.retargeted = _retarget_fingers_to_reference(model, reference_hand, donor)

    if replace_mesh:
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
    else:
        # Keep the model's own hand mesh — the bones are already renamed onto the
        # common naming (above), which is all that is needed for the skeleton to
        # be shared and pooled.
        #
        # Entries WITHIN one $bodygroup are mutually-exclusive alternatives, not
        # pieces drawn together: v_bhdagger's "hands" group offers a male OR a
        # female hand, some rigs a LOD switch.  Concatenating them superimposes
        # two full hands on one skeleton, and since studiomdl gives each bone a
        # single bind the two disagreeing binds shear the fingers into splinters.
        # So keep the first entry of each group and only concatenate ACROSS groups
        # (a rig that splits its left and right hands into two separate groups,
        # which really are drawn together).
        first_per_group: dict[int, str] = {}
        for slot in slots:
            first_per_group.setdefault(id(slot.group), slot.key)
        own_keys = list(first_per_group.values())
        hand_meshes = [model.smds[key] for key in own_keys]
        new_hand = hand_meshes[0] if len(hand_meshes) == 1 else concat_meshes(hand_meshes)

        # Kept because the reference matched too poorly to repose without warping
        # the fingers: the mesh is the model's own (correct) hand, but re-anchor its
        # few off-reference bones so pruning folds them and it still shares the one
        # reference hand skeleton instead of inflating the merged bone count.
        if result.kept_own_hand:
            reference_names = {node.name for node in reference_hand.nodes}
            _rebind_offrig_hand_verts(new_hand, reference_names)

        # The original CSO hands are high-poly; ~1/3 sit just over studiomdl's
        # 2048-vertex-per-submodel cap (the very limit the optimised hand exists
        # to dodge).  A single mesh cannot be split by grouping, so trim only the
        # oversized ones just under the cap — a light, barely-visible reduction
        # that never mixes bones, so the animation is unaffected.
        if unique_vertex_count(new_hand) > vertex_budget:
            decimate_mesh(new_hand, 0.95 * vertex_budget / unique_vertex_count(new_hand))
        result.smd = new_hand

    # Collapse however many hand bodygroups the model has (some split left and
    # right into separate groups) into a single group holding the one mesh.
    retired: set[str] = set()
    for key in keys:
        retired |= {tri.material for tri in model.smds[key].triangles}
        del model.smds[key]

    hand_key = _unique_key(model, HAND_SMD_KEY)
    model.smds[hand_key] = new_hand
    result.replaced_keys.append(hand_key)

    # A kept-own hand shares the reference *names* but may keep its own parent
    # structure (v_bhdagger parents Bip01_L_Finger32 differently), which the merger
    # would rename apart — inflating the skeleton.  Re-anchor its palm and fingers
    # onto the reference hierarchy, in every SMD, so the hand is structurally
    # identical to the shared one.  reparent re-solves each frame, so the own mesh
    # renders unchanged; only the forearm (attached to the model's own arm) and the
    # root are left as the model has them.  Doing it here, locally, keeps these few
    # hand bones out of the global flatten, which would perturb bone pooling.
    if result.kept_own_hand:
        reference_hierarchy = _hierarchy(reference_hand)
        targets = {
            name: parent for name, parent in reference_hierarchy.items()
            if parent is not None and "forearm" not in name.lower()
        }
        for smd in model.smds.values():
            reparent(smd, targets)

        # Keep every reference hand bone the model has, even joints its own mesh
        # puts no geometry on (v_bhdagger's ring finger skips the middle knuckle).
        # Pruning them would re-parent the surviving child onto a grandparent the
        # shared hand keeps, so the merger sees one bone with two parents and renames
        # the whole finger apart.  Held inert, the skeleton stays identical to the
        # shared hand and the model still costs zero extra bones.
        present = {node.name for smd in model.smds.values() for node in smd.nodes}
        for node in reference_hand.nodes:
            if node.name in present and node.name not in model.qc.keepbones:
                model.qc.keepbones.append(node.name)

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


def flatten_conflicting_parents(
    models: list[ModelInput], protect: set[str] | None = None
) -> dict[str, list[str]]:
    """
    Make every bone whose parent disagrees across *models* agree on one parent.

    A bone that appears under two different parents becomes one name with two
    parents at merge time — studiomdl's *illegal parent bone replacement* — so
    the merger renames the copies apart and the shared skeleton inflates (31
    knives needed 214 bones because two rigs hang the forearm and weapon joints
    off different parents than the rest).  Reconciling the parent lets the copies
    collapse into one pooled slot instead.

    Each conflicting bone is moved onto the parent the **most** models already
    give it — falling back to the root only when that parent is not present in
    every model that has the bone — so only the outliers move.  That matters:
    re-anchoring a bone turns its inherited constant channels into time-varying
    ones (bigger, less compressible animation, up against studiomdl's 64 KB
    per-sequence cap), so the fewer bones moved the better.  Rooting everything
    blows v_spknife's idle past 64 KB; matching the majority keeps its weapon
    bones on the hand where they were.  :func:`goldsource.bonepool.reparent`
    re-solves every frame, so motion is unchanged.  Returns ``{model: [bones]}``.

    *protect* names bones that must keep their own parent even when it disagrees
    across models — the hand-mesh bones (palm and fingers).  Re-anchoring those
    warps the shared, reposed hand: ``v_bhdagger`` and ``v_hdagger`` route the
    palm through an extra joint (``Bip01_R_Hand`` under ``Bone02`` rather than
    straight off the forearm), so flattening moved their palms and the fingers
    came out distorted.  The forearm is *not* protected — it sits above the hand
    mesh, is the conflict that actually inflates the skeleton, and reconciling it
    is exactly what lets the knives share one 127-bone rig.
    """
    protect = protect or set()
    hierarchies: dict[str, dict[str, str | None]] = {}
    for model in models:
        hierarchy: dict[str, str | None] = {}
        for smd in model.smds.values():
            for bone, parent in _hierarchy(smd).items():
                hierarchy.setdefault(bone, parent)
        authority = _authority(model)
        if authority is not None:
            hierarchy.update(_hierarchy(authority))
        hierarchies[model.name] = hierarchy

    bones_in: dict[str, set[str]] = {name: set(h) for name, h in hierarchies.items()}
    parents_of: dict[str, list[str | None]] = {}
    for hierarchy in hierarchies.values():
        for bone, parent in hierarchy.items():
            parents_of.setdefault(bone, []).append(parent)

    conflicting = {
        bone for bone, seen in parents_of.items()
        if len(set(seen)) > 1 and bone not in protect
    }
    if not conflicting:
        return {}

    # Target = the most common parent that every holder can actually adopt
    # (the root is always adoptable), so the fewest bones re-anchor.
    target: dict[str, str | None] = {}
    for bone in conflicting:
        holders = [name for name, has in bones_in.items() if bone in has]
        counts: dict[str | None, int] = {}
        for name in holders:
            parent = hierarchies[name][bone]
            counts[parent] = counts.get(parent, 0) + 1
        ordered = sorted(counts, key=lambda p: (-counts[p], p is None, str(p)))
        chosen: str | None = None
        for parent in ordered:
            if parent is None or all(parent in bones_in[name] for name in holders):
                chosen = parent
                break
        target[bone] = chosen

    moved: dict[str, list[str]] = {}
    for model in models:
        hierarchy = hierarchies[model.name]
        targets = {
            bone: target[bone]
            for bone in conflicting
            if bone in hierarchy and hierarchy[bone] != target[bone]
        }
        if not targets:
            continue
        changed: set[str] = set()
        for smd in model.smds.values():
            changed.update(reparent(smd, targets))
        if changed:
            moved[model.name] = sorted(changed)
    return moved


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


def _apply_hand_variants(
    merged: MergeResult,
    variants: list[tuple[str | Path, str | Path]],
    hands_group_name: str = HAND_SMD_KEY,
) -> tuple[int, int]:
    """
    Replace the shared hands bodygroup with a **fixed set** of hand meshes (e.g.
    male + female), each selectable independently of the weapon.

    Every model's hand bones have already been renamed onto the common naming, so
    all these meshes drive the same shared hand skeleton.  The hands bodypart
    therefore becomes a free choice orthogonal to the weapon: index 0 is the
    first variant, 1 the second, and so on, the same for every weapon.  Returns
    ``(variant_count, hands_stride)`` — add ``hands_stride`` to a weapon's
    ``pev_body`` to move from one variant to the next.
    """
    from goldsource.merger import _inject_universal_root

    groups = [bg for bg in merged.qc.bodygroups if bg.name == hands_group_name]
    if not groups:
        return 0, 0

    # Drop whatever meshes the shared-hand collapse left in the hands group.
    for group in groups:
        for entry in group.entries:
            if entry.smd:
                merged.smds.pop(entry.smd, None)

    variant_keys: list[str] = []
    for index, (smd_path, texture_path) in enumerate(variants):
        mesh = SMD.from_file(smd_path)
        texture_name = Path(texture_path).name
        for triangle in mesh.triangles:
            triangle.material = texture_name
        mesh = _inject_universal_root(mesh)
        key = SHARED_HAND_KEY if index == 0 else f"{SHARED_HAND_KEY}_{index + 1}"
        merged.smds[key] = mesh
        variant_keys.append(key)
        if Path(texture_path).exists():
            merged.textures[texture_name] = Path(texture_path).read_bytes()

    entries = [BodyGroupEntry(smd=key) for key in variant_keys]
    for group in groups:
        group.entries = list(entries)

    # Every weapon defaults to the first variant (index 0); recompute from there.
    override = {hands_group_name: {name: 0 for name in merged.model_names}}
    merged.pev_body_map = _recompute_pev_body(
        merged.qc, merged.model_names, merged.bodygroup_indices, overrides=override
    )

    # Stride of the hands group = product of entry counts of every prior group.
    stride = 1
    for group in merged.qc.bodygroups:
        if group.name == hands_group_name:
            break
        stride *= max(1, len(group.entries))
    return len(variant_keys), stride


def _unify_skeleton(merged: MergeResult) -> int:
    """
    Give every output SMD the **full merged skeleton**, like a hand-authored
    single model (``v_model4_30`` writes all 126 bones into all 236 of its SMDs).

    studiomdl builds the compiled skeleton from the union of the reference meshes
    and only needs each SMD to name the bones it actually uses, so by default the
    merger leaves every SMD carrying just its own weapon's subset.  That is leaner
    but lets two SMDs disagree about a bone — different subsets, and a pooled bone
    that one mesh omits can default to a different parent in another, which
    studiomdl rejects as *illegal parent bone replacement*.  Writing one
    consistent skeleton into every SMD removes that whole class of failure.

    The canonical parent and bind of each bone are taken from the reference
    meshes; missing bones are grafted into each SMD static at that bind (they are
    never drawn there, only present so the skeleton agrees).  Returns the number
    of (SMD, bone) grafts performed.
    """
    canon_parent: dict[str, str | None] = {}
    canon_local: dict[str, tuple[float, ...]] = {}
    for raw in _ref_smd_names(merged.qc):
        key = _norm_path(raw)
        smd = merged.smds.get(key) or next(
            (s for k, s in merged.smds.items() if k.split("/")[-1] == key.split("/")[-1]), None)
        if smd is None or not smd.skeleton:
            continue
        by_id = {n.id: n.name for n in smd.nodes}
        frame0 = {b.bone_id: b for b in smd.skeleton[0].bones}
        for node in smd.nodes:
            if node.name in canon_parent:
                continue
            canon_parent[node.name] = by_id[node.parent_id] if node.parent_id >= 0 else None
            b = frame0.get(node.id)
            canon_local[node.name] = ((b.tx, b.ty, b.tz, b.rx, b.ry, b.rz) if b
                                      else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen or name not in canon_parent:
            return
        seen.add(name)
        parent = canon_parent[name]
        if parent:
            visit(parent)
        order.append(name)

    for name in list(canon_parent):
        visit(name)

    grafts = 0
    for smd in merged.smds.values():
        if not smd.skeleton:
            continue
        name_to_id = {n.name: n.id for n in smd.nodes}
        next_id = max((n.id for n in smd.nodes), default=-1) + 1
        added = 0
        for name in order:
            if name in name_to_id:
                continue
            parent = canon_parent[name]
            smd.nodes.append(Node(id=next_id, name=name,
                                  parent_id=name_to_id[parent] if parent else -1))
            name_to_id[name] = next_id
            tx, ty, tz, rx, ry, rz = canon_local[name]
            for frame in smd.skeleton:
                frame.bones.append(BoneTransform(
                    bone_id=next_id, tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz))
            next_id += 1
            added += 1
        if added:
            renumber(smd)
            grafts += added

        # Force every bone onto the canonical parent, re-solving each frame so the
        # motion is unchanged, so no two SMDs disagree about a shared bone's
        # parent.  A batch that already compiles is a no-op here (its reference
        # and animations already agree).  This does not rescue a bone the merger
        # renamed *apart* into two names (v_laserminigun) — that is a naming
        # conflict upstream of the skeleton, still handled with --no-pool-bones.
        by_id = {n.id: n.name for n in smd.nodes}
        new_parents = {}
        for node in smd.nodes:
            current = by_id[node.parent_id] if node.parent_id >= 0 else None
            canonical = canon_parent.get(node.name, current)
            if canonical != current:
                new_parents[node.name] = canonical
        if new_parents:
            reparent(smd, new_parents)
            renumber(smd)
    return grafts


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


def _downscale_textures(result: MergeResult, max_size: int) -> tuple[int, int, int]:
    """
    Shrink every texture whose longest side exceeds *max_size* to fit, in place.

    GoldSource stores textures as 8-bit palettized bitmaps, so the only way to
    make a merged ``.mdl`` smaller is to lower texture *resolution* — pixel count
    is the file.  Model UVs are normalised, so the on-screen mapping is
    unaffected by the image's pixel dimensions; a 512² skin at 256² simply
    samples a smaller bitmap.

    Each texture is decoded through its palette to RGB, resized with LANCZOS,
    then re-quantised to a fresh 256-colour palette.  Masked textures (transparent
    ``{`` skins, or any in a ``masked`` ``$texrendermode``) are left untouched —
    their transparency depends on an exact palette index that requantising would
    destroy.

    Returns ``(textures_resized, bytes_before, bytes_after)``.
    """
    from io import BytesIO
    from PIL import Image

    skip = {m.texture.lower() for m in result.qc.texturemodes if m.mode == "masked"}
    before = after = resized = 0
    for name, data in list(result.textures.items()):
        before += len(data)
        if name.startswith("{") or name.lower() in skip or data[:2] != b"BM":
            after += len(data)
            continue
        img = Image.open(BytesIO(data))
        w, h = img.size
        longest = max(w, h)
        if longest <= max_size:
            after += len(data)
            continue
        scale = max_size / longest
        # Snap each side to a multiple of 16 (the safe GoldSource texture step),
        # never upscaling.  Aspect drift is invisible: UVs are normalised.
        def snap(px: int) -> int:
            return max(16, min(px, round(px * scale / 16) * 16))
        new = img.convert("RGB").resize((snap(w), snap(h)), Image.LANCZOS)
        out = BytesIO()
        new.quantize(colors=256, method=Image.MEDIANCUT).save(out, format="BMP")
        result.textures[name] = out.getvalue()
        after += len(result.textures[name])
        resized += 1
    return resized, before, after


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


def _shorten_output_paths(merged: MergeResult, limit: int = 60) -> dict[str, str]:
    """
    Shorten output SMD paths so studiomdl's fixed path buffer (~64 chars) cannot
    truncate them.

    The merge nests animations as ``<model>/<model>_anims/<smd>`` — the doubled
    model name alone is ~45 chars — so a long weapon name pushes a ``$sequence``
    source path past studiomdl's limit and it silently drops a character
    (``…stab_miss`` becomes ``…stab_mis`` "doesn't exist").  This collapses the
    redundant ``<model>_anims`` level to ``a`` (keeping the readable model name);
    if any path is still too long it aliases the model directory to ``m<i>``.
    Rewrites the SMD keys and every QC reference in place.  Returns the key remap
    (empty when nothing needed shortening).
    """
    keys = list(merged.smds)
    out_len = lambda key: len(key) + len(".smd")
    if all(out_len(key) <= limit for key in keys):
        return {}

    remap: dict[str, str] = {}
    for key in keys:
        parts = key.split("/")
        if len(parts) >= 3 and parts[1].endswith("_anims"):
            remap[key] = parts[0] + "/a/" + "/".join(parts[2:])
        else:
            remap[key] = key

    if any(out_len(value) > limit for value in remap.values()):
        tops: list[str] = []
        for key in keys:
            top = remap[key].split("/", 1)[0]
            if top != "_shared" and top not in tops:
                tops.append(top)
        alias = {top: f"m{index}" for index, top in enumerate(tops)}
        for key in keys:
            value = remap[key]
            top = value.split("/", 1)[0]
            if top in alias:
                remap[key] = alias[top] + value[len(top):]

    # Guarantee the collapsed keys stay unique.
    seen: set[str] = set()
    for key in keys:
        value = remap[key]
        base, counter = value, 2
        while value in seen:
            value = f"{base}_{counter}"
            counter += 1
        remap[key], _ = value, seen.add(value)

    if all(remap[key] == key for key in keys):
        return {}

    merged.smds = {remap[key]: smd for key, smd in merged.smds.items()}
    fix = lambda path: remap.get(path, path)
    if merged.qc.body is not None and merged.qc.body.smd:
        merged.qc.body.smd = fix(merged.qc.body.smd)
    for bodygroup in merged.qc.bodygroups:
        for entry in bodygroup.entries:
            if entry.smd:
                entry.smd = fix(entry.smd)
    for sequence in merged.qc.sequences:
        sequence.smd_paths = [fix(path) for path in sequence.smd_paths]
    return remap


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
    keep_hand_mesh: bool = False,
    hand_match_max_cost: float | None = 1.0,
    retarget_fingers: bool = False,
    hand_variants: list[tuple[str | Path, str | Path]] | None = None,
    unify_skeleton: bool = True,
    pool_bones_pass: bool = True,
    flatten_weapon_bones: bool = False,
    short_paths: bool = True,
    bone_target: int = BONE_LIMIT,
    keep_groups: dict[str, set[str] | str] | None = None,
    single_group: bool = True,
    sanitise: bool = True,
    max_texture_size: int | None = None,
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
                model, reference_hand, reference_rigs,
                texture=None if keep_hand_mesh else hand_texture_name,
                repose=repose_hands, replace_mesh=not keep_hand_mesh,
                vertex_budget=vertex_budget,
                max_match_cost=(
                    hand_match_max_cost
                    if not keep_hand_mesh and (hand_match_max_cost or 0) > 0
                    else None
                ),
                retarget_fingers=retarget_fingers,
            )
            prep.hands = normalisation
            if normalisation.ok:
                pairs = ", ".join(f"{a}->{b}" for a, b in normalisation.pairs)
                kept = " — kept own hand (match too poor to share)" \
                    if normalisation.kept_own_hand else ""
                retgt = f", retargeted {normalisation.retargeted} finger-frames" \
                    if normalisation.retargeted else ""
                log(f"    hands rebound ({pairs}), match cost "
                    f"{normalisation.score:.2f}{kept}{retgt}")
                if normalisation.kept_own_hand:
                    prep.warnings.append(
                        f"hand match cost {normalisation.score:.2f} over "
                        f"{hand_match_max_cost}; kept own hand mesh to avoid distortion"
                    )
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

    if flatten_weapon_bones and len(prepared) > 1:
        log("--- flattening conflicting-parent bones to root")
        # Never re-anchor a hand-mesh bone (palm or finger) here: fingers hang off
        # the palm and the shared rig is reposed onto each model's bind, so a global
        # flatten of them perturbs pooling and can push an unrelated weapon sequence
        # past studiomdl's 64 KB cap.  Kept-own hands are instead normalised onto the
        # reference hand hierarchy locally (in normalise_hands), so no hand bone ever
        # conflicts and none of this reaches flatten.  Only the forearm — above the
        # mesh, the real inflation source — stays movable.
        protect_hand = {
            node.name for node in (reference_hand.nodes if reference_hand else [])
            if "forearm" not in node.name.lower()
        }
        flattened = flatten_conflicting_parents(prepared, protect=protect_hand)
        if flattened:
            total = sum(len(bones) for bones in flattened.values())
            log(f"    re-parented {total} bone(s) across {len(flattened)} model(s) "
                f"to remove parent conflicts")
            for prep in result.preps:
                prep.bones_after = len({
                    node.name
                    for model in prepared if model.name == prep.name
                    for smd in model.smds.values() for node in smd.nodes
                })
        else:
            log("    no parent conflicts found")

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

    if hand_variants and normalise:
        count, stride = _apply_hand_variants(merged, hand_variants)
        if count:
            result.shared_hand = True
            result.hand_variants = count
            merged.hand_variant_stride = stride
            merged.hand_variant_names = [Path(smd).stem for smd, _tex in hand_variants]
            names = ", ".join(Path(smd).stem for smd, _tex in hand_variants)
            log(f"    hands bodypart: {count} shared variants ({names}); "
                f"+{stride} to pev_body switches variant")
            result.warnings.append(
                f"hands variants: index 0 = {Path(hand_variants[0][0]).stem}; "
                f"add {stride} to a weapon's pev_body per further variant")

    if unify_skeleton:
        grafts = _unify_skeleton(merged)
        if grafts:
            log(f"    unified skeleton: every SMD now carries the full bone list "
                f"({grafts} bones grafted across meshes)")

    dropped = _strip_unused_textures(merged)
    if dropped:
        log(f"    dropped {len(dropped)} unused texture(s)")
    if max_texture_size:
        n, before, after = _downscale_textures(merged, max_texture_size)
        if n:
            log(f"    downscaled {n} texture(s) to <={max_texture_size}px: "
                f"{before // 1024} KB -> {after // 1024} KB "
                f"(saved {(before - after) // 1024} KB)")
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

    if short_paths:
        remap = _shorten_output_paths(merged)
        if remap:
            log(f"    shortened {len(remap)} output path(s) to fit studiomdl's "
                f"~64-char limit")

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
