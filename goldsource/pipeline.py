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
from copy import deepcopy
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
from goldsource.qc import QC, BodyGroup, BodyGroupEntry, Sequence
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
from goldsource.smd import SMD, BoneTransform, Node, SkeletonFrame


# Per-finger FABRIK retarget (TS.md Stage 3).  OFF: it was measured to regress
# tightly-gripped pistols — a donor finger too long for the grip gap bulges into
# the weapon regardless of where its tip lands (v_anaconda tip penetration 8x
# worse), and some clipping is knuckle-placement, not curl.  Left in place as the
# groundwork for a penetration-aware Stage 4 solve; the default retarget keeps
# the known-good rotation-copy.  See _retarget_fingers_to_reference.
_FINGER_IK = False

# Bounded constant wrist offset (TS.md Stage 4 §8.3): lift the shared hand out of
# the weapon where the donor hand's placement buries it (v_usp knuckles ~50u deep).
# Bone lengths untouched (a palm translation only); ON by default.
_WRIST_OFFSET = True

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


def sequence_count(directory: Path) -> int:
    """How many ``$sequence`` clips the model in *directory* declares."""
    qcs = list(Path(directory).glob("*.qc"))
    return len(QC.from_file(qcs[0]).sequences) if qcs else 0


def plan_sequence_parts(
    inputs: list[str | Path],
    exclude: list[str] | None = None,
    max_sequences: int | None = None,
) -> list[list[Path]]:
    """
    Split the discovered model directories into groups whose combined
    ``$sequence`` count each stays within *max_sequences*.

    The merged model's sequence total is the sum of its models' clips, and some
    engines cap how many a view model may hold; over that cap the extra sequences
    are unusable, so the build has to be spread across several models
    (``v_x_part_1`` …).  Models keep their given order and each is placed whole —
    a single model with more clips than *max_sequences* still gets its own part
    (it cannot be divided without breaking that weapon).  Returns one group when
    no split is needed (``max_sequences`` unset or the total already fits).
    """
    excluded = {name.lower() for name in (exclude or [])}
    directories: list[Path] = []
    for item in inputs:
        for directory in discover_models(item):
            if directory.name.lower() in excluded or directory in directories:
                continue
            directories.append(directory)

    if not max_sequences or max_sequences <= 0:
        return [directories]

    groups: list[list[Path]] = []
    current: list[Path] = []
    running = 0
    for directory in directories:
        count = sequence_count(directory)
        if current and running + count > max_sequences:
            groups.append(current)
            current, running = [], 0
        current.append(directory)
        running += count
    if current:
        groups.append(current)
    return groups or [directories]


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


def strip_forearm(smd: SMD, elbow_margin: float = 0.0) -> int:
    """
    Cut the **above-the-elbow** part off the forearm in a hand *smd*, in place,
    keeping the forearm proper (elbow→wrist) and the bones.  Returns triangles
    dropped.

    The reference hands' forearm mesh runs well past the elbow toward the shoulder
    (~43% of the forearm verts sit on the far side of the elbow).  The forearm
    itself belongs in view, but that above-elbow stub pokes beyond the screen edge
    when an animation extends the arm (v_axe's hit).  Each ``*_Forearm`` vertex is
    projected onto the elbow→wrist axis (elbow at 0, wrist at the hand child); a
    triangle whose forearm verts all fall behind the elbow (projection <
    ``elbow_margin``; use a small negative value to keep the elbow rounded) is
    dropped.  Triangles straddling the elbow, or anchored by a hand vertex, stay,
    so the forearm keeps its connection to the hand.  The bones are untouched.
    """
    world = world_transforms(smd, 0)
    by_id = {node.id: node for node in smd.nodes}
    # Per forearm bone: (elbow origin, unit axis toward its hand child).
    axes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for node in smd.nodes:
        if "forearm" not in node.name.lower() or node.name not in world:
            continue
        elbow = world[node.name][:3, 3]
        hand = next((world[c.name] for c in smd.nodes
                     if c.parent_id == node.id and "hand" in c.name.lower()
                     and c.name in world), None)
        if hand is None:
            continue
        axis = hand[:3, 3] - elbow
        length = float(np.linalg.norm(axis))
        if length > 1e-6:
            axes[node.id] = (elbow, axis / length)

    def above_elbow(vertex) -> bool:
        frame = axes.get(vertex.bone_id)
        if frame is None:
            return False
        origin, unit = frame
        return float(np.dot(np.array([vertex.x, vertex.y, vertex.z]) - origin, unit)) < elbow_margin

    kept = []
    for triangle in smd.triangles:
        forearm_verts = [v for v in triangle.vertices if v.bone_id in axes]
        if forearm_verts and all(above_elbow(v) for v in forearm_verts):
            continue
        kept.append(triangle)
    dropped = len(smd.triangles) - len(kept)
    smd.triangles = kept
    return dropped


def _posed_world_verts(mesh: SMD, bind: dict, pose: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    World-space position and normal of *mesh*'s vertices under *pose*, deduplicated.

    Mesh vertices are stored per triangle-corner, so most positions repeat; collapsing
    the duplicates cuts the nearest-neighbour search several-fold with no loss.
    """
    id_to_name = {node.id: node.name for node in mesh.nodes}
    xform = {name: pose[name] @ np.linalg.inv(bind[name]) for name in bind if name in pose}
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for triangle in mesh.triangles:
        for vertex in triangle.vertices:
            matrix = xform.get(id_to_name.get(vertex.bone_id))
            if matrix is None:
                continue
            positions.append((matrix @ np.array([vertex.x, vertex.y, vertex.z, 1.0]))[:3])
            normals.append(matrix[:3, :3] @ np.array([vertex.nx, vertex.ny, vertex.nz]))
    if not positions:
        return np.empty((0, 3)), np.empty((0, 3))
    P = np.array(positions)
    N = np.array(normals)
    _, keep = np.unique(P.round(2), axis=0, return_index=True)
    return P[keep], N[keep]


def _mean_penetration(hand: SMD, hand_bind: dict, weapon_P: dict, weapon_N: dict,
                      anim: SMD, frames: list[int]) -> float:
    """
    Penetration depth per contact vertex — how far *hand*'s vertices sit *inside*
    the weapon, averaged over the hand vertices that are near it (over *frames*).

    A hand vertex is inside when it is near the weapon surface and on the negative
    side of that surface's outward normal.  Normalising by the count of near ("in
    contact") vertices — not the raw sum — makes the score independent of how many
    vertices each hand mesh happens to have, so own and shared hands compare fairly.
    High values mean the hand clips through the weapon (passes through the gun).
    """
    penetration = 0.0
    contact = 0
    for frame in frames:
        pose = world_transforms(anim, frame)
        hand_P, _ = _posed_world_verts(hand, hand_bind, pose)
        if len(hand_P) == 0 or frame not in weapon_P:
            continue
        wp, wn = weapon_P[frame], weapon_N[frame]
        index = np.empty(len(hand_P), dtype=int)
        distance = np.empty(len(hand_P))
        for i in range(0, len(hand_P), 256):
            batch = hand_P[i:i + 256]
            d = np.linalg.norm(batch[:, None, :] - wp[None, :, :], axis=2)
            index[i:i + 256] = d.argmin(1)
            distance[i:i + 256] = d.min(1)
        normal = wn[index]
        normal = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9)
        signed = np.einsum("ij,ij->i", hand_P - wp[index], normal)
        near = distance < 3.0
        inside = near & (signed < -0.15)
        penetration += float((-signed[inside]).sum())
        contact += int(near.sum())
    return 1000.0 * penetration / max(contact, 1)


def _shared_hand_excess_penetration(
    model: ModelInput, reference_hand: SMD, reference_rigs: list, frames: int = 4
) -> float:
    """
    How much MORE the retargeted shared hand clips into the weapon than the model's
    own hand does (averaged over animation frames).  Returns 0 when it cannot be
    measured.

    The bind-pose match cost only checks the *rest* fit; a rig can match at rest yet
    have the shared hand pass through the weapon once the animation curls the
    fingers (v_kingcobra, v_bloodhunter).  This poses the weapon and each hand over
    the idle and compares their penetration, giving a dynamic "does it actually grip
    the gun" score the cost check misses.  Computed on a copy, so the model is left
    untouched for the real normalisation.
    """
    slots = _hand_slots(model, _HANDS_GROUP_RE)
    if not slots:
        return 0.0
    hand_keys = {slot.key for slot in slots}
    own = model.smds[slots[0].key]
    # The weapon is everything that is not a hand mesh; some weapons are split into
    # several meshes (v_bloodhunter), so use them all — the hand must not clip any.
    weapon_meshes = [
        smd for key, smd in model.smds.items()
        if not smd.is_animation and key not in hand_keys and smd.triangles
    ]
    anim = next((smd for key, smd in model.smds.items()
                 if smd.is_animation and "idle" in key.lower() and smd.skeleton), None)
    if anim is None:
        anim = next((smd for smd in model.smds.values() if smd.is_animation and smd.skeleton), None)
    if not weapon_meshes or anim is None:
        return 0.0

    count = min(len(anim.skeleton), 24)
    frame_idx = list(range(0, count, max(1, count // frames)))
    weapon_binds = [(mesh, world_transforms(mesh, 0)) for mesh in weapon_meshes]
    weapon_P: dict[int, np.ndarray] = {}
    weapon_N: dict[int, np.ndarray] = {}
    for frame in frame_idx:
        pose = world_transforms(anim, frame)
        parts_p: list[np.ndarray] = []
        parts_n: list[np.ndarray] = []
        for mesh, bind in weapon_binds:
            p, n = _posed_world_verts(mesh, bind, pose)
            if len(p):
                parts_p.append(p)
                parts_n.append(n)
        if parts_p:
            weapon_P[frame] = np.vstack(parts_p)
            weapon_N[frame] = np.vstack(parts_n)
    if not weapon_P:
        return 0.0

    own_pen = _mean_penetration(own, world_transforms(own, 0), weapon_P, weapon_N, anim, frame_idx)

    shared_model = deepcopy(model)
    norm = normalise_hands(shared_model, reference_hand, reference_rigs, texture=None,
                           repose=False, replace_mesh=True, max_match_cost=None,
                           retarget_fingers=True)
    if not norm.ok or not norm.replaced_keys:
        return 0.0
    shared_anim = next((smd for key, smd in shared_model.smds.items()
                        if smd.is_animation and "idle" in key.lower() and smd.skeleton), None)
    if shared_anim is None:
        shared_anim = next((smd for smd in shared_model.smds.values()
                            if smd.is_animation and smd.skeleton), None)
    shared_hand = shared_model.smds[norm.replaced_keys[0]]
    shared_pen = _mean_penetration(shared_hand, world_transforms(shared_hand, 0),
                                   weapon_P, weapon_N, shared_anim, frame_idx)
    return shared_pen - own_pen


def _swing_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minimal 3×3 rotation taking direction *a* onto direction *b* (a pure swing,
    no roll about the axis).  Used to redirect a bone onto the IK-solved bone
    direction while leaving the weapon's roll — which side the nail faces —
    untouched.
    """
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3)
        # Antiparallel: 180° about any axis perpendicular to a.
        perp = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, np.array([0.0, 1.0, 0.0]))
        perp = perp / np.linalg.norm(perp)
        k = np.array([[0.0, -perp[2], perp[1]],
                      [perp[2], 0.0, -perp[0]],
                      [-perp[1], perp[0], 0.0]])
        return np.eye(3) + 2.0 * (k @ k)          # Rodrigues at θ=π
    k = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return np.eye(3) + k + (k @ k) * (1.0 / (1.0 + c))


def _fabrik_solve(base: np.ndarray, lengths: list[float], target: np.ndarray,
                  init: list[np.ndarray], iters: int = 16, tol: float = 1e-3
                  ) -> list[np.ndarray]:
    """
    FABRIK: solve joint positions of a chain of fixed *lengths* rooted at *base*
    so the end reaches *target*, seeded from *init* (``len(lengths)+1`` points).

    Fixed segment length is FABRIK's defining invariant, which is exactly the
    dimension guarantee we need — a donor bone length is never touched, only the
    joint angles change.  When the target is out of reach the chain straightens
    toward it (donor finger shorter than the original needs); when it is in reach
    the longer donor finger simply curls further, which is the point.
    """
    pts = [p.astype(float).copy() for p in init]
    n = len(lengths)
    reach = float(sum(lengths))
    to_target = target - base
    if float(np.linalg.norm(to_target)) >= reach:
        d = to_target / (float(np.linalg.norm(to_target)) + 1e-12)
        acc = base.astype(float).copy()
        pts[0] = acc.copy()
        for i in range(n):
            acc = acc + d * lengths[i]
            pts[i + 1] = acc.copy()
        return pts
    for _ in range(iters):
        pts[n] = target.astype(float).copy()
        for i in range(n - 1, -1, -1):          # backward reaching
            d = pts[i] - pts[i + 1]
            d /= (float(np.linalg.norm(d)) + 1e-12)
            pts[i] = pts[i + 1] + d * lengths[i]
        pts[0] = base.astype(float).copy()
        for i in range(n):                       # forward reaching
            d = pts[i + 1] - pts[i]
            d /= (float(np.linalg.norm(d)) + 1e-12)
            pts[i + 1] = pts[i] + d * lengths[i]
        if float(np.linalg.norm(pts[n] - target)) < tol:
            break
    return pts


def _leaf_tip_local(mesh: SMD, leaf_id: int | None, bind_world: np.ndarray | None
                    ) -> np.ndarray | None:
    """
    Bone-local offset of a finger's **distal tip** — the mesh vertex bound to the
    leaf phalanx that sits farthest from its joint, expressed in the leaf bone's
    bind frame.  SMD reference vertices live in reference-pose *world* space, so
    the bind world inverse brings them local.  ``None`` if nothing is bound to
    the leaf (then that finger falls back to the plain rotation swap).
    """
    if leaf_id is None or bind_world is None:
        return None
    inv = np.linalg.inv(bind_world)
    best: np.ndarray | None = None
    best_d = -1.0
    for triangle in mesh.triangles:
        for v in triangle.vertices:
            if v.bone_id != leaf_id:
                continue
            local = inv @ np.array([v.x, v.y, v.z, 1.0])
            d = float(np.linalg.norm(local[:3]))
            if d > best_d:
                best_d, best = d, local[:3]
    return best


def _solve_frame_fingers(plans: list[dict], shared: dict[str, np.ndarray],
                         world: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    One animation frame: return ``{finger bone name → new 3×3 LOCAL rotation}``
    that curls each donor-length finger so its tip lands on the **original**
    fingertip instead of overshooting into the weapon (see
    :func:`_retarget_fingers_to_reference`).
    """
    out: dict[str, np.ndarray] = {}
    for plan in plans:
        palm = plan["palm"]
        bones = plan["bones"]
        if palm not in world or not all(b in world for b in bones):
            continue
        palm_w = world[palm]
        seg = plan["seg"]
        axis = plan["axis"]
        # Base: the knuckle placed at the DONOR offset from the (unchanged) palm.
        base = (palm_w @ np.append(shared[bones[0]][:3, 3], 1.0))[:3]
        # Current world direction of each bone's donor axis under the weapon's
        # own rotation — the roll to keep, and the (overshooting) seed pose.
        cur_dir = [world[name][:3, :3] @ axis[i] for i, name in enumerate(bones)]
        init = [base.copy()]
        acc = base.copy()
        for i in range(len(bones)):
            acc = acc + seg[i] * (cur_dir[i] / (float(np.linalg.norm(cur_dir[i])) + 1e-12))
            init.append(acc.copy())
        # Goal: where the original finger tip actually was this frame — the point
        # the artist placed on the weapon surface.
        goal = (world[bones[-1]] @ np.append(plan["own_tip"], 1.0))[:3]
        pts = _fabrik_solve(base, seg, goal, init)
        # Positions → per-bone local rotation: swing each bone's original world
        # rotation onto the solved direction, then express it relative to the new
        # parent.  Forcing the donor translation on write reproduces these exact
        # positions, so bone lengths stay byte-identical.
        parent_rot = palm_w[:3, :3]
        for i, name in enumerate(bones):
            world_rot = _swing_rotation(cur_dir[i], pts[i + 1] - pts[i]) @ world[name][:3, :3]
            out[name] = parent_rot.T @ world_rot
            parent_rot = world_rot
    return out


def _retarget_fingers_to_reference(
    model: ModelInput, reference_hand: SMD, reference_rigs: list, donor: SMD
) -> int:
    """
    Rewrite every animation's **finger** bones so they drive the shared reference
    hand without stretching *and* without clipping into the weapon.  Returns
    bone-frames rewritten.

    A single shared hand mesh cannot be re-posed per weapon (see
    :func:`_repose_hand_to_model`), so when every weapon shares one fixed hand
    (``--default-hands``) each weapon's animation still carries *its own* finger
    bone offsets frame by frame.  Those offsets are the source rig's bone lengths;
    driving the shared mesh — whose vertices expect the reference hand's lengths —
    with them stretches the fingers, and where a source bone is much longer or
    shorter a triangle explodes into a sliver (v_bhdagger's worst edge reached
    36x).  So the reference finger's translation is forced in ``[R | S_trans]``,
    giving the bone exactly the length the mesh was bound to.

    Keeping the weapon's rotation verbatim (the old behaviour) leaves the grip
    off by the donor/original length difference: the donor fingers run ~5–9%
    longer, so the same curl overshoots the original fingertip and sinks the tip
    into the gun (v_deagle/v_usp visibly, v_anaconda barely).  So the rotation is
    re-solved per finger by **FABRIK** (:func:`_solve_frame_fingers`) with the
    goal set to the *original* hand's fingertip world position at that frame — the
    contact point the artist authored — so a longer finger curls further and lands
    on the surface instead of through it.  Only rotations change; the forced donor
    translation keeps every bone length byte-identical, so the compile-gate drift
    stays exactly 0.  Palm and forearm are left on the weapon's own animation, so
    the hand still sits exactly where it grips the weapon.
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

    # One FABRIK plan per finger chain: donor segment lengths + the axis toward
    # each child (all in reference dimensions), plus the model's own distal-tip
    # offset that fixes where the tip must land.
    ref_bind = world_transforms(reference_hand, 0)
    ref_id = {n.name: n.id for n in reference_hand.nodes}
    own_bind = world_transforms(donor, 0)
    own_id = {n.name: n.id for n in donor.nodes}
    plans: list[dict] = []
    for rig in (reference_rigs or []) if _FINGER_IK else []:
        for chain in rig.chains:
            if not all(b in shared for b in chain):
                continue
            leaf = chain[-1]
            ref_tip = _leaf_tip_local(reference_hand, ref_id.get(leaf), ref_bind.get(leaf))
            own_tip = _leaf_tip_local(donor, own_id.get(leaf), own_bind.get(leaf))
            if ref_tip is None or own_tip is None:
                continue                      # no distal geometry → plain swap
            seg: list[float] = []
            axis: list[np.ndarray] = []
            for i, name in enumerate(chain):
                off = shared[chain[i + 1]][:3, 3] if i + 1 < len(chain) else ref_tip
                length = max(float(np.linalg.norm(off)), 1e-6)
                seg.append(length)
                axis.append(off / length)
            plans.append({"palm": rig.hand, "bones": list(chain),
                          "seg": seg, "axis": axis, "own_tip": own_tip})

    count = 0
    for smd in model.smds.values():
        id_to_name = {node.id: node.name for node in smd.nodes}
        if smd.is_animation:
            for frame_index, frame in enumerate(smd.skeleton):
                world = world_transforms(smd, frame_index)   # weapon pose, pre-rewrite
                new_rot = _solve_frame_fingers(plans, shared, world)
                for bone in frame.bones:
                    name = id_to_name.get(bone.bone_id)
                    if name not in shared:
                        continue
                    local = _mat4_from_bt(bone).copy()       # weapon grip + motion
                    if name in new_rot:
                        local[:3, :3] = new_rot[name]        # FABRIK-corrected curl
                    local[:3, 3] = shared[name][:3, 3]       # reference length (locked)
                    solved = _bt_from_mat4(bone.bone_id, local)
                    bone.tx, bone.ty, bone.tz = solved.tx, solved.ty, solved.tz
                    bone.rx, bone.ry, bone.rz = solved.rx, solved.ry, solved.rz
                    count += 1
        else:
            # A reference (non-animation) mesh: agree with the shared hand's finger
            # bind so studiomdl's single per-bone bind is consistent.  The weapon's
            # own hand mesh is replaced, so its finger verts never render here.
            for frame in smd.skeleton:
                for bone in frame.bones:
                    name = id_to_name.get(bone.bone_id)
                    if name not in shared:
                        continue
                    solved = _bt_from_mat4(bone.bone_id, shared[name])
                    bone.tx, bone.ty, bone.tz = solved.tx, solved.ty, solved.tz
                    bone.rx, bone.ry, bone.rz = solved.rx, solved.ry, solved.rz
                    count += 1
    return count


# Bounded wrist offset (TS.md Stage 4 / §8.3, "cheap variant"): the max distance
# the hand may be lifted out of the weapon, in units, and the line-search step.
_WRIST_OFFSET_MAX = 1.5
_WRIST_OFFSET_STEP = 0.1


def _signed_to_surface(points: np.ndarray, wP: np.ndarray, wN: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    For each of *points*, the distance to the nearest weapon vertex and the
    signed distance along that vertex's outward normal (negative = inside the
    weapon).  Batched to keep the pairwise distance matrix small.
    """
    idx = np.empty(len(points), dtype=int)
    dist = np.empty(len(points))
    for i in range(0, len(points), 256):
        batch = points[i:i + 256]
        d = np.linalg.norm(batch[:, None, :] - wP[None, :, :], axis=2)
        idx[i:i + 256] = d.argmin(1)
        dist[i:i + 256] = d.min(1)
    signed = np.einsum("ij,ij->i", points - wP[idx], wN[idx])
    return dist, signed, idx


def _apply_wrist_offsets(model: ModelInput, reference_hand: SMD,
                         reference_rigs: list, donor: SMD) -> int:
    """
    Lift the shared hand out of the weapon with a **bounded constant wrist offset**
    per hand (TS.md Stage 4, cheap variant §8.3).  Returns hands offset.

    Rotation retargeting (:func:`_retarget_fingers_to_reference`) fixes the finger
    *shape* but not placement: the donor hand can sit *inside* the gun — on the
    tightly-gripped pistols the knuckle row of v_usp is buried ~50 units deep, a
    pure placement error no finger-curl solve can reach.  So the whole hand below
    the palm is translated rigidly, just far enough to clear the surface.

    Only **rotations were touched before and only a palm translation here**, so
    every bone length stays byte-identical.  The offset is solved once on the idle
    frame — the direction is the mean outward weapon normal under the penetrating
    finger vertices, the distance a line search that stops as soon as penetration
    clears (capped at :data:`_WRIST_OFFSET_MAX`, so the grip cannot float away) —
    then held constant across all frames, expressed in the forearm frame so it
    tracks the arm.  It is applied to the **palm** bone: on these rigs the weapon
    hangs off the forearm as a sibling of the palm, so the gun does not move; where
    a weapon bone *does* descend from the palm (v_anaconda), its world transform is
    restored per frame so the gun stays pinned exactly.
    """
    slots = _hand_slots(model, _HANDS_GROUP_RE)
    hand_keys = {slot.key for slot in slots}
    weapon_meshes = [smd for key, smd in model.smds.items()
                     if not smd.is_animation and key not in hand_keys and smd.triangles]
    idle = next((smd for key, smd in model.smds.items()
                 if smd.is_animation and "idle" in key.lower() and smd.skeleton), None)
    if idle is None:
        idle = next((smd for smd in model.smds.values()
                     if smd.is_animation and smd.skeleton), None)
    if not weapon_meshes or idle is None or not reference_rigs:
        return 0

    ref_bind = world_transforms(reference_hand, 0)
    ref_id = {n.name: n.id for n in reference_hand.nodes}
    ref_id_to_name = {n.id: n.name for n in reference_hand.nodes}
    is_finger = lambda name: "finger" in name.lower()

    idle_pose = world_transforms(idle, 0)
    weapon_pts: list[np.ndarray] = []
    weapon_nrm: list[np.ndarray] = []
    for mesh in weapon_meshes:
        p, n = _posed_world_verts(mesh, world_transforms(mesh, 0), idle_pose)
        if len(p):
            weapon_pts.append(p)
            weapon_nrm.append(n)
    if not weapon_pts:
        return 0
    wP = np.vstack(weapon_pts)
    wN = np.vstack(weapon_nrm)
    wN = wN / (np.linalg.norm(wN, axis=1, keepdims=True) + 1e-9)

    applied = 0
    for rig in reference_rigs:
        palm, forearm = rig.hand, rig.forearm
        if palm not in idle_pose or (forearm and forearm not in idle_pose):
            continue
        chain_fingers = {b for chain in rig.chains for b in chain}
        # Idle-frame finger vertices of the donor hand, posed by the model's anim.
        xform = {name: idle_pose[name] @ np.linalg.inv(ref_bind[name])
                 for name in ref_bind if name in idle_pose}
        finger_pts: list[np.ndarray] = []
        for triangle in reference_hand.triangles:
            for v in triangle.vertices:
                name = ref_id_to_name.get(v.bone_id)
                if name in chain_fingers and name in xform:
                    finger_pts.append((xform[name] @ np.array([v.x, v.y, v.z, 1.0]))[:3])
        if not finger_pts:
            continue
        fp = np.array(finger_pts)

        dist, signed, idx = _signed_to_surface(fp, wP, wN)
        inside = (dist < 3.0) & (signed < -0.15)
        if not inside.any():
            continue                                    # already clear of the gun
        # Push direction: mean outward normal where the fingers are buried.
        direction = wN[idx[inside]].mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction = direction / norm

        # Line search: smallest lift that clears (or minimises) penetration.
        def penetration(shift: float) -> float:
            _, s, _ = _signed_to_surface(fp + shift * direction, wP, wN)
            near = s > -3.0
            return float((-s[near & (s < -0.15)]).sum())

        base_pen = penetration(0.0)
        best_t, best_pen = 0.0, base_pen
        t = _WRIST_OFFSET_STEP
        while t <= _WRIST_OFFSET_MAX + 1e-9:
            pen = penetration(t)
            if pen < best_pen - 1e-6:
                best_pen, best_t = pen, t
            if pen <= 0.01 * base_pen:                  # cleared — stop, stay close
                best_pen, best_t = pen, t
                break
            t += _WRIST_OFFSET_STEP
        if best_t <= 0.0:
            continue

        # World lift → constant offset in the forearm frame (tracks the arm).
        anchor = forearm if forearm and forearm in idle_pose else palm
        forearm_rot = idle_pose[anchor][:3, :3]
        offset = forearm_rot.T @ (best_t * direction)   # in palm's parent frame

        _shift_palm(model, palm, chain_fingers, offset)
        applied += 1
    return applied


def _shift_palm(model: ModelInput, palm: str, chain_fingers: set[str],
                offset: np.ndarray) -> None:
    """
    Add *offset* (in the palm's parent frame) to the palm bone's local translation
    in every animation frame, so the whole hand below it rides along.  Any weapon
    bone that descends from the palm is a direct non-finger child; its world
    transform is restored per frame so the gun stays pinned.
    """
    for smd in model.smds.values():
        if not smd.is_animation:
            continue
        name_to_id = {n.name: n.id for n in smd.nodes}
        id_to_name = {n.id: n.name for n in smd.nodes}
        palm_id = name_to_id.get(palm)
        if palm_id is None:
            continue
        child_map: dict[int, list[int]] = {}
        for n in smd.nodes:
            child_map.setdefault(n.parent_id, []).append(n.id)
        weapon_children = [c for c in child_map.get(palm_id, [])
                           if id_to_name.get(c) not in chain_fingers]
        for frame_index, frame in enumerate(smd.skeleton):
            bt = {b.bone_id: b for b in frame.bones}
            palm_bone = bt.get(palm_id)
            if palm_bone is None:
                continue
            # Record the gun-carrying children's world before the palm moves.
            saved = {}
            if weapon_children:
                world = world_transforms(smd, frame_index)
                for c in weapon_children:
                    nm = id_to_name.get(c)
                    if nm in world:
                        saved[c] = world[nm]
            palm_bone.tx += float(offset[0])
            palm_bone.ty += float(offset[1])
            palm_bone.tz += float(offset[2])
            if saved:
                new_palm_world = world_transforms(smd, frame_index)[palm]
                inv_palm = np.linalg.inv(new_palm_world)
                for c, world_c in saved.items():
                    child = bt.get(c)
                    if child is None:
                        continue
                    solved = _bt_from_mat4(c, inv_palm @ world_c)
                    child.tx, child.ty, child.tz = solved.tx, solved.ty, solved.tz
                    child.rx, child.ry, child.rz = solved.rx, solved.ry, solved.rz


# Per-finger curl relief (TS.md Stage 4 penetration term): the most a finger may
# be un-curled toward the open bind (fraction of the grip->open rotation), and the
# line-search step.  Bounded so the grip stays a grip.
_FINGER_RELAX = True
_RELAX_MAX = 0.5
_RELAX_STEP = 0.05
_RELAX_GATE = 0.2       # target max finger penetration depth (units)


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """3×3 rotation of *angle* radians about unit *axis* (Rodrigues)."""
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rotate_toward(current: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """
    Rotate *current* a fraction *alpha* of the way toward *target* (both 3×3, same
    frame).  ``alpha=0`` keeps the grip; ``alpha=1`` reaches the open bind.  Used
    to back a finger's flexion off just enough to lift it out of the weapon.
    """
    rel = target @ current.T
    cos = (np.trace(rel) - 1.0) / 2.0
    angle = float(np.arccos(max(-1.0, min(1.0, cos))))
    if angle < 1e-6:
        return current
    ax = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    n = float(np.linalg.norm(ax))
    if n < 1e-9:
        return current                              # ~180°: leave as-is (rare here)
    return _axis_angle_matrix(ax / n, angle * alpha) @ current


def _relax_finger_curl(model: ModelInput, reference_hand: SMD,
                       reference_rigs: list) -> int:
    """
    Un-curl each finger toward the open bind just enough that it stops clipping the
    weapon (TS.md Stage 4 penetration term, cheap constant-per-finger variant).
    Returns fingers relaxed.

    The donor fingers run 5-9% longer than the originals, so after rotation copy
    they wrap the grip a touch too deep and the pads poke through (~0.4-0.9 units on
    most pistols — a wrap-around penetration a rigid wrist offset cannot reach,
    since lifting one side of the finger buries the other).  Per finger, the smallest
    fraction of the grip->open rotation that clears the surface is solved on the idle
    frame and held constant across all frames, so the finger still animates and grips
    — just a hair looser.  Rotations only; bone lengths are never touched.
    """
    slots = _hand_slots(model, _HANDS_GROUP_RE)
    hand_keys = {slot.key for slot in slots}
    weapon_meshes = [smd for key, smd in model.smds.items()
                     if not smd.is_animation and key not in hand_keys and smd.triangles]
    idle = next((smd for key, smd in model.smds.items()
                 if smd.is_animation and "idle" in key.lower() and smd.skeleton), None)
    if idle is None:
        idle = next((smd for smd in model.smds.values()
                     if smd.is_animation and smd.skeleton), None)
    if not weapon_meshes or idle is None or not reference_rigs:
        return 0

    ref_bind = world_transforms(reference_hand, 0)
    ref_id_to_name = {n.id: n.name for n in reference_hand.nodes}
    open_rot = {}
    if reference_hand.skeleton:
        for bone in reference_hand.skeleton[0].bones:
            name = ref_id_to_name.get(bone.bone_id, "")
            if "finger" in name.lower():
                open_rot[name] = _mat4_from_bt(bone)[:3, :3]

    # Reference-hand vertices grouped by the finger bone they ride.
    verts_by_bone: dict[str, list[np.ndarray]] = {}
    for triangle in reference_hand.triangles:
        for v in triangle.vertices:
            name = ref_id_to_name.get(v.bone_id)
            if name in open_rot:
                verts_by_bone.setdefault(name, []).append(np.array([v.x, v.y, v.z, 1.0]))
    verts_by_bone = {k: np.array(v) for k, v in verts_by_bone.items()}

    id_to_name_idle = {n.id: n.name for n in idle.nodes}
    name_to_id_idle = {n.name: n.id for n in idle.nodes}

    # Solve over several idle frames, not one: an un-curl that clears frame 0 but
    # digs in as the grip shifts would otherwise be accepted and make things worse.
    count = len(idle.skeleton)
    # Sample the idle densely (~10 poses spread across it): a constant backoff
    # solved on a sparse set can dig in on the poses between, so validate on many.
    step = max(1, count // 10)
    frame_indices = sorted({min(f, count - 1) for f in range(0, count, step)})
    frames: list[tuple] = []
    for fi in frame_indices:
        pose = world_transforms(idle, fi)
        wp, wn = [], []
        for mesh in weapon_meshes:
            p, n = _posed_world_verts(mesh, world_transforms(mesh, 0), pose)
            if len(p):
                wp.append(p)
                wn.append(n)
        if not wp:
            continue
        wP = np.vstack(wp)
        wN = np.vstack(wn)
        wN = wN / (np.linalg.norm(wN, axis=1, keepdims=True) + 1e-9)
        local = {id_to_name_idle[b.bone_id]: _mat4_from_bt(b)
                 for b in idle.skeleton[fi].bones if b.bone_id in id_to_name_idle}
        frames.append((pose, wP, wN, local))
    if not frames:
        return 0

    def finger_penetration(chain: list[str], alpha: float) -> float:
        """
        **Deepest** penetration of *chain*'s verts, un-curled by *alpha*, over the
        sampled frames.  Max, not sum: the visible defect is the single deepest
        poke, and minimising the sum can trade many shallow pokes for a few deeper
        ones — lowering the total while making the clipping look *worse*.
        """
        parent_name = _parent_of(idle, chain[0], name_to_id_idle, id_to_name_idle)
        worst = 0.0
        for pose, wP, wN, local in frames:
            parent_world = pose.get(parent_name)
            if parent_world is None:
                continue
            pts = []
            world_c = parent_world
            for name in chain:
                lm = local.get(name)
                if lm is None:
                    break
                rot = _rotate_toward(lm[:3, :3], open_rot[name], alpha) if alpha > 0 else lm[:3, :3]
                m = np.eye(4)
                m[:3, :3] = rot
                m[:3, 3] = lm[:3, 3]
                world_c = world_c @ m
                if name in verts_by_bone and name in ref_bind:
                    xf = world_c @ np.linalg.inv(ref_bind[name])
                    pts.append((verts_by_bone[name] @ xf.T)[:, :3])
            if not pts:
                continue
            hp = np.vstack(pts)
            _, signed, _ = _signed_to_surface(hp, wP, wN)
            deep = -signed[(signed < -0.15) & (signed > -3.0)]
            if len(deep):
                worst = max(worst, float(deep.max()))
        return worst

    idle_local = frames[0][3]
    alphas: dict[str, float] = {}
    for rig in reference_rigs:
        for chain in rig.chains:
            if not all(b in open_rot and b in idle_local for b in chain):
                continue
            base = finger_penetration(chain, 0.0)
            if base <= _RELAX_GATE:
                continue                            # already within the depth gate
            best_a, best_pen = 0.0, base
            a = _RELAX_STEP
            while a <= _RELAX_MAX + 1e-9:
                pen = finger_penetration(chain, a)
                if pen < best_pen - 1e-6:            # only accept a real improvement
                    best_pen, best_a = pen, a
                if pen <= _RELAX_GATE:               # cleared the gate — stop, stay close
                    break
                a += _RELAX_STEP
            if best_a > 0.0:
                for name in chain:
                    alphas[name] = best_a

    if not alphas:
        return 0
    _apply_finger_relax(model, alphas, open_rot)
    return len(alphas)


def _parent_of(smd: SMD, name: str, name_to_id: dict, id_to_name: dict) -> str | None:
    """Name of *name*'s parent bone in *smd* (the palm, for a finger root)."""
    nid = name_to_id.get(name)
    by_id = {n.id: n for n in smd.nodes}
    node = by_id.get(nid)
    if node is None:
        return None
    return id_to_name.get(node.parent_id)


def _apply_finger_relax(model: ModelInput, alphas: dict[str, float],
                        open_rot: dict[str, np.ndarray]) -> None:
    """
    Rotate each finger bone in *alphas* a constant fraction toward its open bind, in
    every animation frame, preserving the per-frame grip motion and the donor length.
    """
    for smd in model.smds.values():
        if not smd.is_animation:
            continue
        id_to_name = {n.id: n.name for n in smd.nodes}
        for frame in smd.skeleton:
            for bone in frame.bones:
                name = id_to_name.get(bone.bone_id)
                alpha = alphas.get(name)
                if alpha is None:
                    continue
                local = _mat4_from_bt(bone)
                local[:3, :3] = _rotate_toward(local[:3, :3], open_rot[name], alpha)
                solved = _bt_from_mat4(bone.bone_id, local)
                bone.tx, bone.ty, bone.tz = solved.tx, solved.ty, solved.tz
                bone.rx, bone.ry, bone.rz = solved.rx, solved.ry, solved.rz


# Full per-frame penetration solve (TS.md Stage 4 §8.2).  OFF: measured to be both
# impractical (~4 min/model — a per-frame weapon re-pose) and unreliable on exactly
# the grips it was meant for.  It helps mid cases (v_bglock18 0.80->0.53u) but makes
# the tightest wrap-around WORSE (v_usp 0.71->0.83u): a finger hooked >90° round a
# cylindrical grip cannot be rotated out of one wall without poking the other, so
# rotation-only optimisation just trades penetration between sides.  For those grips
# the donor hand geometrically does not fit — keep-own is the honest answer.  Left in
# place as documented groundwork.
_CONTACT_FIT = False
_GN_ITERS = 6
_GN_LAMBDA = 0.4          # Levenberg damping: stability + keeps the pose near the grip
_GN_STEP_MAX = 0.20       # max joint rotation per iteration (rad)
_GN_TOTAL_MAX = 0.55      # max cumulative joint rotation from the grip pose (rad)
_GN_DEPTH = 0.15          # a vertex counts as penetrating past this depth (units)


def _finger_fk(chain: list[str], parent_world: np.ndarray,
               local_rot: list[np.ndarray], donor_trans: list[np.ndarray]
               ) -> list[np.ndarray]:
    """World 4×4 per joint of *chain* from *parent_world* and per-joint local
    rotation + (locked) donor translation."""
    worlds = []
    w = parent_world
    for i in range(len(chain)):
        m = np.eye(4)
        m[:3, :3] = local_rot[i]
        m[:3, 3] = donor_trans[i]
        w = w @ m
        worlds.append(w)
    return worlds


def _gn_finger_frame(chain: list[str], parent_world: np.ndarray,
                     local_rot: list[np.ndarray], donor_trans: list[np.ndarray],
                     verts: list[np.ndarray], bind_inv: list[np.ndarray],
                     wP: np.ndarray, wN: np.ndarray) -> list[np.ndarray] | None:
    """
    Gauss-Newton: rotate the joints of one finger, one frame, so its penetrating
    vertices ride out to the weapon surface.  Returns new per-joint local rotations,
    or ``None`` when the finger does not clip (no change).

    The Jacobian of a vertex's penetration depth w.r.t. a small rotation about a
    world axis *e* at joint *j* is ``(v − p_j) × n`` (n = weapon surface normal at
    the vertex), a plain cross product — so one linear solve per iteration moves
    every joint at once.  Damped and bounded: the pose stays a grip, and because
    only rotations change, every bone length is preserved exactly.
    """
    m = len(chain)
    rot = [r.copy() for r in local_rot]
    total = np.zeros((m, 3))                    # cumulative rotation vector per joint
    changed = False
    for _ in range(_GN_ITERS):
        worlds = _finger_fk(chain, parent_world, rot, donor_trans)
        p = [w[:3, 3] for w in worlds]
        pts, jidx = [], []
        for i in range(m):
            if len(verts[i]) == 0:
                continue
            xf = worlds[i] @ bind_inv[i]
            pts.append((verts[i] @ xf.T)[:, :3])
            jidx.append(np.full(len(verts[i]), i))
        if not pts:
            return rot if changed else None
        V = np.vstack(pts)
        B = np.concatenate(jidx)
        _, signed, idx = _signed_to_surface(V, wP, wN)
        depth = -signed
        inside = (depth > _GN_DEPTH) & (signed > -3.0)
        if not inside.any():
            break
        Vi, Bi, ni, ri = V[inside], B[inside], wN[idx[inside]], depth[inside]
        J = np.zeros((len(ri), 3 * m))
        for r_i in range(len(ri)):
            for j in range(Bi[r_i] + 1):        # joints 0..b move this vertex
                J[r_i, 3 * j:3 * j + 3] = np.cross(Vi[r_i] - p[j], ni[r_i])
        JtJ = J.T @ J + _GN_LAMBDA * np.eye(3 * m)
        delta = np.linalg.solve(JtJ, J.T @ ri).reshape(m, 3)
        moved = False
        for j in range(m):
            wvec = delta[j]
            mag = float(np.linalg.norm(wvec))
            if mag > _GN_STEP_MAX:
                wvec = wvec * (_GN_STEP_MAX / mag)
            cand = total[j] + wvec
            cmag = float(np.linalg.norm(cand))
            if cmag > _GN_TOTAL_MAX:
                cand = cand * (_GN_TOTAL_MAX / cmag)
                wvec = cand - total[j]
            if float(np.linalg.norm(wvec)) < 1e-6:
                continue
            total[j] = cand
            angle = float(np.linalg.norm(wvec))
            dR = _axis_angle_matrix(wvec / (angle + 1e-12), angle)
            parent_rot = (parent_world[:3, :3] if j == 0 else worlds[j - 1][:3, :3])
            rot[j] = parent_rot.T @ dR @ parent_rot @ rot[j]
            moved = True
        if moved:
            changed = True
        else:
            break
    return rot if changed else None


def _contact_fit(model: ModelInput, reference_hand: SMD,
                 reference_rigs: list, donor: SMD) -> int:
    """
    Per-frame penetration solve (TS.md Stage 4 §8.2) for grips the cheap passes miss.
    Returns finger-frames adjusted.  Self-gating: per frame it only touches fingers
    that still clip, and a finger already clear costs one nearest-surface query.
    """
    slots = _hand_slots(model, _HANDS_GROUP_RE)
    hand_keys = {slot.key for slot in slots}
    weapon_meshes = [smd for key, smd in model.smds.items()
                     if not smd.is_animation and key not in hand_keys and smd.triangles]
    if not weapon_meshes or not reference_rigs:
        return 0
    weapon_binds = [(mesh, world_transforms(mesh, 0)) for mesh in weapon_meshes]

    ref_bind = world_transforms(reference_hand, 0)
    ref_id_to_name = {n.id: n.name for n in reference_hand.nodes}
    finger_names = {n for n in ref_bind if "finger" in n.lower()}
    verts_by_bone: dict[str, np.ndarray] = {}
    tmp: dict[str, list] = {}
    for triangle in reference_hand.triangles:
        for v in triangle.vertices:
            name = ref_id_to_name.get(v.bone_id)
            if name in finger_names:
                tmp.setdefault(name, []).append([v.x, v.y, v.z, 1.0])
    for k, v in tmp.items():
        verts_by_bone[k] = np.array(v)
    bind_inv = {n: np.linalg.inv(ref_bind[n]) for n in finger_names if n in ref_bind}

    chains = []
    for rig in reference_rigs:
        for chain in rig.chains:
            if all(b in bind_inv for b in chain):
                chains.append((rig.hand, chain))
    if not chains:
        return 0

    def weapon_cloud(smd, frame_index):
        pose = world_transforms(smd, frame_index)
        ps, ns = [], []
        for mesh, bind in weapon_binds:
            p, n = _posed_world_verts(mesh, bind, pose)
            if len(p):
                ps.append(p)
                ns.append(n)
        if not ps:
            return pose, None, None
        P = np.vstack(ps)
        N = np.vstack(ns)
        N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
        return pose, P, N

    adjusted = 0
    for smd in model.smds.values():
        if not smd.is_animation or not smd.skeleton:
            continue
        name_to_id = {n.name: n.id for n in smd.nodes}
        for frame_index, frame in enumerate(smd.skeleton):
            pose, wP, wN = weapon_cloud(smd, frame_index)
            if wP is None:
                continue
            bt = {b.bone_id: b for b in frame.bones}
            for palm, chain in chains:
                if palm not in pose:
                    continue
                ids = [name_to_id.get(b) for b in chain]
                if any(i is None or i not in bt for i in ids):
                    continue
                local_rot = [_mat4_from_bt(bt[i])[:3, :3] for i in ids]
                donor_trans = [np.array([bt[i].tx, bt[i].ty, bt[i].tz]) for i in ids]
                verts = [verts_by_bone.get(b, np.empty((0, 4))) for b in chain]
                binv = [bind_inv[b] for b in chain]
                new_rot = _gn_finger_frame(chain, pose[palm], local_rot, donor_trans,
                                           verts, binv, wP, wN)
                if new_rot is None:
                    continue
                for i in range(len(chain)):
                    bone = bt[ids[i]]
                    mat = np.eye(4)
                    mat[:3, :3] = new_rot[i]
                    mat[:3, 3] = donor_trans[i]
                    solved = _bt_from_mat4(bone.bone_id, mat)
                    bone.rx, bone.ry, bone.rz = solved.rx, solved.ry, solved.rz
                    adjusted += 1
    return adjusted


# Grip carving: per-vertex weapon-side recess where the donor fingers penetrate, so
# the shared hand can stay on every weapon.  OFF: it needs a reliable "into the gun"
# direction, and these decompiled meshes provide none — vertex normals are
# inconsistent (partly inward) and the mesh Laplacian isn't interior on thin,
# non-convex grips, so every direction heuristic carves the wrong way somewhere
# (v_usp 0.71→1.2/2.2u).  A correct carve needs a true solid inside/outside
# (generalised winding number); these non-watertight meshes don't support one
# cheaply.  Left as documented groundwork.
_CARVE_GRIPS = False
_CARVE_RADIUS = 1.6       # groove half-width (units)
_CARVE_MARGIN = 0.12      # extra clearance behind the finger
_CARVE_MAX = 1.6          # cap on how deep a vertex may be carved


def _carve_grips(model: ModelInput, reference_rigs: list) -> int:
    """
    Recess the weapon grip out of the donor fingers.  Returns vertices moved.

    Decompiled weapon meshes carry inconsistent, often inward-facing vertex normals,
    so "which way is into the gun" cannot be read from them.  Instead the interior
    direction is the mesh **Laplacian** — from a vertex toward the mean of its
    triangle neighbours, which points into the solid for a convex grip and needs no
    normal.  A finger vertex on the interior side of a weapon vertex (positive
    Laplacian projection) is penetrating by that projection; the weapon vertex is
    then pushed inward by the deepest such projection across sampled frames (smooth
    radial falloff, capped).  Only the weapon mesh moves — the shared hand and every
    bone are untouched, so the hand still reads identically on all 56 weapons.
    """
    slots = _hand_slots(model, _HANDS_GROUP_RE)
    if not slots:
        return 0
    hand_keys = {s.key for s in slots}
    hand_mesh = model.smds[slots[0].key]
    weapon_meshes = [smd for key, smd in model.smds.items()
                     if not smd.is_animation and key not in hand_keys and smd.triangles]
    if not weapon_meshes:
        return 0

    hid = {n.id: n.name for n in hand_mesh.nodes}
    is_finger = lambda b: "finger" in hid.get(b, "").lower()
    hbind = world_transforms(hand_mesh, 0)
    hbind_inv = {n: np.linalg.inv(hbind[n]) for n in hbind}
    finger_by_bone: dict[str, list] = {}
    for tri in hand_mesh.triangles:
        for v in tri.vertices:
            name = hid.get(v.bone_id)
            if name and is_finger(v.bone_id) and name in hbind_inv:
                finger_by_bone.setdefault(name, []).append([v.x, v.y, v.z, 1.0])
    finger_by_bone = {k: np.array(v) for k, v in finger_by_bone.items()}
    if not finger_by_bone:
        return 0

    frames = []
    for smd in model.smds.values():
        if smd.is_animation and smd.skeleton:
            n = len(smd.skeleton)
            for fi in sorted({0, n // 2, n - 1}):
                frames.append((smd, fi))
    if not frames:
        return 0

    moved = 0
    for mesh in weapon_meshes:
        mid = {n.id: n.name for n in mesh.nodes}
        mbind = world_transforms(mesh, 0)
        mbind_inv = {n: np.linalg.inv(mbind[n]) for n in mbind}
        # Unique weapon vertices + triangle adjacency for the Laplacian.
        index: dict[tuple, int] = {}
        order: list[tuple] = []
        pos_list, bone_list, corners = [], [], []
        neigh: list[set] = []
        for tri in mesh.triangles:
            uid = []
            for v in tri.vertices:
                key = (round(v.x, 3), round(v.y, 3), round(v.z, 3))
                i = index.get(key)
                if i is None:
                    i = len(order)
                    index[key] = i
                    order.append(key)
                    pos_list.append([v.x, v.y, v.z])
                    bone_list.append(mid.get(v.bone_id))
                    corners.append([])
                    neigh.append(set())
                corners[i].append(v)
                uid.append(i)
            for a in range(3):                       # mutual neighbours
                neigh[uid[a]].update(uid[b] for b in range(3) if b != a)
        ref_pos = np.array(pos_list)
        U = len(order)
        # Reference-pose interior direction (Laplacian), normalised.
        lap = np.zeros((U, 3))
        for i in range(U):
            if neigh[i]:
                lap[i] = ref_pos[list(neigh[i])].mean(0) - ref_pos[i]
        lap_n = np.linalg.norm(lap, axis=1, keepdims=True)
        lap_dir = lap / (lap_n + 1e-9)
        by_bone: dict[str, np.ndarray] = {}
        for i, b in enumerate(bone_list):
            by_bone.setdefault(b, []).append(i)
        by_bone = {b: np.array(idx) for b, idx in by_bone.items() if b in mbind_inv}
        carve = np.zeros(U)

        for smd, fi in frames:
            pose = world_transforms(smd, fi)
            Wp = np.full((U, 3), np.nan)
            Dw = np.zeros((U, 3))                     # world interior direction
            for b, idx in by_bone.items():
                if b not in pose:
                    continue
                x = pose[b] @ mbind_inv[b]
                Wp[idx] = (ref_pos[idx] @ x[:3, :3].T) + x[:3, 3]
                Dw[idx] = lap_dir[idx] @ x[:3, :3].T
            valid = ~np.isnan(Wp[:, 0])
            if not valid.any():
                continue
            Fp = []
            for b, arr in finger_by_bone.items():
                if b in pose:
                    Fp.append((arr @ (pose[b] @ hbind_inv[b]).T)[:, :3])
            if not Fp:
                continue
            F = np.vstack(Fp)
            widx = np.nonzero(valid)[0]
            Wv, Dv = Wp[widx], Dw[widx]
            wdot = np.einsum("ij,ij->i", Wv, Dv)     # W·d per weapon vertex
            # For each weapon vertex, deepest finger on its interior side within radius.
            for s in range(0, len(widx), 256):
                sl = slice(s, s + 256)
                wv, dv, wd = Wv[sl], Dv[sl], wdot[sl]
                dist = np.linalg.norm(wv[:, None, :] - F[None, :, :], axis=2)  # (chunk, F)
                t = (F @ dv.T).T - wd[:, None]        # (chunk, F): (f - w)·d
                fall = np.clip(1.0 - dist / _CARVE_RADIUS, 0.0, 1.0)
                cand = np.where((t > _CARVE_MARGIN) & (dist < _CARVE_RADIUS),
                                (t + _CARVE_MARGIN) * fall, 0.0).max(1)
                gi = widx[sl]
                carve[gi] = np.maximum(carve[gi], cand)

        carve = np.minimum(carve, _CARVE_MAX)
        hit = carve > 1e-3
        if not hit.any():
            continue
        new_pos = ref_pos + carve[:, None] * lap_dir  # push toward interior
        for i in np.nonzero(hit)[0]:
            p = new_pos[i]
            for v in corners[i]:
                v.x, v.y, v.z = float(p[0]), float(p[1]), float(p[2])
            moved += 1
    return moved


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
        result.retargeted = _retarget_fingers_to_reference(
            model, reference_hand, reference_rigs, donor)
        if _WRIST_OFFSET:
            _apply_wrist_offsets(model, reference_hand, reference_rigs, donor)
        if _FINGER_RELAX:
            _relax_finger_curl(model, reference_hand, reference_rigs)

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
    if _CARVE_GRIPS and replace_mesh and retarget_fingers:
        result.carved = _carve_grips(model, reference_rigs)

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
        if bodygroup.name.startswith(f"{HAND_SMD_KEY}_original"):
            # A hand deliberately parked out of the shared group (a rig that could
            # not be matched keeps its own hand here).  It DOES hold a hand mesh, so
            # the detection below would pull it back into "hands" and the shared-hand
            # replacement would overwrite it — leave it exactly where it is.
            continue
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
    trim_forearm: bool = False,
    keep_own: set[str] | None = None,
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
        if trim_forearm:
            strip_forearm(mesh)
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
    # Models whose rig could not be matched keep their own hand in a separate
    # bodygroup; give them a trailing blank here so they show NO shared hand (else
    # both the shared and their own hand draw at once).
    keep_own = keep_own or set()
    blank_index = len(entries)
    if keep_own:
        entries.append(BodyGroupEntry(smd=""))
    for group in groups:
        group.entries = list(entries)

    # Shared-hand models default to variant 0; kept-own models pick the blank.
    override = {hands_group_name: {
        name: (blank_index if name in keep_own else 0) for name in merged.model_names
    }}
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


def _uniquify_weapon_bones(model: ModelInput) -> dict[str, str]:
    """
    Rename a P-model's weapon bones model-unique, keeping the shared ``Bip01``
    player skeleton.  Returns the rename map.

    Every ``p_``/``w_`` model rides the same biped (``Bip01*``), which must merge
    into one shared skeleton, but each carries its own weapon bone(s) off the
    hand.  Some rigs give those generic names (``p``, ``Knife_Wow``) that would
    collide across weapons; the merger shares a same-named, same-parent bone, so
    two weapons that hold differently (``p`` on ``p_ironfan`` vs
    ``p_tomahawk_xmas``) would then be posed by one shared bone and one would be
    wrong.  Prefixing the non-``Bip01`` bones with the model name keeps every
    weapon on its own bone so the single player pose places each correctly.
    """
    names: set[str] = set()
    for smd in model.smds.values():
        names |= {node.name for node in smd.nodes}
    renames = {
        name: f"{model.name}__{name}"
        for name in names
        if not name.startswith("Bip01")
    }
    if renames:
        for smd in model.smds.values():
            rename_bones(smd, renames)
        _rename_qc_bones(model.qc, renames)
    return renames


def _collapse_to_player_pose(
    merged: MergeResult, sequence_name: str = "player", anim_key: str = "a/player"
) -> int:
    """
    Replace a P (third-person) model's per-weapon sequences with ONE animation
    that holds the biped in its carry pose and every weapon bone at its own resting
    place.  Returns the bone count of the pose (0 if nothing to do).

    A ``p_``/``w_`` model rides the player skeleton and the engine plays a *single*
    sequence on it no matter which weapon ``pev_body`` selects, so every weapon
    bone has to be posed by that one clip (this is how the stock CS ``weapons.mdl``
    works — one ``player`` sequence over a shared skeleton, weapons chosen by
    bodygroup).  After :func:`_unify_skeleton` every reference mesh carries the
    full skeleton with each weapon bone at its own bind (its held pose), so this
    lifts that frame-0 pose into a two-frame animation and drops the per-model
    sequences.  Requires pooling OFF so every weapon keeps its own bone.
    """
    references = [smd for smd in merged.smds.values() if not smd.is_animation and smd.skeleton]
    if not references:
        return 0
    reference = max(references, key=lambda smd: len(smd.nodes))

    pose = SMD()
    pose.nodes = deepcopy(reference.nodes)
    frame = reference.skeleton[0]
    pose.skeleton = [
        SkeletonFrame(time=0, bones=deepcopy(frame.bones)),
        SkeletonFrame(time=1, bones=deepcopy(frame.bones)),
    ]

    for key in [k for k, smd in merged.smds.items() if smd.is_animation]:
        del merged.smds[key]
    merged.smds[anim_key] = pose
    merged.qc.sequences = [Sequence(name=sequence_name, smd_paths=[anim_key], fps=30)]
    return len(pose.nodes)


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
    trim_forearm: bool = False,
    contact_keep_own: float | None = None,
    player_model: bool = False,
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

    if player_model:
        # A P/W model is posed by one shared sequence, so every weapon needs its
        # own bone (no pooling), and the whole skeleton must be in every SMD.
        pool_bones_pass = False
        unify_skeleton = True

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
    kept_own_hand_models: set[str] = set()
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

        if player_model:
            _uniquify_weapon_bones(model)
            # A p_/w_ model is one weapon = one bodygroup.  Some decompiles carry a
            # spurious extra group (p_luger keeps its skin "upgrade" meshes in a
            # second, mis-named "hands" group); the merger would reference those
            # meshes without writing them and the compile fails on the missing SMD.
            if len(model.qc.bodygroups) > 1:
                log(f"    dropped {len(model.qc.bodygroups) - 1} extra bodygroup(s) "
                    f"(p/w model keeps only its weapon)")
                model.qc.bodygroups = model.qc.bodygroups[:1]
            # Hitboxes are for the player's own model, not the held/dropped weapon;
            # keeping them just pins (or dangles on) bones and breaks the compile.
            model.qc.hboxes = []

        prep.renamed_bodygroups = dedupe_bodygroup_names(model.qc)
        if prep.renamed_bodygroups:
            total = sum(count - 1 for count in prep.renamed_bodygroups.values())
            log(f"    renamed {total} duplicate bodygroup name(s): "
                f"{', '.join(sorted(prep.renamed_bodygroups))}")

        if normalise and reference_hand is not None:
            effective_max_cost = (
                hand_match_max_cost
                if not keep_hand_mesh and (hand_match_max_cost or 0) > 0
                else None
            )
            # Dynamic keep-own trigger: even a rig that fits at rest may have the
            # shared hand clip through the weapon once the fingers curl.  Measure
            # that penetration and, if the shared hand grips much worse than the
            # model's own, force keeping the own hand (a cost of -1 does that).
            if contact_keep_own is not None and not keep_hand_mesh:
                excess = _shared_hand_excess_penetration(model, reference_hand, reference_rigs)
                if excess > contact_keep_own:
                    log(f"    shared hand clips the weapon (excess penetration "
                        f"{excess:.0f} > {contact_keep_own:.0f}); keeping own hand")
                    effective_max_cost = -1.0
            normalisation = normalise_hands(
                model, reference_hand, reference_rigs,
                texture=None if keep_hand_mesh else hand_texture_name,
                repose=repose_hands, replace_mesh=not keep_hand_mesh,
                vertex_budget=vertex_budget,
                max_match_cost=effective_max_cost,
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
                        f"{hand_match_max_cost}; kept own hand (rig cannot fit the shared hand)"
                    )
                    # Keep this own hand OUT of the shared "hands" group so the
                    # shared-hand collapse and --default-hands variant replacement
                    # never overwrite it (its rig — e.g. mirrored handedness — cannot
                    # be matched to the shared hand, so the shared one would grip wrong).
                    kept_own_hand_models.add(model.name)
                    for bodygroup in model.qc.bodygroups:
                        if bodygroup.name == HAND_SMD_KEY:
                            bodygroup.name = f"{HAND_SMD_KEY}_original"
                    dedupe_bodygroup_names(model.qc)
                else:
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
        count, stride = _apply_hand_variants(merged, hand_variants, trim_forearm=trim_forearm,
                                             keep_own=kept_own_hand_models)
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

    if player_model:
        bones = _collapse_to_player_pose(merged)
        if bones:
            log(f"    player pose: one 'player' sequence holds all weapons "
                f"({bones}-bone skeleton); weapon chosen by pev_body")

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
