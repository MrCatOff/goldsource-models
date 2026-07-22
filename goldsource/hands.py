"""
Hand normalisation.

Every decompiled weapon model ships its own hand mesh, rigged to its own bone
names.  To merge models into one skeleton we replace all of them with a single
optimised hand mesh (``storage/hands/default_hand.smd``), re-bound to each
model's existing hand bones so the model's original animations keep driving it.

Why the bone names are matched *geometrically*
----------------------------------------------
The CS/CSO rigs use opaque names (``Bone05``, ``Bone09``, …) whose numbering
does not correspond to finger order — and the ordering differs between the left
and right hand of the very same rig::

    Bip01_L_Finger0(thumb) -> Bone05   Bip01_R_Finger0(thumb) -> Bone27
    Bip01_L_Finger1(index) -> Bone21   Bip01_R_Finger1(index) -> Bone31
    Bip01_L_Finger2        -> Bone17   Bip01_R_Finger2        -> Bone35
    Bip01_L_Finger3        -> Bone13   Bip01_R_Finger3        -> Bone39
    Bip01_L_Finger4(pinky) -> Bone09   Bip01_R_Finger4(pinky) -> Bone43

So instead of trusting names, each finger chain is matched by its bind-pose
position expressed in hand-local space.  This is name-agnostic and works for
any rig whose hand has the usual five three-segment finger chains.

Why swapping the mesh does not break the animations
---------------------------------------------------
studiomdl converts each reference mesh's vertices to bone-local space using
*that mesh's own* skeleton block.  A replacement hand therefore only has to
name the same bones — its bind pose may differ freely from the weapon mesh's.
At runtime the animations drive the bones and the new hand follows.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from goldsource.smd import SMD
from goldsource.skeleton import child_ids, world_transforms, rename_bones, remove_bones, renumber


# A hand is a bone with at least this many finger-like child chains.
MIN_FINGER_CHAINS = 4
# A finger chain must have at least this many bones (proximal/middle/distal).
MIN_CHAIN_LENGTH = 3
# How far above the palm bone to look for the forearm.
MAX_FOREARM_DEPTH = 4


# ---------------------------------------------------------------------------
# Rig detection
# ---------------------------------------------------------------------------

@dataclass
class HandRig:
    """A detected hand: the palm bone, its ancestors and its finger chains."""
    hand: str
    forearm: str | None
    chains: list[list[str]] = field(default_factory=list)
    # Ancestors of the palm bone, nearest first.  The forearm is *chosen* from
    # this chain by arm length rather than assumed to be the direct parent —
    # some rigs insert a near-coincident wrist bone between the two.
    ancestors: list[str] = field(default_factory=list)

    @property
    def bones(self) -> set[str]:
        names = {self.hand}
        if self.forearm:
            names.add(self.forearm)
        for chain in self.chains:
            names.update(chain)
        return names


def _chain_from(smd: SMD, kids: dict[int, list[int]], start_id: int) -> list[str]:
    """Follow a single-child chain from *start_id* and return the bone names."""
    by_id = {n.id: n for n in smd.nodes}
    chain = [by_id[start_id].name]
    current = start_id
    while True:
        children = kids.get(current, [])
        if len(children) != 1:
            break
        current = children[0]
        chain.append(by_id[current].name)
    return chain


def detect_rigs(smd: SMD) -> list[HandRig]:
    """
    Find every hand-like bone in *smd*: a bone with at least
    :data:`MIN_FINGER_CHAINS` children that each start a chain of at least
    :data:`MIN_CHAIN_LENGTH` bones.
    """
    kids = child_ids(smd)
    by_id = {n.id: n for n in smd.nodes}
    rigs: list[HandRig] = []

    for node in smd.nodes:
        chains: list[list[str]] = []
        for child_id in kids.get(node.id, []):
            chain = _chain_from(smd, kids, child_id)
            if len(chain) >= MIN_CHAIN_LENGTH:
                chains.append(chain)
        if len(chains) >= MIN_FINGER_CHAINS:
            ancestors: list[str] = []
            current = by_id.get(node.parent_id)
            while current is not None and len(ancestors) < MAX_FOREARM_DEPTH:
                ancestors.append(current.name)
                current = by_id.get(current.parent_id)
            rigs.append(HandRig(
                hand=node.name,
                forearm=ancestors[0] if ancestors else None,
                chains=chains,
                ancestors=ancestors,
            ))
    return rigs


# ---------------------------------------------------------------------------
# Geometric matching
# ---------------------------------------------------------------------------

def _local_positions(smd: SMD, rig: HandRig) -> dict[str, np.ndarray]:
    """Bind-pose position of each finger-chain root, in hand-local space."""
    world = world_transforms(smd)
    hand = world.get(rig.hand)
    if hand is None:
        return {}
    inv = np.linalg.inv(hand)
    result: dict[str, np.ndarray] = {}
    for chain in rig.chains:
        mat = world.get(chain[0])
        if mat is not None:
            result[chain[0]] = (inv @ mat)[:3, 3]
    return result


def _match_chains(
    src_smd: SMD, src_rig: HandRig,
    dst_smd: SMD, dst_rig: HandRig,
) -> tuple[dict[str, str], float]:
    """
    Pair each of *src_rig*'s finger chains with the nearest unused chain of
    *dst_rig*, comparing hand-local bind positions.

    Returns ``({src_chain_root: dst_chain_root}, total_distance)``.  The total
    distance doubles as a confidence score used to decide which detected hand
    is the left one and which is the right.
    """
    src_pos = _local_positions(src_smd, src_rig)
    dst_pos = _local_positions(dst_smd, dst_rig)

    candidates = sorted(
        (float(np.linalg.norm(sp - dp)), s, d)
        for s, sp in src_pos.items()
        for d, dp in dst_pos.items()
    )

    pairing: dict[str, str] = {}
    used_src: set[str] = set()
    used_dst: set[str] = set()
    total = 0.0
    for distance, src_name, dst_name in candidates:
        if src_name in used_src or dst_name in used_dst:
            continue
        used_src.add(src_name)
        used_dst.add(dst_name)
        pairing[src_name] = dst_name
        total += distance

    # Unmatched chains (rigs with differing finger counts) count against the score.
    total += 10.0 * (len(src_pos) - len(pairing))
    return pairing, total


def _bone_distance(world: dict[str, np.ndarray], a: str, b: str) -> float | None:
    """Bind-pose distance between two bones, or None if either is missing."""
    if a not in world or b not in world:
        return None
    return float(np.linalg.norm(world[a][:3, 3] - world[b][:3, 3]))


def _pick_forearm(
    ref_world: dict[str, np.ndarray], ref_rig: HandRig,
    model_world: dict[str, np.ndarray], model_rig: HandRig,
) -> tuple[str | None, float]:
    """
    Choose which ancestor of the model's palm bone plays the forearm, by
    matching the reference rig's forearm-to-hand length.

    The direct parent is not reliable: some rigs put a near-coincident wrist
    bone directly above the palm, with the real forearm one level higher.
    Binding the reference forearm mesh to that wrist bone would slide the whole
    forearm down the arm by its own length.  Returns ``(bone, penalty)``.
    """
    if not model_rig.ancestors:
        return None, 0.0

    if ref_rig.forearm is None:
        return model_rig.ancestors[0], 0.0

    target = _bone_distance(ref_world, ref_rig.hand, ref_rig.forearm)
    if target is None:
        return model_rig.ancestors[0], 0.0

    best_name, best_error = None, float("inf")
    for candidate in model_rig.ancestors:
        distance = _bone_distance(model_world, model_rig.hand, candidate)
        if distance is None:
            continue
        error = abs(distance - target)
        if error < best_error:
            best_name, best_error = candidate, error

    if best_name is None:
        return model_rig.ancestors[0], 0.0
    return best_name, best_error


@dataclass
class HandMatch:
    """Result of matching the reference hand rig onto one model."""
    mapping: dict[str, str] = field(default_factory=dict)   # reference bone -> model bone
    score: float = 0.0                                       # lower is better
    pairs: list[tuple[str, str]] = field(default_factory=list)  # (ref hand, model hand)
    unmapped: list[str] = field(default_factory=list)


def match_hands(
    ref_smd: SMD, ref_rigs: list[HandRig],
    model_smd: SMD, model_rigs: list[HandRig],
) -> HandMatch:
    """
    Build a ``reference bone -> model bone`` rename map.

    When both skeletons expose exactly two hands, both left/right assignments
    are scored and the cheaper one wins — so the sides are resolved from the
    geometry rather than from bone names, which are unreliable.
    """
    if not ref_rigs or not model_rigs:
        return HandMatch(unmapped=[r.hand for r in ref_rigs])

    ref_world = world_transforms(ref_smd)
    model_world = world_transforms(model_smd)

    def build(assignment: list[tuple[HandRig, HandRig]]) -> HandMatch:
        mapping: dict[str, str] = {}
        score = 0.0
        pairs: list[tuple[str, str]] = []
        for ref_rig, model_rig in assignment:
            mapping[ref_rig.hand] = model_rig.hand
            forearm, penalty = _pick_forearm(ref_world, ref_rig, model_world, model_rig)
            if ref_rig.forearm and forearm:
                mapping[ref_rig.forearm] = forearm
                score += penalty
            chain_pairing, cost = _match_chains(ref_smd, ref_rig, model_smd, model_rig)
            score += cost
            pairs.append((ref_rig.hand, model_rig.hand))

            ref_by_root = {c[0]: c for c in ref_rig.chains}
            model_by_root = {c[0]: c for c in model_rig.chains}
            for ref_root, model_root in chain_pairing.items():
                for ref_bone, model_bone in zip(ref_by_root[ref_root], model_by_root[model_root]):
                    mapping[ref_bone] = model_bone
        return HandMatch(mapping=mapping, score=score, pairs=pairs)

    if len(ref_rigs) == 2 and len(model_rigs) == 2:
        straight = build([(ref_rigs[0], model_rigs[0]), (ref_rigs[1], model_rigs[1])])
        swapped = build([(ref_rigs[0], model_rigs[1]), (ref_rigs[1], model_rigs[0])])
        best = straight if straight.score <= swapped.score else swapped
    else:
        # Fall back to greedy pairing on whatever is available.
        pairing: list[tuple[HandRig, HandRig]] = []
        remaining = list(model_rigs)
        for ref_rig in ref_rigs:
            if not remaining:
                break
            scored = [(_match_chains(ref_smd, ref_rig, model_smd, m)[1], i)
                      for i, m in enumerate(remaining)]
            scored.sort()
            pairing.append((ref_rig, remaining.pop(scored[0][1])))
        best = build(pairing)

    mapped = set(best.mapping)
    all_ref_bones: set[str] = set()
    for rig in ref_rigs:
        all_ref_bones |= rig.bones
    best.unmapped = sorted(all_ref_bones - mapped)
    return best


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

@dataclass
class HandNormalisation:
    """Outcome of normalising one model's hands."""
    model_name: str
    smd: SMD | None = None                       # the rebound hand mesh
    mapping: dict[str, str] = field(default_factory=dict)
    pairs: list[tuple[str, str]] = field(default_factory=list)
    score: float = 0.0
    unmapped: list[str] = field(default_factory=list)
    bone_renames: dict[str, str] = field(default_factory=dict)  # model bone -> reference bone
    replaced_keys: list[str] = field(default_factory=list)   # SMD keys swapped out
    retired_textures: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.smd is not None


def canonical_rename_map(match: HandMatch) -> dict[str, str]:
    """
    Invert a match into ``{model bone: reference bone}``.

    Renaming the *model* onto the reference naming — rather than renaming the
    reference hand onto each model — is what lets every model share one hand
    skeleton.  Source rigs disagree about names (``Bone_Lefthand`` in most CSO
    models, ``"Bone 04"`` in others), and a bone name is the only thing
    studiomdl merges on, so without canonicalisation each differently-named rig
    would contribute its own 34-bone duplicate of the same hand.

    Entries that would rename a bone onto a name already used by a *different*
    bone are dropped, so this can never collapse two bones into one.
    """
    inverted: dict[str, str] = {}
    for reference_bone, model_bone in match.mapping.items():
        inverted.setdefault(model_bone, reference_bone)
    return inverted


def safe_rename_map(mapping: dict[str, str], existing: set[str]) -> dict[str, str]:
    """
    Drop renames that would collide with an unrelated bone already in the
    skeleton, so a rename can never merge two distinct bones.
    """
    renamed_away = set(mapping)
    safe: dict[str, str] = {}
    taken: set[str] = set()
    for source, target in mapping.items():
        if target in taken:
            continue
        if target in existing and target not in renamed_away:
            continue
        safe[source] = target
        taken.add(target)
    return safe


def build_normalised_hand(reference_hand: SMD, texture: str | None = None) -> SMD:
    """
    Return the reference hand with its own root folded away and ids renumbered,
    ready to drop into any model whose hand bones have been canonicalised.

    Dropping the root leaves the forearms as roots, matching what the merger
    produces for the weapon meshes; the merger then injects a single identity
    ``Universal_Root`` across every mesh.  The result depends only on the
    reference hand, so it is byte-identical for every model — which is what
    lets the hands bodygroup collapse to one shared entry.
    """
    hand = deepcopy(reference_hand)

    rigs = detect_rigs(hand)
    claimed: set[str] = set()
    for rig in rigs:
        claimed |= rig.bones
    surplus = {n.name for n in hand.nodes if n.name not in claimed}
    if surplus:
        remove_bones(hand, surplus)
    renumber(hand)

    if texture:
        for tri in hand.triangles:
            tri.material = texture

    return hand


def hand_texture_names(smd: SMD) -> set[str]:
    """Distinct material names used by *smd*'s triangles."""
    return {tri.material for tri in smd.triangles}


def load_reference_hand(path: str | Path) -> tuple[SMD, list[HandRig]]:
    """Load the optimised hand mesh and detect its rigs."""
    smd = SMD.from_file(path)
    rigs = detect_rigs(smd)
    if not rigs:
        raise ValueError(
            f"No hand rig found in {path}: expected a bone with at least "
            f"{MIN_FINGER_CHAINS} finger chains of {MIN_CHAIN_LENGTH}+ bones."
        )
    return smd, rigs
