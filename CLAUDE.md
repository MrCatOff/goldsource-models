# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python project for working with GoldSource engine (Half-Life) weapon models. The workflow involves decompiling `.mdl` files into `.qc`/`.smd` source files, modifying them, and recompiling with StudioMDL.

## Environment

- Python 3.14 virtual environment at `.venv/`
- Activate: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt` (once a requirements file exists)

## Storage Layout

- `storage/decompiled/` — decompiled model source files (gitignored); each subdirectory is one weapon model containing:
  - `<model>.qc` — StudioMDL compilation script (model definition, sequences, attachments, hitboxes)
  - `*.smd` — reference meshes and per-animation SMD files in `<model>_anims/`
  - `*.BMP` — textures

## GoldSource Model Pipeline

Models are decompiled with **Crowbar** and recompiled with **StudioMDL** (from the Half-Life SDK). The `.qc` file drives compilation: it references SMD meshes, declares bodygroups, sequences (animations with sound events), attachments, and hitboxes.

Key `.qc` sequence event codes:
- `5001` — weapon fire effect
- `5004` — play sound (`weapons/<sound>.wav`)

## CLI

`python -m goldsource <subcommand>` merges many decompiled models into one model
with per-weapon submodels, unattended.

```bash
# full pipeline: normalise hands, prune bones, merge, compile
python -m goldsource merge storage/decompiled/pistols -o storage/build/pistols -n v_pistols --compile

python -m goldsource analyze storage/decompiled/pistols   # bone budget, writes nothing
python -m goldsource hands   storage/decompiled/pistols   # show the hand bone mapping
python -m goldsource compile storage/build/pistols/v_pistols.qc
```

Useful flags: `--exclude NAME` (drop a model), `--rename FIND=REPLACE` (sequence
names), `--no-hands` / `--no-prune` / `--no-share-hands` / `--no-pool-bones`
(disable a pass), `--keep-hitbox-bones`, `--dry-run`.

**One weapon submodel per model, by default.** A source model's bodygroups
mostly *split* one weapon into pieces that are all drawn together; a few carry a
real choice (a scope, a cycling LED strip). Every switchable group costs a
bodypart out of the 32 available and multiplies `pev_body`, so only the first
entry survives. Name the ones worth keeping:

```bash
python -m goldsource merge ... --keep-group v_ak47lor:led --keep-group v_m32:*
python -m goldsource merge ... --groups groups.json   # {"v_ak47lor": ["led"], "v_m32": "*"}
```

`--all-groups` keeps every group. Pieces of one weapon are still concatenated
into a single submodel wherever the 2048-vertex cap allows (`--vertex-budget`).

**Bone pooling is what sets the batch size.** Only one weapon is drawn at a
time, so unrelated weapons can share bone slots — merging costs the *largest*
model's bone count, not the sum. All 58 `more_weapons` models fit one 113-bone
skeleton; before pooling they needed ten separate builds. `--bone-target N`
lowers the pool ceiling, trading animation size for bones. Run `analyze` first.

Output layout: `<out>/<model>.qc`, `<out>/<source_model>/*.smd`,
`<out>/_shared/hand.smd`, flat `.BMP` textures, and `models.ini` mapping each
source model to its `pev_body` value and sequence indices.

### Pipeline modules

- `goldsource/pipeline.py` — orchestration (`run()`); everything else is a step
- `goldsource/bonepool.py` — shared weapon-bone slots and exact re-parenting
- `goldsource/hands.py` — hand rig detection and **geometric** bone matching
- `goldsource/skeleton.py` — transform-preserving bone removal / renumbering / grafting
- `goldsource/merger.py` — bodygroup + sequence merging, `pev_body` encoding
- `goldsource/compiler.py` — studiomdl invocation

### Invariants the tests enforce (`python -m pytest tests`)

- **Bone folding is exact.** Removing a bone bakes its transform into its
  children, so every surviving bone keeps its world transform in every frame.
  Animations are never resampled or approximated.
- **Hand bones are matched by bind-pose geometry, never by name.** The CSO rig
  numbers left-hand fingers in descending order (`Finger1`→`Bone21`) and
  right-hand fingers ascending (`Finger1`→`Bone31`); a name-based table would
  silently swap index and pinky on one side.
- **The forearm is picked by arm length, not by being the palm's parent.**
  `v_deagle` puts a near-coincident wrist bone directly above the palm (0.36
  units) with the real forearm one level higher (8.54, matching the reference's
  8.45). Binding to the direct parent would slide the forearm mesh ~8.5 units
  down the arm.
- **Model hand bones are renamed onto the reference naming, not vice versa.**
  Bone names are the only thing studiomdl merges on, so a rig that calls its
  palm `"Bone 04"` instead of `Bone_Lefthand` would otherwise contribute a
  second 34-bone copy of the same hand. Renames are collision-checked
  (`safe_rename_map`) so two bones can never collapse into one, and QC
  `$attachment`/`$hbox`/`$controller` bone references are renamed with them.
- **However many hand bodygroups a model has, it comes out with one.**
  `v_deagle` splits hands into separate `rhand`/`lhand` groups; since one mesh
  now covers both hands, leaving two groups would render it twice.
- Weapon bones from unrelated models *may* collide by name and get renamed
  apart (`v_ana` and `v_deagle` both use `Bone25`/`Bone50`) — that is correct.
  The invariant is that no *hand* bone is ever renamed apart.
- **Handedness is read in forearm-local space, never hand-local.** A rig mirrors
  the hand bone's own axes along with the geometry, so a left and a right hand
  give near-identical finger coordinates in hand-local space — the two side
  assignments score within 0.2% and the winner is noise. The forearm frame
  brings in the arm-to-hand relationship, where the mirroring lives. All
  pairings are scored as whole assignments; pairing greedily in reference order
  would hand a one-handed model to whichever reference rig came first.
- **A rig that is not a full pair of five-fingered hands gets a trimmed mesh.**
  An unmatched finger's geometry is rebound to the palm (it rides rigidly
  instead of articulating); an unmatched whole hand is dropped. Emitting it
  anyway would leave bones no animation drives — a frozen hand in the bind pose.
- **Duplicate `$bodygroup` names are made unique before merging.** The merger
  aligns groups by name via `bodygroup_by_name`, which returns only the first
  match, so duplicates were silently dropped with their meshes (24 of 58
  `more_weapons` models are affected; `v_ak47chimera` lost 17 of 19 submodels).
- **Models lacking a group share ONE blank entry, not one each.** `pev_body`
  is a mixed-radix product of every group's entry count, so per-model blanks
  multiply it past 32 bits — and studiomdl then dies with an access violation
  and no diagnostic. `_check_bodygroup_limits` guards both that and the
  32-bodypart cap.
- **The Euler gimbal threshold is 1e-13, not 1e-6.** The degenerate branch of
  `_extract_euler_zyx` discards the Z rotation, an error of order `cy`. Folding
  a long bone chain amplifies it by each joint's lever arm — a 3e-7 slip at the
  shoulder became 1e-5 units at the fingertips. Lowering the threshold took
  worst-case animation drift from 1.0e-05 to 5.0e-12.
- **Hitboxes do not pin bones** (they are inert on view models). A single
  `$hbox "root"` would otherwise keep a redundant root alive, giving `Bone01`
  two different parents across models — which studiomdl rejects, forcing the
  merger to duplicate the whole shared hand skeleton.
- **The shared hand mesh is re-posed onto each model's own hand bind.** The
  optimised hand is authored around the reference rig, but a model's animations
  drive *its* hand bones, which can sit far from there (v_ak47chimera's hands
  are ~90 units off, and the mismatch is non-rigid — a rigid fit still leaves
  6-28 units). Bound at the reference pose but animated to the model's, the mesh
  stretches away from the weapon: the forearm reads as an elongated bone and the
  hand detaches. `_repose_hand_to_model` moves every hand bone to the model's
  bind and carries its vertices along, so bind and animation agree (hand/weapon
  bind delta drops to 0). Every weapon still shows the one optimised hand
  *design*; models whose hands land in the same place still share an entry, but
  divergent ones each keep their own — one shared mesh for all is impossible
  without distorting exactly these models, which was the visible bug.
- **A near-complete rig is grown to the full hand, not trimmed.** A rig that
  maps four fingers instead of five would otherwise carry its own hand mesh
  trimmed of the fifth; the missing finger is injected back frozen (static at
  the reference bind pose, since the model never animates it) so the mesh is
  byte-identical to the full hand and one shared hand serves all 53 complete
  models. A hand the model *entirely* lacks — a one-handed weapon like
  `v_portal` or `v_rpg_remapped` — is left absent; a whole frozen hand would
  float beside the weapon. Completion only *adds* a bone, so every animated
  bone keeps its exact transform.
- **Bodypart index 0 belongs to the first model, or a blank.** A viewer opens
  every bodypart at index 0, so index 0 is the default view. Two failure modes:
  blank *last* makes a part group the first model lacks default to some *other*
  model's piece, stacking several weapons at once; blank *always first* pushes
  the first model's own piece to index 1, so a weapon split across many all-on
  parts (v_ak47chimera has 12) shows only its first fragment. So index 0 is the
  first model's piece when it has one, and a blank otherwise — the default shows
  exactly the first model, whole, and nothing else. A weapon split across N
  all-on parts stays N bodyparts (its ~2000-vertex pieces cannot merge under the
  2048 cap), but its single `pev_body` selects all of them together.
- **Re-parenting a bone is exact, so weapons can share bone slots.** Animation
  is stored parent-relative, so moving a bone under a new parent is only
  `local = parent_world⁻¹ @ world` re-solved per frame — verified at 6.2e-10
  over 1.16M transforms across all 58 models. That is what makes pooling legal:
  only one weapon is drawn at a time, so a slot can serve a different weapon in
  every sequence. `v_model4_30`, a hand-built 23-weapon model, does the same
  thing (70 of its 104 weapon bones are shared). Cost is animation *size*, not
  accuracy — a re-anchored bone stops inheriting its old joint's motion, so
  constant channels become time-varying and studiomdl's RLE gets less out of
  them. `plan_pool` therefore spends spare bone budget before it reshapes.
- **A model that claims a pool slot must hold every slot above it.** Otherwise
  that slot falls back to a different parent for this model, the merged model
  sees one bone name with two parents, and the merger renames them apart —
  undoing the sharing for every model on that slot (30 models went from 127
  bones to 812).
- **Re-parenting must check for cycles.** A bone the pool did not move may sit
  *under* one it did; re-parenting the moved bone onto it closes a loop and
  world resolution recurses forever. The moved bone is rooted instead.
- **Models with no hand rig get an inert copy of the anchor bone.** Without it
  their pooled weapon roots come out as roots while every other model's hang off
  the anchor — parents disagree, and the merger renames the slots apart for
  exactly the models with the least to share (113 bones became 150). The pose is
  copied from a *prepared* model, not the reference hand, so the grafted chain
  matches what pruning left the others with.
- **Bodygroups are renamed to one canonical `weapon`/`weapon_2`/… sequence.**
  The merger aligns groups across models by name and the sources do not agree —
  the same slot is variously `bodypart1`, `body`, `studio`, even `waepon`. Each
  spelling would become its own bodypart, and since `pev_body` is a mixed-radix
  product over *every* bodypart, 22 of them put it 3000x past the 32-bit
  ceiling. The hand is not always in a group named for it (`v_rpg_remapped`
  keeps it in `body`), so hand groups are found by which mesh they hold.
- **2048 vertices per submodel is a hard limit even in the bundled Sven Co-op
  studiomdl** — `--vertex-budget 4096` fails to compile. `v_ak47chimera` and
  `v_awpchimera` therefore need 12 submodels each, and those 8 extra bodyparts
  alone multiply `pev_body` by 3^8.
- **`--keep-animated-bones` and bone pooling work against each other.** The
  first keeps bones alive per model precisely because they move; the second
  exists to share them. Use `--no-pool-bones` with it.