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
names), `--no-hands` / `--no-prune` / `--no-share-hands` (disable a pass),
`--keep-hitbox-bones`, `--dry-run`.

**Batch size is limited by the 127-bone budget, not by the tool.** The pistols
(~7 bones each beyond the shared 34-bone hand) fit 9 to a model; the larger CSO
weapons in `more_weapons` carry 20-80 bones each, so only ~5 fit. Run `analyze`
first — it reports the budget and names the models to drop.

Output layout: `<out>/<model>.qc`, `<out>/<source_model>/*.smd`,
`<out>/_shared/hand.smd`, flat `.BMP` textures, and `models.ini` mapping each
source model to its `pev_body` value and sequence indices.

### Pipeline modules

- `goldsource/pipeline.py` — orchestration (`run()`); everything else is a step
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
- **Normalised hand meshes come out byte-identical**, so the hands bodygroup
  collapses to one shared entry instead of one copy per model.