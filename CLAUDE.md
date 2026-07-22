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
- **Hitboxes do not pin bones** (they are inert on view models). A single
  `$hbox "root"` would otherwise keep a redundant root alive, giving `Bone01`
  two different parents across models — which studiomdl rejects, forcing the
  merger to duplicate the whole shared hand skeleton (96 bones becomes 150).
- **Normalised hand meshes come out byte-identical**, so the hands bodygroup
  collapses to one shared entry instead of one copy per model.