# goldsource — GoldSource weapon-model merger

Merge many decompiled GoldSource / Half-Life (CS 1.6) weapon view-models into a
single `.mdl` with one submodel per weapon — fully unattended. It normalises
every model onto one shared hand, prunes and pools bones so unrelated weapons
share bone slots, packs the meshes, and compiles with studiomdl.

The main reason to do this: CS 1.6 can only precache ~512 models. Merging N
weapons into one file frees N−1 precache slots.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

Run it as a module (examples below use the venv Python directly so no activation
is needed):

```bash
.\.venv\Scripts\python.exe -m goldsource <subcommand> ...
```

`bin/studiomdl.exe` (Sven Co-op build, enforces the real CS 1.6 limits) and
`storage/hands/default_hand.smd` (the optimised replacement hand) ship with the
repo, so the defaults work out of the box.

## Subcommands

| command | what it does |
|---|---|
| `merge`   | the whole pipeline: normalise hands → prune/pool bones → pack → merge → (optionally) compile |
| `analyze` | report the merged bone budget and conflicts, **writes nothing** |
| `hands`   | show how the reference hand maps onto each model's bones |
| `compile` | run studiomdl on an existing `.qc` |

Global flag `-q` / `--quiet` (before the subcommand) silences progress output.

## Quick start

```bash
# Preview first — bone budget and hand matches, no files written
.\.venv\Scripts\python.exe -m goldsource analyze storage/decompiled/pistols

# Merge all pistols into one model and compile it
.\.venv\Scripts\python.exe -m goldsource merge storage/decompiled/pistols -o storage/build/pistols -n v_pistols --compile
```

`inputs` can be individual model directories **or** a parent directory that
contains them — every subfolder with exactly one `.qc` is picked up.

Output layout: `<out>/<name>.qc`, `<out>/<model>/*.smd`, shared hand under
`<out>/_shared/`, flat `.BMP` textures, and `models.ini` mapping each source
weapon to its `pev_body` value and sequence indices.

## Common tasks

```bash
# Skip specific weapons (repeatable)
... merge storage/decompiled/more_weapons -o out -n v_pack --exclude v_ak47chimera --exclude v_janus7

# Merge only chosen weapons (list them instead of a parent dir)
... merge storage/decompiled/more_weapons/v_svd storage/decompiled/more_weapons/v_f2000 -o out -n v_pack

# Rename sequences on the way in (repeatable)
... merge ... --rename fire=shoot --rename idle1=idle

# Compile an already-merged QC (e.g. after hand-editing it)
.\.venv\Scripts\python.exe -m goldsource compile storage/build/pistols/v_pistols.qc

# See the hand bone mapping for a model
.\.venv\Scripts\python.exe -m goldsource hands storage/decompiled/pistols/v_deagle
```

### Reducing geometry (heavy weapons)

Some CSO weapons carry ~20k triangles, which forces many submodels. `--decimate`
reduces weapon meshes (lossy; the hand and animations are left untouched):

```bash
# Halve every weapon mesh
... merge ... --decimate 0.5

# Global 0.5, but crush one heavy model into a single submodel
... merge ... --decimate 0.5 --decimate-model v_ak47chimera=0.12
```

Start at `0.5`. Lower ratios cut more but look blockier — see
`storage/build/DECIMATE.md` for a per-ratio table and a Blender alternative.

### Keeping switchable bodygroups

By default each model contributes **one** weapon submodel; switchable groups
(scopes, glowing strips) collapse to their first entry. Keep the ones you want:

```bash
... merge ... --keep-group v_ak47lor:led --keep-group v_m32:*      # :* = all of a model's
... merge ... --groups groups.json                                 # {"v_ak47lor": ["led"], "v_m32": "*"}
... merge ... --all-groups                                         # keep every switchable group
```

## Useful flags

| flag | effect |
|---|---|
| `-o, --output DIR` | output directory (required for `merge`) |
| `-n, --name NAME` | output model name (`.mdl` added if missing) |
| `--compile` | run studiomdl on the result |
| `--studiomdl EXE` | use a specific studiomdl (e.g. one with higher limits) |
| `--exclude NAME` | skip a model (repeatable) |
| `--decimate RATIO` | reduce weapon meshes to RATIO of their vertices (lossy) |
| `--decimate-model M=RATIO` | override `--decimate` for one model |
| `--vertex-budget N` | vertices allowed per submodel (studiomdl MAXSTUDIOVERTS, default 2048) |
| `--bone-target N` | how far the bone pool may grow before re-anchoring (default 127) |
| `--no-pool-bones` | don't share weapon bone slots (costs the *sum* of every model's bones) |
| `--keep-animated-bones` | don't fold moving bones — use if studiomdl reports a sequence over 64K |
| `--shared-hand` | one reference-posed hand for all (less geometry, but stretches off-pose models) |
| `--no-hands` | keep each model's original hands |
| `--dry-run` | analyse without writing files |

Run `... merge --help` for the full list.

## What limits how many weapons fit in one file

studiomdl enforces several hard caps; the pipeline works around them but they
still bound a single `.mdl`:

- **127 bones.** Bone pooling lets unrelated weapons share slots, so a merge
  costs the *largest* model's bone count, not the sum — but a batch of heavy
  weapons can still hit the ceiling. `analyze` reports the budget first.
- **2048 vertices per submodel.** A weapon over this is split into several
  submodels (or use `--decimate`).
- **~80–90 textures per model.** In practice this is the ceiling for packing
  *many* small weapons — roughly **30 weapons per file** — since each weapon
  brings a few unique skins. Beyond it studiomdl aborts with an access
  violation and no diagnostic.
- **64 KB per compiled sequence.** A few large weapons trip this; add
  `--keep-animated-bones` for those.

When a batch is too big, split it and compile several files. The heaviest models
(the 12-piece chimeras, 78-bone or 64K-sequence weapons) don't merge cleanly —
keep those as their own `.mdl`.

## Development

```bash
.\.venv\Scripts\python.exe -m pytest tests      # 88 tests
```

See `CLAUDE.md` for the pipeline's module layout and the invariants the tests
enforce (exact bone folding, geometric hand matching, bone pooling, and the
studiomdl limits above).
