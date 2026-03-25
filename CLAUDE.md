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