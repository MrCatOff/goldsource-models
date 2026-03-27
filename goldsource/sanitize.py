"""
Utilities for sanitising non-ASCII filenames inside a decompiled model directory.

Decompilers (e.g. Crowbar) sometimes produce files whose names contain characters
from the original source encoding (Korean, Japanese, etc.).  StudioMDL and the
merger pipeline require purely ASCII filenames.

:func:`sanitize_directory` renames every non-ASCII file it finds and patches all
text references to those files inside ``.smd`` and ``.qc`` files in the same
directory tree.
"""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize_directory(directory: str | Path) -> dict[str, str]:
    """
    Rename all non-ASCII files under *directory* and update text references.

    Returns a mapping ``{old_filename: new_filename}`` for every file that was
    renamed (bare names only, no directory component).  Returns an empty dict if
    nothing needed renaming.

    Algorithm
    ---------
    1.  Walk the tree and collect every file whose name contains a non-ASCII
        character.
    2.  Generate a unique ASCII replacement name for each (see :func:`_ascii_stem`).
    3.  Patch every ``.smd`` and ``.qc`` file in the tree, replacing occurrences
        of the old bare filename with the new one (case-insensitive match so that
        ``Foo.BMP`` and ``foo.bmp`` are both caught).
    4.  Rename the physical files.
    """
    root = Path(directory)

    # --- 1. Collect non-ASCII files -----------------------------------------
    non_ascii: list[Path] = [
        p for p in root.rglob("*")
        if p.is_file() and not p.name.isascii()
    ]
    if not non_ascii:
        return {}

    # --- 2. Build old→new name map (bare names, collision-free) -------------
    existing_names: set[str] = {
        p.name.lower()
        for p in root.rglob("*")
        if p.is_file() and p.name.isascii()
    }
    rename_map: dict[str, str] = {}          # old_bare_name → new_bare_name
    path_map:   dict[Path, Path] = {}        # old_path     → new_path

    for old_path in non_ascii:
        new_name = _unique_name(old_path.name, existing_names)
        rename_map[old_path.name] = new_name
        path_map[old_path] = old_path.with_name(new_name)
        existing_names.add(new_name.lower())

    # --- 3. Patch text references in .smd and .qc files ---------------------
    text_files = list(root.rglob("*.smd")) + list(root.rglob("*.qc"))
    for tf in text_files:
        _patch_text_file(tf, rename_map)

    # --- 4. Rename files -----------------------------------------------------
    for old_path, new_path in path_map.items():
        old_path.rename(new_path)

    return rename_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ascii_stem(name: str) -> str:
    """
    Strip non-ASCII characters from a filename stem, replace spaces with
    underscores, and collapse runs of underscores.  Returns an empty string
    if nothing ASCII remains.
    """
    ascii_only = "".join(c if c.isascii() else "" for c in name)
    ascii_only = ascii_only.replace(" ", "_")
    ascii_only = re.sub(r"_+", "_", ascii_only).strip("_")
    return ascii_only


def _unique_name(original: str, taken: set[str]) -> str:
    """
    Return an ASCII filename derived from *original* that does not collide with
    any name already in *taken* (case-insensitive comparison).

    If stripping non-ASCII yields an empty stem, falls back to ``noname``.
    """
    stem = Path(original).stem
    ext  = Path(original).suffix  # includes the dot, e.g. ".BMP"

    base = _ascii_stem(stem) or "noname"
    candidate = base + ext
    if candidate.lower() not in taken:
        return candidate

    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        if candidate.lower() not in taken:
            return candidate
        counter += 1


def _patch_text_file(path: Path, rename_map: dict[str, str]) -> None:
    """
    Replace every occurrence of each key in *rename_map* with its value inside
    *path*.  The match is case-insensitive on the old name so that variations
    in capitalisation are also caught.  The file is only rewritten when at
    least one replacement was made.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    changed = False
    for old_name, new_name in rename_map.items():
        pattern = re.compile(re.escape(old_name), re.IGNORECASE)
        new_text, n = pattern.subn(new_name, text)
        if n:
            text = new_text
            changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
