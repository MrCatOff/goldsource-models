"""
StudioMDL invocation.

studiomdl resolves ``$cd`` / ``$cdtexture`` and every ``studio`` path relative
to the *current working directory*, not to the QC's location, so the compile
always runs with the QC's directory as cwd and is handed a bare filename.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STUDIOMDL = Path("bin") / "studiomdl.exe"


@dataclass
class CompileResult:
    """Outcome of one studiomdl run."""
    returncode: int
    stdout: str
    output_mdl: Path | None
    command: list[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.output_mdl is not None and self.output_mdl.exists()

    @property
    def crashed(self) -> bool:
        """
        True when studiomdl died on a Windows exception rather than exiting.

        It has fixed-size arrays it fills without bounds checks (bodyparts,
        bones, textures), so an oversized model takes it down with an access
        violation and no diagnostic at all.
        """
        return self.returncode not in range(0, 256)

    @property
    def failure_reason(self) -> str | None:
        if self.ok:
            return None
        if self.crashed:
            code = self.returncode & 0xFFFFFFFF
            known = {
                0xC0000005: "access violation",
                0xC00000FD: "stack overflow",
            }.get(code, "crash")
            return (
                f"studiomdl {known} (0x{code:08X}) — it exceeded an internal "
                f"limit and produced no diagnostic; check the bodygroup, bone "
                f"and texture counts reported above"
            )
        return f"studiomdl exited with code {self.returncode}"

    @property
    def warnings(self) -> list[str]:
        return [
            line.strip() for line in self.stdout.splitlines()
            if "warning" in line.lower() or "error" in line.lower()
        ]


def find_studiomdl(explicit: str | Path | None = None) -> Path | None:
    """
    Locate a studiomdl binary: the explicit path, then ``bin/studiomdl.exe``
    relative to the project root, then ``PATH``.
    """
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.exists() else None

    project_root = Path(__file__).resolve().parent.parent
    bundled = project_root / DEFAULT_STUDIOMDL
    if bundled.exists():
        return bundled

    on_path = shutil.which("studiomdl") or shutil.which("studiomdl.exe")
    return Path(on_path) if on_path else None


def compile_qc(
    qc_path: str | Path,
    studiomdl: str | Path | None = None,
    ignore_warnings: bool = False,
    keep_unused_bones: bool = False,
    timeout: int = 600,
) -> CompileResult:
    """
    Compile *qc_path* with studiomdl and return a :class:`CompileResult`.

    Raises :class:`FileNotFoundError` when no studiomdl binary can be found.
    """
    qc = Path(qc_path).resolve()
    if not qc.exists():
        raise FileNotFoundError(f"QC file not found: {qc}")

    binary = find_studiomdl(studiomdl)
    if binary is None:
        raise FileNotFoundError(
            "studiomdl not found. Pass --studiomdl <path> or place it at bin/studiomdl.exe."
        )

    command = [str(binary.resolve())]
    if ignore_warnings:
        command.append("-i")
    if keep_unused_bones:
        command.append("-k")
    command.append(qc.name)

    proc = subprocess.run(
        command,
        cwd=qc.parent,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )

    stdout = (proc.stdout or "") + (proc.stderr or "")

    # $modelname may include a path prefix; studiomdl writes it relative to cwd.
    modelname = _read_modelname(qc)
    output_mdl = (qc.parent / modelname) if modelname else None

    return CompileResult(
        returncode=proc.returncode,
        stdout=stdout,
        output_mdl=output_mdl,
        command=command,
    )


def _read_modelname(qc: Path) -> str | None:
    """Cheap scan for ``$modelname`` so we know which .mdl to look for."""
    try:
        for line in qc.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("$modelname"):
                parts = stripped.split(None, 1)
                if len(parts) == 2:
                    return parts[1].strip().strip('"')
    except OSError:
        pass
    return None
