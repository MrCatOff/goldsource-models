"""
GoldSource SMD (StudioMDL Data) parser and writer.

SMD v1 format sections:
  - nodes:    bone hierarchy  (id "name" parent_id)
  - skeleton: per-frame bone transforms  (bone_id tx ty tz rx ry rz)
  - triangles: mesh data  (texture / 3 × vertex lines)
               vertex line: bone_id x y z nx ny nz u v

Reference (mesh) SMDs contain all three sections.
Animation SMDs contain only nodes + skeleton (triangles section is absent or empty).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A single bone in the skeleton hierarchy."""
    id: int
    name: str
    parent_id: int  # -1 for root bones


@dataclass
class BoneTransform:
    """Position and Euler-angle rotation for one bone at one frame."""
    bone_id: int
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float


@dataclass
class SkeletonFrame:
    """All bone transforms for a single animation frame."""
    time: int
    bones: list[BoneTransform] = field(default_factory=list)


@dataclass
class Vertex:
    """A single mesh vertex with its parent bone, position, normal, and UV."""
    bone_id: int
    x: float
    y: float
    z: float
    nx: float
    ny: float
    nz: float
    u: float
    v: float


@dataclass
class Triangle:
    """Three vertices that share a texture/material name."""
    material: str
    v0: Vertex
    v1: Vertex
    v2: Vertex

    @property
    def vertices(self) -> tuple[Vertex, Vertex, Vertex]:
        return self.v0, self.v1, self.v2


@dataclass
class SMD:
    """
    Full representation of an SMD file.

    Create programmatically or load from a file::

        smd = SMD.from_file("v_ana/ref_Anaconda.smd")
        smd = SMD(version=1, nodes=[...], skeleton=[...], triangles=[...])

    Save back to disk::

        smd.save("output/ref_Anaconda.smd")
        text = smd.to_string()
    """

    version: int = 1
    nodes: list[Node] = field(default_factory=list)
    skeleton: list[SkeletonFrame] = field(default_factory=list)
    triangles: list[Triangle] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> SMD:
        """Parse an SMD file from *path* and return a new :class:`SMD` instance."""
        return _parse(Path(path).read_text(encoding="utf-8", errors="replace"))

    @classmethod
    def from_string(cls, text: str) -> SMD:
        """Parse SMD content from a string."""
        return _parse(text)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_string(self) -> str:
        """Render the SMD back to its text representation."""
        lines: list[str] = []

        lines.append(f"version {self.version}")

        # nodes
        lines.append("nodes")
        for n in self.nodes:
            lines.append(f'  {n.id} "{n.name}" {n.parent_id}')
        lines.append("end")

        # skeleton
        lines.append("skeleton")
        for frame in self.skeleton:
            lines.append(f"  time {frame.time}")
            for b in frame.bones:
                lines.append(
                    f"    {b.bone_id}"
                    f" {b.tx:f} {b.ty:f} {b.tz:f}"
                    f" {b.rx:f} {b.ry:f} {b.rz:f}"
                )
        lines.append("end")

        # triangles (only written when present)
        if self.triangles:
            lines.append("triangles")
            for tri in self.triangles:
                lines.append(tri.material)
                for v in tri.vertices:
                    lines.append(
                        f"  {v.bone_id}"
                        f" {v.x:f} {v.y:f} {v.z:f}"
                        f" {v.nx:f} {v.ny:f} {v.nz:f}"
                        f" {v.u:f} {v.v:f}"
                    )
            lines.append("end")

        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> None:
        """Write the SMD to *path*, creating parent directories if needed."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.to_string(), encoding="utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_animation(self) -> bool:
        """True when the SMD has no triangle data (i.e. is an animation SMD)."""
        return not self.triangles

    @property
    def frame_count(self) -> int:
        """Number of skeleton frames."""
        return len(self.skeleton)

    def node_by_id(self, bone_id: int) -> Node | None:
        """Return the :class:`Node` with the given *bone_id*, or ``None``."""
        for n in self.nodes:
            if n.id == bone_id:
                return n
        return None

    def node_by_name(self, name: str) -> Node | None:
        """Return the :class:`Node` with *name*, or ``None``."""
        for n in self.nodes:
            if n.name == name:
                return n
        return None


# ---------------------------------------------------------------------------
# Internal parser
# ---------------------------------------------------------------------------

def _parse(text: str) -> SMD:
    lines = iter(text.splitlines())

    def next_token_line() -> list[str] | None:
        """Return the next non-empty, non-comment line as a token list."""
        for raw in lines:
            stripped = raw.strip()
            if stripped and not stripped.startswith("//"):
                return stripped.split()
        return None

    smd = SMD()

    tokens = next_token_line()
    if tokens is None or tokens[0] != "version":
        raise ValueError("SMD file does not start with 'version'")
    smd.version = int(tokens[1])

    while True:
        tokens = next_token_line()
        if tokens is None:
            break

        section = tokens[0]

        if section == "nodes":
            _parse_nodes(smd, next_token_line, lines)
        elif section == "skeleton":
            _parse_skeleton(smd, next_token_line, lines)
        elif section == "triangles":
            _parse_triangles(smd, next_token_line, lines)
        # unknown sections are silently skipped

    return smd


def _parse_nodes(
    smd: SMD,
    next_token_line,  # callable
    lines,
) -> None:
    while True:
        tokens = next_token_line()
        if tokens is None or tokens[0] == "end":
            break
        # format:  id "name" parent_id
        bone_id = int(tokens[0])
        # name may contain spaces; re-join and strip quotes
        name_raw = " ".join(tokens[1:-1])
        name = name_raw.strip('"')
        parent_id = int(tokens[-1])
        smd.nodes.append(Node(id=bone_id, name=name, parent_id=parent_id))


def _parse_skeleton(
    smd: SMD,
    next_token_line,
    lines,
) -> None:
    current_frame: SkeletonFrame | None = None

    while True:
        tokens = next_token_line()
        if tokens is None or tokens[0] == "end":
            break

        if tokens[0] == "time":
            current_frame = SkeletonFrame(time=int(tokens[1]))
            smd.skeleton.append(current_frame)
        else:
            if current_frame is None:
                raise ValueError("Bone transform found before any 'time' line")
            bone_id, tx, ty, tz, rx, ry, rz = (
                int(tokens[0]),
                float(tokens[1]),
                float(tokens[2]),
                float(tokens[3]),
                float(tokens[4]),
                float(tokens[5]),
                float(tokens[6]),
            )
            current_frame.bones.append(
                BoneTransform(bone_id=bone_id, tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz)
            )


def _parse_triangles(
    smd: SMD,
    next_token_line,
    lines,
) -> None:
    while True:
        tokens = next_token_line()
        if tokens is None or tokens[0] == "end":
            break

        # First token line after section open (or after previous triangle) is
        # the material name — it may contain spaces, so re-join.
        material = " ".join(tokens)

        verts: list[Vertex] = []
        for _ in range(3):
            vtokens = next_token_line()
            if vtokens is None:
                raise ValueError("Unexpected end of file inside triangle")
            bone_id = int(vtokens[0])
            x, y, z = float(vtokens[1]), float(vtokens[2]), float(vtokens[3])
            nx, ny, nz = float(vtokens[4]), float(vtokens[5]), float(vtokens[6])
            u, v = float(vtokens[7]), float(vtokens[8])
            verts.append(Vertex(bone_id=bone_id, x=x, y=y, z=z,
                                nx=nx, ny=ny, nz=nz, u=u, v=v))

        smd.triangles.append(Triangle(material=material,
                                      v0=verts[0], v1=verts[1], v2=verts[2]))
