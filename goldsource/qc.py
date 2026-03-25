"""
GoldSource QC (StudioMDL script) parser and writer.

Supports all standard GoldSrc commands documented at
https://the303.org/tutorials/gold_qc.htm plus Sven-Coop extensions.

Reference SMD usage
-------------------
Every model needs at least one $sequence. For static props the reference SMD
itself can be reused as the idle sequence.

Round-trip example
------------------
    qc = QC.from_file("v_ana/v_ana.qc")
    qc.modelname = "v_ana_new.mdl"
    qc.save("output/v_ana_new.qc")
    print(qc.to_string())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SequenceEvent:
    """A single { event … } block inside a $sequence."""
    event_type: int
    frame: int
    options: str = ""  # optional parameter string (without quotes)


@dataclass
class BodyGroupEntry:
    """One studio/blank line inside a $bodygroup block."""
    smd: str              # SMD filename (without .smd), or "" for blank
    reverse: bool = False
    scale: float | None = None

    @property
    def is_blank(self) -> bool:
        return self.smd == ""


@dataclass
class BodyGroup:
    """$bodygroup – a named set of swappable mesh alternatives."""
    name: str
    entries: list[BodyGroupEntry] = field(default_factory=list)


@dataclass
class Attachment:
    """$attachment – named point in bone-local space."""
    id: int
    bone: str
    x: float
    y: float
    z: float


@dataclass
class HitBox:
    """$hbox – axis-aligned box in bone-local space."""
    group: int
    bone: str
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float


@dataclass
class BoneController:
    """$controller – game-code controllable bone axis."""
    id: int          # 0–3, or 4 for mouth
    bone: str
    axis: str        # XR | YR | ZR | X | Y | Z
    min: float
    max: float


@dataclass
class TextureRenderMode:
    """$texrendermode – render flag for a specific texture."""
    texture: str
    mode: str        # masked | additive | flatshade | fullbright | chrome


@dataclass
class TextureGroup:
    """$texturegroup – swappable skin sets."""
    name: str
    skins: list[list[str]] = field(default_factory=list)
    # skins[0] = default textures, skins[1..n] = replacement sets


@dataclass
class Sequence:
    """$sequence – one named animation clip."""
    name: str
    smd_paths: list[str] = field(default_factory=list)
    # Up to 9 paths for blended sequences (3×3 grid); usually just one.

    fps: float | None = None
    loop: bool = False

    # Frame range trim
    frame_start: int | None = None
    frame_end: int | None = None

    # Per-sequence transforms
    origin: tuple[float, float, float] | None = None
    rotate: float | None = None
    scale: float | None = None

    # Blend controller
    blend_axis: str | None = None   # XR | YR | ZR | X | Y | Z
    blend_min: float | None = None
    blend_max: float | None = None

    # Motion extraction / zeroing
    motion_extract: list[str] = field(default_factory=list)  # LX LY LZ
    motion_zero: list[str] = field(default_factory=list)     # X Y Z

    # Activity tag
    activity: str | None = None
    activity_weight: int = 1

    events: list[SequenceEvent] = field(default_factory=list)


@dataclass
class QC:
    """
    Full representation of a GoldSource QC file.

    Create programmatically or load from a file::

        qc = QC.from_file("v_ana/v_ana.qc")
        qc = QC(modelname="mymodel.mdl", ...)

    Save back to disk::

        qc.save("output/mymodel.qc")
        text = qc.to_string()
    """

    # --- Basic directives ---
    modelname: str = ""
    cd: str = ""
    cdtexture: str = ""
    cliptotextures: bool = False
    scale: float = 1.0
    gamma: float | None = None

    # --- Positioning ---
    origin: tuple[float, float, float] | None = None
    origin_rotation: float | None = None
    eyeposition: tuple[float, float, float] | None = None

    # --- Model flags ---
    flags: int = 0

    # --- Bounding volumes ---
    bbox: tuple[float, float, float, float, float, float] | None = None
    cbox: tuple[float, float, float, float, float, float] | None = None

    # --- Body / mesh ---
    body: BodyGroupEntry | None = None   # $body (single mesh, no group)
    bodygroups: list[BodyGroup] = field(default_factory=list)

    # --- Skeleton ---
    attachments: list[Attachment] = field(default_factory=list)
    hboxes: list[HitBox] = field(default_factory=list)
    controllers: list[BoneController] = field(default_factory=list)
    rename_bones: dict[str, str] = field(default_factory=dict)
    mirror_bones: list[str] = field(default_factory=list)

    # --- Textures ---
    texturemodes: list[TextureRenderMode] = field(default_factory=list)
    texturegroups: list[TextureGroup] = field(default_factory=list)

    # --- Animations ---
    sequences: list[Sequence] = field(default_factory=list)

    # --- Includes ---
    includes: list[str] = field(default_factory=list)

    # --- Sven-Coop extensions ---
    keepallbones: bool = False
    keepbones: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> QC:
        """Parse a QC file from *path* and return a new :class:`QC` instance."""
        return _parse(Path(path).read_text(encoding="utf-8", errors="replace"))

    @classmethod
    def from_string(cls, text: str) -> QC:
        """Parse QC content from a string."""
        return _parse(text)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_string(self) -> str:
        """Render the QC back to its text representation."""
        out: list[str] = []

        def line(s: str = "") -> None:
            out.append(s)

        if self.modelname:
            line(f'$modelname "{self.modelname}"')
        if self.cd:
            line(f'$cd "{self.cd}"')
        if self.cdtexture:
            line(f'$cdtexture "{self.cdtexture}"')
        if self.cliptotextures:
            line("$cliptotextures")
        line(f"$scale {_fmt(self.scale)}")

        if self.gamma is not None:
            line(f"$gamma {_fmt(self.gamma)}")
        if self.eyeposition is not None:
            x, y, z = self.eyeposition
            line(f"$eyeposition {_fmt(x)} {_fmt(y)} {_fmt(z)}")
        if self.origin is not None:
            x, y, z = self.origin
            rot = f" {_fmt(self.origin_rotation)}" if self.origin_rotation is not None else ""
            line(f"$origin {_fmt(x)} {_fmt(y)} {_fmt(z)}{rot}")

        line()

        # body / bodygroups
        if self.body is not None:
            entry = self.body
            parts = ["$body studio", f'"{entry.smd}"']
            if entry.reverse:
                parts.append("reverse")
            if entry.scale is not None:
                parts.extend(["scale", _fmt(entry.scale)])
            line(" ".join(parts))
            line()

        for bg in self.bodygroups:
            line(f'$bodygroup "{bg.name}"')
            line("{")
            for e in bg.entries:
                if e.is_blank:
                    line("\tblank")
                else:
                    parts = [f'studio "{e.smd}"']
                    if e.reverse:
                        parts.append("reverse")
                    if e.scale is not None:
                        parts.extend(["scale", _fmt(e.scale)])
                    line(f"\t{' '.join(parts)}")
            line("}")

        line()
        line(f"$flags {self.flags}")
        line()

        # bbox / cbox
        if self.cbox is not None:
            line(f"$cbox {' '.join(_fmt(v) for v in self.cbox)}")
        if self.bbox is not None:
            line(f"$bbox {' '.join(_fmt(v) for v in self.bbox)}")

        # attachments
        if self.attachments:
            line()
            for a in self.attachments:
                line(f'$attachment {a.id} "{a.bone}" {_fmt(a.x)} {_fmt(a.y)} {_fmt(a.z)}')

        # hitboxes
        if self.hboxes:
            line()
            for h in self.hboxes:
                coords = " ".join(_fmt(v) for v in (h.x1, h.y1, h.z1, h.x2, h.y2, h.z2))
                line(f'$hbox {h.group} "{h.bone}" {coords}')

        # controllers
        if self.controllers:
            line()
            for c in self.controllers:
                line(f'$controller {c.id} "{c.bone}" {c.axis} {_fmt(c.min)} {_fmt(c.max)}')

        # texture modes
        for tm in self.texturemodes:
            line(f"$texrendermode {tm.texture} {tm.mode}")

        # texture groups
        for tg in self.texturegroups:
            line(f"$texturegroup {tg.name}")
            line("{")
            for skin in tg.skins:
                quoted = " ".join(f'"{t}"' for t in skin)
                line(f"\t{{ {quoted} }}")
            line("}")

        # rename / mirror bones
        for old, new in self.rename_bones.items():
            line(f'$renamebone "{old}" "{new}"')
        for bone in self.mirror_bones:
            line(f'$mirrorbone "{bone}"')

        # Sven-Coop
        if self.keepallbones:
            line("$keepallbones")
        for bone in self.keepbones:
            line(f'$keepbone "{bone}"')

        # includes
        for inc in self.includes:
            line(f'$include "{inc}"')

        # sequences
        if self.sequences:
            line()
        for seq in self.sequences:
            line(f'$sequence "{seq.name}" {{')
            for p in seq.smd_paths:
                line(f'\t"{p}"')

            for axis in seq.motion_extract:
                line(f"\t{axis}")
            for axis in seq.motion_zero:
                line(f"\t{axis}")

            if seq.blend_axis is not None:
                line(f"\tblend {seq.blend_axis} {_fmt(seq.blend_min)} {_fmt(seq.blend_max)}")
            if seq.frame_start is not None:
                line(f"\tframe {seq.frame_start} {seq.frame_end}")
            if seq.origin is not None:
                x, y, z = seq.origin
                line(f"\torigin {_fmt(x)} {_fmt(y)} {_fmt(z)}")
            if seq.rotate is not None:
                line(f"\trotate {_fmt(seq.rotate)}")
            if seq.scale is not None:
                line(f"\tscale {_fmt(seq.scale)}")
            if seq.activity is not None:
                line(f"\t{seq.activity} {seq.activity_weight}")

            for ev in seq.events:
                opts = f' "{ev.options}"' if ev.options else ""
                line(f"\t{{ event {ev.event_type} {ev.frame}{opts} }}")

            if seq.fps is not None:
                line(f"\tfps {_fmt(seq.fps)}")
            if seq.loop:
                line("\tloop")

            line("}")

        return "\n".join(out) + "\n"

    def save(self, path: str | Path) -> None:
        """Write the QC to *path*, creating parent directories if needed."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.to_string(), encoding="utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def sequence_by_name(self, name: str) -> Sequence | None:
        """Return the :class:`Sequence` with *name*, or ``None``."""
        for s in self.sequences:
            if s.name == name:
                return s
        return None

    def bodygroup_by_name(self, name: str) -> BodyGroup | None:
        """Return the :class:`BodyGroup` with *name*, or ``None``."""
        for bg in self.bodygroups:
            if bg.name == name:
                return bg
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt(v: float | None) -> str:
    """Format a float with up to 6 decimal places, stripping trailing zeros."""
    if v is None:
        return "0"
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r'"[^"]*"|//[^\n]*|/\*.*?\*/|[{}\S]+', re.DOTALL)


def _tokenize(text: str) -> list[str]:
    """Return all meaningful tokens; comments are discarded."""
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group()
        if tok.startswith("//") or tok.startswith("/*"):
            continue
        tokens.append(tok)
    return tokens


def _unquote(s: str) -> str:
    """Strip surrounding quotes from a token if present."""
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _TokenStream:
    def __init__(self, tokens: list[str]) -> None:
        self._t = tokens
        self._i = 0

    def peek(self) -> str | None:
        return self._t[self._i] if self._i < len(self._t) else None

    def next(self) -> str:
        tok = self._t[self._i]
        self._i += 1
        return tok

    def expect(self, value: str) -> None:
        tok = self.next()
        if tok != value:
            raise ValueError(f"Expected {value!r}, got {tok!r}")

    def at_end(self) -> bool:
        return self._i >= len(self._t)


def _parse(text: str) -> QC:
    tokens = _tokenize(text)
    ts = _TokenStream(tokens)
    qc = QC()

    _MOTION_EXTRACT = {"LX", "LY", "LZ"}
    _MOTION_ZERO = {"X", "Y", "Z"}
    _BLEND_AXES = {"XR", "YR", "ZR", "X", "Y", "Z"}

    while not ts.at_end():
        cmd = ts.next().lower()

        if cmd == "$modelname":
            qc.modelname = _unquote(ts.next())

        elif cmd == "$cd":
            qc.cd = _unquote(ts.next())

        elif cmd == "$cdtexture":
            qc.cdtexture = _unquote(ts.next())

        elif cmd == "$cliptotextures":
            qc.cliptotextures = True

        elif cmd == "$scale":
            qc.scale = float(ts.next())

        elif cmd == "$gamma":
            qc.gamma = float(ts.next())

        elif cmd == "$flags":
            qc.flags = int(ts.next())

        elif cmd == "$eyeposition":
            qc.eyeposition = (float(ts.next()), float(ts.next()), float(ts.next()))

        elif cmd == "$origin":
            x, y, z = float(ts.next()), float(ts.next()), float(ts.next())
            qc.origin = (x, y, z)
            # optional rotation follows if next token is a number
            nxt = ts.peek()
            if nxt is not None and _is_number(nxt):
                qc.origin_rotation = float(ts.next())

        elif cmd == "$bbox":
            qc.bbox = (
                float(ts.next()), float(ts.next()), float(ts.next()),
                float(ts.next()), float(ts.next()), float(ts.next()),
            )

        elif cmd == "$cbox":
            qc.cbox = (
                float(ts.next()), float(ts.next()), float(ts.next()),
                float(ts.next()), float(ts.next()), float(ts.next()),
            )

        elif cmd == "$body":
            _expect_word(ts, "studio")
            smd = _unquote(ts.next())
            reverse, scale = _parse_body_opts(ts)
            qc.body = BodyGroupEntry(smd=smd, reverse=reverse, scale=scale)

        elif cmd == "$bodygroup":
            name = _unquote(ts.next())
            bg = BodyGroup(name=name)
            ts.expect("{")
            while ts.peek() != "}":
                entry_cmd = ts.next().lower()
                if entry_cmd == "studio":
                    smd = _unquote(ts.next())
                    reverse, scale = _parse_body_opts(ts)
                    bg.entries.append(BodyGroupEntry(smd=smd, reverse=reverse, scale=scale))
                elif entry_cmd == "blank":
                    bg.entries.append(BodyGroupEntry(smd=""))
                else:
                    pass  # unknown line inside bodygroup
            ts.expect("}")
            qc.bodygroups.append(bg)

        elif cmd == "$attachment":
            aid = int(ts.next())
            bone = _unquote(ts.next())
            x, y, z = float(ts.next()), float(ts.next()), float(ts.next())
            qc.attachments.append(Attachment(id=aid, bone=bone, x=x, y=y, z=z))

        elif cmd == "$hbox":
            group = int(ts.next())
            bone = _unquote(ts.next())
            x1, y1, z1 = float(ts.next()), float(ts.next()), float(ts.next())
            x2, y2, z2 = float(ts.next()), float(ts.next()), float(ts.next())
            qc.hboxes.append(HitBox(group=group, bone=bone,
                                    x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2))

        elif cmd == "$controller":
            cid = int(ts.next())
            bone = _unquote(ts.next())
            axis = ts.next()
            cmin, cmax = float(ts.next()), float(ts.next())
            qc.controllers.append(BoneController(id=cid, bone=bone, axis=axis,
                                                  min=cmin, max=cmax))

        elif cmd == "$texrendermode":
            texture = ts.next()
            mode = ts.next()
            qc.texturemodes.append(TextureRenderMode(texture=texture, mode=mode))

        elif cmd == "$texturegroup":
            name = ts.next()
            tg = TextureGroup(name=name)
            ts.expect("{")
            while ts.peek() != "}":
                ts.expect("{")
                skin: list[str] = []
                while ts.peek() != "}":
                    skin.append(_unquote(ts.next()))
                ts.expect("}")
                tg.skins.append(skin)
            ts.expect("}")
            qc.texturegroups.append(tg)

        elif cmd == "$renamebone":
            old = _unquote(ts.next())
            new = _unquote(ts.next())
            qc.rename_bones[old] = new

        elif cmd == "$mirrorbone":
            qc.mirror_bones.append(_unquote(ts.next()))

        elif cmd == "$keepallbones":
            qc.keepallbones = True

        elif cmd in ("$keepbone", "$protected"):
            qc.keepbones.append(_unquote(ts.next()))

        elif cmd == "$include":
            qc.includes.append(_unquote(ts.next()))

        elif cmd == "$sequence":
            qc.sequences.append(_parse_sequence(ts))

        # silently skip unknown/obsolete commands
        else:
            pass

    return qc


def _parse_sequence(ts: _TokenStream) -> Sequence:
    name = _unquote(ts.next())
    seq = Sequence(name=name)

    _MOTION_EXTRACT = {"LX", "LY", "LZ"}
    _MOTION_ZERO = {"X", "Y", "Z"}
    _BLEND_AXES = {"XR", "YR", "ZR", "X", "Y", "Z"}

    # The first tokens after the name may be bare SMD paths (without braces)
    # followed by an optional block, OR everything may be inside a block.
    if ts.peek() != "{":
        # Inline path(s) before the block
        while ts.peek() is not None and ts.peek() not in ("{", "}") and not ts.peek().startswith("$"):
            tok = ts.peek()
            if tok in _MOTION_EXTRACT or tok in _MOTION_ZERO or tok in _BLEND_AXES:
                break
            if tok.lower() in ("fps", "loop", "frame", "origin", "rotate",
                                "scale", "blend", "activity"):
                break
            seq.smd_paths.append(_unquote(ts.next()))

    if ts.peek() == "{":
        ts.next()  # consume "{"
        while ts.peek() != "}":
            tok = ts.next()
            tl = tok.lower()

            if tok.startswith('"') or (not tok.startswith("{") and _looks_like_path(tok)):
                seq.smd_paths.append(_unquote(tok))

            elif tl == "fps":
                seq.fps = float(ts.next())

            elif tl == "loop":
                seq.loop = True

            elif tl == "frame":
                seq.frame_start = int(ts.next())
                seq.frame_end = int(ts.next())

            elif tl == "origin":
                seq.origin = (float(ts.next()), float(ts.next()), float(ts.next()))

            elif tl == "rotate":
                seq.rotate = float(ts.next())

            elif tl == "scale":
                seq.scale = float(ts.next())

            elif tl == "blend":
                nxt = ts.peek()
                if nxt is not None and nxt.upper() in {"XR", "YR", "ZR", "X", "Y", "Z"}:
                    seq.blend_axis = ts.next()
                    seq.blend_min = float(ts.next())
                    seq.blend_max = float(ts.next())

            elif tok.upper() in _MOTION_EXTRACT:
                seq.motion_extract.append(tok.upper())

            elif tok.upper() in _MOTION_ZERO:
                seq.motion_zero.append(tok.upper())

            elif tok == "{":
                # event block: { event <type> <frame> ["options"] }
                inner = ts.next().lower()
                if inner == "event":
                    etype = int(ts.next())
                    eframe = int(ts.next())
                    opts = ""
                    if ts.peek() and ts.peek() != "}":
                        opts = _unquote(ts.next())
                    ts.expect("}")
                    seq.events.append(SequenceEvent(event_type=etype,
                                                    frame=eframe,
                                                    options=opts))
                else:
                    # skip unknown inner block
                    depth = 1
                    while depth > 0:
                        t = ts.next()
                        if t == "{":
                            depth += 1
                        elif t == "}":
                            depth -= 1

            elif tok.startswith("ACT_") or tok.startswith("act_"):
                seq.activity = tok.upper()
                seq.activity_weight = int(ts.next())

            # unknown tokens inside sequence block are silently skipped

        ts.expect("}")

    return seq


def _parse_body_opts(ts: _TokenStream) -> tuple[bool, float | None]:
    """Consume optional 'reverse' and 'scale #' tokens after a studio entry."""
    reverse = False
    scale = None
    while ts.peek() in ("reverse", "scale"):
        opt = ts.next().lower()
        if opt == "reverse":
            reverse = True
        elif opt == "scale":
            scale = float(ts.next())
    return reverse, scale


def _expect_word(ts: _TokenStream, word: str) -> None:
    tok = ts.next()
    if tok.lower() != word:
        raise ValueError(f"Expected '{word}', got {tok!r}")


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _looks_like_path(tok: str) -> bool:
    """Heuristic: token looks like an SMD file path (not a sub-command)."""
    lower = tok.lower()
    keywords = {
        "fps", "loop", "frame", "origin", "rotate", "scale", "blend",
        "lx", "ly", "lz", "x", "y", "z", "xr", "yr", "zr",
        "reverse", "blank", "studio",
    }
    if lower in keywords:
        return False
    if lower.startswith("act_"):
        return False
    if lower.startswith("$"):
        return False
    # paths typically contain letters, digits, underscores, slashes, brackets
    return bool(re.search(r'[a-zA-Z0-9_/\\]', tok))
